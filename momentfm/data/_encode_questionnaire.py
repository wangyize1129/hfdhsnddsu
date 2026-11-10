"""Encode frozen BERT question/answer pairs before starting MOMENT training."""
from pathlib import Path

import argparse
import hashlib
import json
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def read_questionnaires(path):
    samples = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    ids, patterns, pairs_by_subject = [], [], []
    reference = None
    for sample in samples:
        sid = sample["subject_id"]
        if not isinstance(sid, str) or len(sid) != 3 or not sid.isdigit():
            raise ValueError("subject_id must be a three-digit string")
        pairs = sample["qa_pairs"]
        if len(pairs) != 30:
            raise ValueError(f"{sid}: expected 30 question/answer pairs")
        questions = [pair["question"] for pair in pairs]
        if any(not isinstance(q, str) or not q.strip() for q in questions):
            raise ValueError(f"{sid}: invalid question text")
        if reference is None:
            reference = questions
        if questions != reference:
            raise ValueError(f"{sid}: inconsistent question ordering/text")
        answers = [pair["answer"] for pair in pairs]
        if any(type(a) is not int or a not in (0, 1) for a in answers):
            raise ValueError(f"{sid}: answers must be integer 0 or 1")
        ids.append(sid)
        patterns.append("".join(map(str, answers)))
        pairs_by_subject.append([(q, str(a)) for q, a in zip(questions, answers)])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Empty questionnaire data or duplicate subject IDs")
    return ids, patterns, pairs_by_subject


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bert-model", default="google-bert/bert-base-uncased")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    started = time.time()
    ids, patterns, samples = read_questionnaires(args.input_json)
    unique_pairs = list(dict.fromkeys(pair for sample in samples for pair in sample))
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    lengths = tokenizer([q for q, a in unique_pairs], text_pair=[a for q, a in unique_pairs],
                        truncation=False, padding=False)["input_ids"]
    if max(map(len, lengths)) > args.max_length:
        raise ValueError("A question/answer pair exceeds max_length; refusing to truncate")
    model = AutoModel.from_pretrained(args.bert_model).to(args.device).eval()
    model.requires_grad_(False)
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(unique_pairs), args.batch_size):
            chunk = unique_pairs[start:start + args.batch_size]
            inputs = tokenizer([q for q, a in chunk], text_pair=[a for q, a in chunk],
                               padding="max_length", max_length=args.max_length,
                               truncation=False, return_tensors="pt").to(args.device)
            result = model(**inputs).last_hidden_state[:, 0, :]
            vectors.append(result.float().cpu().numpy())
    unique_vectors = np.concatenate(vectors)
    pair_index = {pair: i for i, pair in enumerate(unique_pairs)}
    embeddings = np.stack([unique_vectors[[pair_index[p] for p in sample]].mean(axis=0)
                           for sample in samples]).astype(np.float32)
    if embeddings.shape != (len(ids), 768) or not np.isfinite(embeddings).all():
        raise RuntimeError(f"Invalid embeddings: {embeddings.shape}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "text_features.npz"
    with (output / "text_features.tmp").open("wb") as handle:
        np.savez_compressed(handle, subject_ids=np.asarray(ids), embeddings=embeddings,
                            answer_patterns=np.asarray(patterns))
    (output / "text_features.tmp").replace(cache_path)
    source_file = Path(args.bert_model) / "source.json"
    source = json.loads(source_file.read_text()) if source_file.is_file() else {}
    metadata = {
        "model": source.get("repo", args.bert_model), "load_path": args.bert_model,
        "model_commit": source.get("commit", getattr(model.config, "_commit_hash", None)),
        "pretrained_files": source.get("files", {}),
        "input_sha256": hashlib.sha256(Path(args.input_json).read_bytes()).hexdigest(),
        "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "pooling": "mean of 30 last-layer CLS question/answer embeddings",
        "pair_format": "[CLS] question [SEP] 0/1 [SEP]",
        "max_length": args.max_length, "subjects": len(ids), "dimension": 768,
        "unique_qa_pairs_encoded": len(unique_pairs), "unique_questionnaires": len(set(patterns)),
        "dtype": "float32", "encoder_frozen": True,
        "seconds": time.time() - started, "torch": torch.__version__,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"event": "text_encoding_complete", **metadata}), flush=True)


if __name__ == "__main__":
    main()
