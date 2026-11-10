"""Offline BioLORD targets for masked affirmative/negative spans."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / 'pretrained' / 'biolord'


def local_model_files(model_dir):
    """Validate an actual directory and fingerprint local model/tokenizer files."""
    directory = Path(model_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Local BioLORD directory not found: {directory}. Set CONFIG.text_model to your local weights directory.")
    if not (directory / 'config.json').is_file():
        raise FileNotFoundError(directory / 'config.json')
    if not any((directory / name).is_file() for name in
               ('model.safetensors', 'model.safetensors.index.json', 'pytorch_model.bin', 'pytorch_model.bin.index.json')):
        raise FileNotFoundError(f"No local model weights in {directory}")
    if not any((directory / name).is_file() for name in ('tokenizer.json', 'vocab.txt')):
        raise FileNotFoundError(f"No local MPNet tokenizer in {directory}; expected tokenizer.json or vocab.txt")
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in {'.json', '.txt', '.safetensors', '.bin', '.model'}:
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            files[path.name] = digest.hexdigest()
    return directory, files


def load_local_encoder(model_dir):
    directory, files = local_model_files(model_dir)
    # These flags precede the transformers import. No Hub resolution is allowed.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    safetensors = (directory / 'model.safetensors').is_file() or (directory / 'model.safetensors.index.json').is_file()
    model = AutoModel.from_pretrained(str(directory), local_files_only=True,
                                     trust_remote_code=False, use_safetensors=safetensors)
    return tokenizer, model, directory, files


def prepare(source):
    samples = json.loads(Path(source).read_text(encoding='utf-8-sig'))
    subjects, answers, rewritten = [], [], []
    variants = [dict() for _ in range(30)]
    verbs = {'experiences': 'experience', 'has': 'have', 'feels': 'feel',
             'loses': 'lose', 'sees': 'see', 'finds': 'find', 'talks': 'talk',
             'sweats': 'sweat', 'believes': 'believe'}
    for sample in samples:
        sid, texts = sample['subject_id'], sample['statements']
        if not isinstance(sid, str) or len(sid) != 3 or not sid.isdigit() or len(texts) != 30:
            raise ValueError('Expected three-digit IDs and 30 statements')
        subjects.append(sid)
        row, bits = [], []
        for q, text in enumerate(texts):
            bit = int('does not' not in text)
            if bit:
                if text.startswith('The participant regularly gets up'):
                    text = text.replace('The participant regularly gets up', 'The participant does regularly get up', 1)
                elif not text.startswith('The participant does '):
                    match = re.match(r'The participant (\w+) ', text)
                    if not match or match[1] not in verbs:
                        raise ValueError(f'Unsupported affirmative: {text}')
                    text = text.replace('The participant '+match[1], 'The participant does '+verbs[match[1]], 1)
            if bit in variants[q] and variants[q][bit] != text:
                raise ValueError(f'Multiple variants for question {q+1}')
            variants[q][bit] = text
            row.append(text)
            bits.append(bit)
        answers.append(bits)
        rewritten.append({'subject_id': sid, 'statements': row})
    if len(subjects) != len(set(subjects)) or any(set(v) != {0, 1} for v in variants):
        raise ValueError('Duplicate IDs or missing binary variants')
    # One answer-independent prompt per question, shared by every subject.
    prompts = [re.sub(r'\bdoes not\b|\bdoes\b', '<MASK>', v[1]) for v in variants]
    return rewritten, np.asarray(answers, dtype=np.int64), variants, prompts


def encode_spans(tokenizer, model, texts, spans, device, max_length, return_tensor=False):
    import torch
    from transformers.utils import logging
    logging.set_verbosity_error()
    logging.disable_progress_bar()
    lengths = tokenizer(texts, truncation=False, padding=False)['input_ids']
    if any(len(x) > max_length for x in lengths):
        raise ValueError('Text exceeds max_length; refusing truncation')
    batch = tokenizer(texts, padding=True, truncation=False, return_offsets_mapping=True, return_tensors='pt')
    offsets = batch.pop('offset_mapping')
    masks = []
    for row, (start, end) in zip(offsets, spans):
        masks.append((row[:, 1] > start) & (row[:, 0] < end) & (row[:, 1] > row[:, 0]))
    mask = torch.stack(masks).to(device)
    if not mask.any(dim=1).all():
        raise ValueError('Failed to locate target span')
    with torch.inference_mode():
        hidden = model(**{k: v.to(device) for k, v in batch.items()}).last_hidden_state.float()
        pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return pooled if return_tensor else pooled.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default=str(DEFAULT_MODEL_DIR), help='Local BioLORD directory; network loading is disabled')
    p.add_argument('--device', default='auto', help='Device is selected automatically: CUDA when available, otherwise CPU')
    p.add_argument('--max-length', type=int, required=True)
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, answers, variants, prompts = prepare(args.input)
    (output/'questionnaire_explicit.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'masked_templates.json').write_text(json.dumps(prompts, indent=2), encoding='utf-8')
    if args.prepare_only:
        return
    import torch
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError('Text batch size and maximum length must be positive')
    tokenizer, model, model_dir, model_files = load_local_encoder(args.model)
    if model.config.model_type != 'mpnet' or model.config.hidden_size != 768 or tokenizer.mask_token is None:
        raise ValueError('Expected BioLORD MPNet encoder and mask tokenizer')
    model.to(args.device).eval().requires_grad_(False)
    def encode(texts, spans):
        return torch.cat([encode_spans(tokenizer, model, texts[i:i+args.batch_size], spans[i:i+args.batch_size], args.device, args.max_length, return_tensor=True)
                               for i in range(0, len(texts), args.batch_size)])
    texts = [v[b] for v in variants for b in (0, 1)]
    spans = []
    for text in texts:
        match = re.search(r'\bdoes not\b|\bdoes\b', text)
        spans.append(match.span())
    prototypes = encode(texts, spans).reshape(30, 2, 768)
    masked = [s.replace('<MASK>', tokenizer.mask_token) for s in prompts]
    mask_spans = [(s.index(tokenizer.mask_token), s.index(tokenizer.mask_token)+len(tokenizer.mask_token)) for s in masked]
    conditions = encode(masked, mask_spans)
    centers = prototypes.mean(dim=1)
    scales = torch.linalg.vector_norm(prototypes[:, 1]-prototypes[:, 0], dim=-1)/2
    if (scales < 1e-6).any():
        raise ValueError('Indistinguishable polarity prototypes')
    targets = (prototypes-centers[:, None])/scales[:, None, None]
    # Only cache serialization crosses back to CPU; all prototype math stays on device.
    conditions, targets, prototypes, centers, scales = [
        value.cpu().numpy() for value in (conditions, targets, prototypes, centers, scales)]
    cache = output/'reconstruction_features.npz' 
    np.savez_compressed(cache, subject_ids=np.asarray([r['subject_id'] for r in rows]), answers=answers,
                        conditions=conditions, targets=targets, prototypes=prototypes, centers=centers, scales=scales)
    metadata = {'schema': 2, 'max_length': args.max_length, 'batch_size': args.batch_size, 'model': str(model_dir), 'local_files': model_files,
                'local_files_only': True, 'frozen': True,
                'input_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                'cache_sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
                'target': 'centered and pair-distance-scaled contextual first polarity span',
                'prompts': 'shared affirmative template with one mask, independent of subject answer'}
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
