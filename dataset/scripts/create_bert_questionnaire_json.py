"""Convert raw NMS questionnaires into 30 BERT text pairs per subject.

Uses only the Python standard library. Run from any working directory:
    python dataset/scripts/create_bert_questionnaire_json.py
"""

import argparse
import json
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parents[1]


def convert_questionnaires(input_dir: Path) -> list[dict]:
    files = sorted(input_dir.glob("questionnaire_response_*.json"))
    if not files:
        raise ValueError(f"No questionnaire_response_*.json files in {input_dir}")

    samples = []
    seen_subjects = set()
    reference_questions = None
    expected_ids = {f"{number:02d}" for number in range(1, 31)}

    for path in files:
        with path.open(encoding="utf-8-sig") as handle:
            raw = json.load(handle)
        subject_id = raw.get("subject_id")
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError(f"{path.name}: missing or invalid subject_id")
        if subject_id in seen_subjects:
            raise ValueError(f"{path.name}: duplicate subject_id {subject_id}")
        if raw.get("resource_type") != "questionnaire_response":
            raise ValueError(f"{path.name}: unexpected resource_type")

        items = raw.get("item")
        if not isinstance(items, list) or len(items) != 30:
            raise ValueError(f"{path.name}: expected exactly 30 questionnaire items")

        pairs = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"{path.name}: questionnaire item must be an object")
            question_id = item.get("link_id")
            question = item.get("text")
            answer = item.get("answer")
            if not isinstance(question_id, str) or question_id not in expected_ids:
                raise ValueError(f"{path.name}: invalid question ID {question_id!r}")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{path.name}: question {question_id} has no text")
            # Missing answers must never silently become 0.
            if type(answer) is not bool:
                raise ValueError(f"{path.name}: question {question_id} needs a boolean answer")
            pairs.append({
                "question_id": question_id,
                "question": question.strip(),
                "answer": int(answer),
            })

        if {pair["question_id"] for pair in pairs} != expected_ids:
            raise ValueError(f"{path.name}: duplicate or missing question IDs")
        pairs.sort(key=lambda pair: pair["question_id"])
        questions = [(pair["question_id"], pair["question"]) for pair in pairs]
        if reference_questions is None:
            reference_questions = questions
        elif questions != reference_questions:
            raise ValueError(f"{path.name}: question text differs from the first questionnaire")

        samples.append({
            "subject_id": subject_id,
            "qa_pairs": [
                {"question": pair["question"], "answer": pair["answer"]}
                for pair in pairs
            ],
        })
        seen_subjects.add(subject_id)

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATASET_DIR / "questionnaire")
    parser.add_argument(
        "--output", type=Path,
        default=DATASET_DIR / "preprocessed" / "questionnaire_bert.json",
    )
    args = parser.parse_args()
    try:
        samples = convert_questionnaires(args.input_dir)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Conversion failed: {exc}\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(samples, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved {len(samples)} samples, {len(samples) * 30} question-answer pairs")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
