import json
from collections.abc import Callable
from typing import Any

from core_rag.vector_store import LocalVectorStorage

def expand_question_parent_hits(
    parent_hits: list[dict[str, Any]],
    db: LocalVectorStorage,
    load_parent_text: Callable[[str], str],
    get_classifier_response: Callable[[str], str]
) -> list[dict[str, Any]]:
    """Include the next speaker parent when a retrieved parent is a question."""
    prompt_prefix = (
        "Decide whether this transcript parent is primarily a direct question or request "
        "to another speaker that the next speaker turn could answer. Do not classify "
        "rhetorical, quoted, or incidental questions as question parents. "
        'Return only JSON with exactly one boolean field: {"is_question": true|false}.\n'
        "Parent text:\n"
    )

    expanded = []
    seen_parent_ids = set()
    for hit in parent_hits:
        metadata = hit["metadata"]
        parent_id = metadata.get("parent_id")
        if not parent_id or parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)
        current = {
            "parent_id": parent_id,
            "metadata": metadata,
            "text": hit.get("text") or load_parent_text(parent_id)
        }
        expanded.append(current)

        response_text = get_classifier_response(
            prompt_prefix + json.dumps(hit.get("text") or current["text"])
        ).strip()
        if response_text.startswith("```"):
            lines = response_text.splitlines()
            if lines and lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            response_text = "\n".join(lines).strip()

        try:
            classification = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Question classifier returned invalid JSON for parent {parent_id!r}."
            ) from error
        if (
            not isinstance(classification, dict)
            or set(classification) != {"is_question"}
            or type(classification["is_question"]) is not bool
        ):
            raise ValueError(
                f"Question classifier returned an invalid result for parent {parent_id!r}; "
                "expected exactly one boolean field named 'is_question'."
            )
        if not classification["is_question"]:
            continue

        required_fields = ("ticker", "year", "month", "index", "speaker_number")
        missing_fields = [
            field for field in required_fields
            if metadata.get(field) is None or not str(metadata[field]).strip()
        ]
        if missing_fields:
            raise ValueError(
                f"Retrieved question parent {parent_id!r} is missing metadata fields: "
                f"{', '.join(missing_fields)}. Reindex transcript chunks with speaker_number."
            )

        raw_speaker_number = str(metadata["speaker_number"])
        if not (raw_speaker_number.isascii() and raw_speaker_number.isdecimal()):
            raise ValueError(
                f"Retrieved question parent {parent_id!r} has invalid speaker_number "
                f"{raw_speaker_number!r}."
            )
        next_metadata = db.get_parent_by_speaker_number(
            ticker=metadata["ticker"],
            year=metadata["year"],
            month=metadata["month"],
            index=metadata["index"],
            speaker_number=int(raw_speaker_number) + 1
        )
        if next_metadata is None:
            continue

        next_parent_id = next_metadata["parent_id"]
        if next_parent_id not in seen_parent_ids:
            seen_parent_ids.add(next_parent_id)
            expanded.append({
                "parent_id": next_parent_id,
                "metadata": next_metadata,
                "text": load_parent_text(next_parent_id)
            })

    return expanded
