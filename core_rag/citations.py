import re
from typing import Dict


_SOURCE_MARKER = re.compile(r"\[\[(S\d+)\]\]")
_MARKDOWN_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_CITATION_FIELDS = ("year", "month", "index", "page_number", "speaker", "speaker_number")


def render_citations(report: str, sources: Dict[str, Dict[str, str]]) -> str:
    """Replace source markers with citations and require a source for each bullet."""
    rendered_lines = []

    for line_number, line in enumerate(report.splitlines(), start=1):
        source_ids = list(dict.fromkeys(_SOURCE_MARKER.findall(line)))
        if _MARKDOWN_BULLET.match(line) and not source_ids:
            raise ValueError(f"Final report bullet on line {line_number} has no source ID.")

        citations = []
        for source_id in source_ids:
            metadata = sources.get(source_id)
            if metadata is None:
                raise ValueError(
                    f"Final report references unknown source ID {source_id!r} on line {line_number}."
                )

            values = {}
            for field in _CITATION_FIELDS:
                value = metadata.get(field)
                if value is None or not str(value).strip():
                    raise ValueError(
                        f"Citation source {source_id!r} is missing metadata field {field!r}."
                    )
                values[field] = str(value).strip()

            citations.append(
                f"({values['year']}-{values['month']}, index {values['index']}, "
                f"p. {values['page_number']},"
                f" {values['speaker']})"
            )

        if source_ids:
            line = _SOURCE_MARKER.sub("", line).rstrip()
            line = f"{line} {' '.join(citations)}"
        rendered_lines.append(line)

    return "\n".join(rendered_lines)
