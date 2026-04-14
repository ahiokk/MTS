from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

AgentMode = Literal["generate", "modify"]

DEFAULT_FEW_SHOT_LIMIT = int(os.getenv("LOCALSCRIPT_FEW_SHOT_LIMIT", "2"))
DEFAULT_FEW_SHOT_PATH = Path(
    os.getenv(
        "LOCALSCRIPT_FEW_SHOT_PATH",
        Path(__file__).resolve().parent.parent / "data" / "few_shot_examples.json",
    )
)

STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "как",
    "чтобы",
    "или",
    "для",
    "это",
    "полученных",
    "получи",
    "полученного",
    "добавь",
    "сделай",
    "через",
    "using",
    "need",
    "data",
    "code",
}


@dataclass(frozen=True, slots=True)
class FewShotExample:
    id: str
    mode: AgentMode
    output_format: str
    tags: tuple[str, ...]
    paths: tuple[str, ...]
    prompt: str
    output: str
    context: str | None = None
    existing_code: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievedExample:
    example: FewShotExample
    score: int
    matched_tags: tuple[str, ...]
    matched_paths: tuple[str, ...]


def _normalize_token(token: str) -> str:
    return token.lower().strip("_")


def _tokenize(text: str) -> set[str]:
    tokens = {_normalize_token(token) for token in re.findall(r"[A-Za-zА-Яа-яЁё_][A-Za-zА-Яа-яЁё0-9_]{2,}", text)}
    return {
        token
        for token in tokens
        if token and token not in STOP_WORDS
    }


def infer_task_tags(
    user_text: str,
    *,
    mode: AgentMode,
    output_format: str,
    context_attached: bool,
    existing_code_attached: bool,
    feedback_attached: bool,
) -> list[str]:
    lowered = user_text.lower()
    tags: list[str] = [mode, output_format]

    if context_attached:
        tags.append("has_context")
    if existing_code_attached:
        tags.append("has_existing_code")
    if feedback_attached:
        tags.append("has_feedback")

    if "wf.vars" in lowered:
        tags.append("wf_vars")
    if "wf.initvariables" in lowered:
        tags.append("wf_init_variables")

    if "последн" in lowered or "last" in lowered:
        tags.append("last_element")
    if "увелич" in lowered or "increment" in lowered:
        tags.append("increment")
    if "очист" in lowered or "clear" in lowered or "remove every other key" in lowered:
        tags.append("cleanup")
    if "discount" in lowered or "markdown" in lowered:
        tags.append("discount_markdown_filter")
    if "iso 8601" in lowered or ("datum" in lowered and "time" in lowered):
        tags.append("iso8601")
    if "unix" in lowered:
        tags.append("unix_time")
    if "items" in lowered and ("массив" in lowered or "array" in lowered):
        tags.append("ensure_arrays")
    if "квадрат" in lowered or "squared" in lowered:
        tags.append("square_patch")
    if "restbody" in lowered or "rest response" in lowered:
        tags.append("restbody")
    if "lowcode" in lowered or "low-code" in lowered or "lua{" in lowered or "фрагмент" in lowered:
        tags.append("lowcode_fragment")

    return list(dict.fromkeys(tags))


@lru_cache(maxsize=1)
def load_few_shot_examples() -> tuple[FewShotExample, ...]:
    payload = json.loads(DEFAULT_FEW_SHOT_PATH.read_text(encoding="utf-8"))
    raw_examples = payload.get("examples", [])
    examples: list[FewShotExample] = []

    for item in raw_examples:
        examples.append(
            FewShotExample(
                id=item["id"],
                mode=item["mode"],
                output_format=item["output_format"],
                tags=tuple(item.get("tags", [])),
                paths=tuple(item.get("paths", [])),
                prompt=item["prompt"],
                output=item["output"],
                context=item.get("context"),
                existing_code=item.get("existing_code"),
            )
        )

    return tuple(examples)


def _path_overlap(query_paths: list[str], example_paths: tuple[str, ...]) -> list[str]:
    overlaps: list[str] = []
    for query_path in query_paths:
        for example_path in example_paths:
            if query_path == example_path or query_path.startswith(example_path) or example_path.startswith(query_path):
                overlaps.append(example_path)
    return list(dict.fromkeys(overlaps))


def retrieve_few_shot_examples(
    *,
    user_text: str,
    mode: AgentMode,
    output_format: str,
    referenced_paths: list[str],
    context_attached: bool,
    existing_code_attached: bool,
    feedback_attached: bool,
    limit: int = DEFAULT_FEW_SHOT_LIMIT,
) -> list[RetrievedExample]:
    if limit <= 0:
        return []

    inferred_tags = set(
        infer_task_tags(
            user_text,
            mode=mode,
            output_format=output_format,
            context_attached=context_attached,
            existing_code_attached=existing_code_attached,
            feedback_attached=feedback_attached,
        )
    )
    query_tokens = _tokenize(user_text)
    ranked: list[RetrievedExample] = []

    for example in load_few_shot_examples():
        score = 0

        if example.mode == mode:
            score += 24
        if example.output_format == output_format:
            score += 36

        matched_tags = sorted(inferred_tags.intersection(example.tags))
        score += min(len(matched_tags), 4) * 12

        matched_paths = _path_overlap(referenced_paths, example.paths)
        score += len(matched_paths) * 15

        example_tokens = _tokenize(example.prompt)
        token_overlap = query_tokens.intersection(example_tokens)
        score += min(len(token_overlap), 6) * 2

        if existing_code_attached and example.existing_code:
            score += 8
        if context_attached and example.context:
            score += 6

        if score < 40:
            continue

        ranked.append(
            RetrievedExample(
                example=example,
                score=score,
                matched_tags=tuple(matched_tags),
                matched_paths=tuple(matched_paths),
            )
        )

    ranked.sort(key=lambda item: (-item.score, item.example.id))
    return ranked[:limit]


def render_retrieved_examples(examples: list[RetrievedExample]) -> str:
    if not examples:
        return ""

    blocks: list[str] = ["Retrieved local few-shot examples:"]
    for index, retrieved in enumerate(examples, start=1):
        example = retrieved.example
        block_lines = [
            f"Example {index}",
            f"Mode: {example.mode}",
            f"Output format: {example.output_format}",
            f"Task:\n{example.prompt.strip()}",
        ]
        if example.context:
            block_lines.append(f"Context:\n{example.context.strip()}")
        if example.existing_code:
            block_lines.append(f"Existing code:\n{example.existing_code.strip()}")
        block_lines.append(f"Correct output:\n{example.output.strip()}")
        blocks.append("\n".join(block_lines))

    return "\n\n".join(blocks)


__all__ = [
    "DEFAULT_FEW_SHOT_LIMIT",
    "DEFAULT_FEW_SHOT_PATH",
    "FewShotExample",
    "RetrievedExample",
    "infer_task_tags",
    "load_few_shot_examples",
    "retrieve_few_shot_examples",
    "render_retrieved_examples",
]
