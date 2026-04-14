from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

OUTPUT_FORMATS = {
    "auto",
    "raw_lua",
    "lowcode_lua_fragment",
    "json_with_lua_fields",
}

ALLOWED_ARRAY_HELPERS = {
    "new",
    "markAsArray",
}

FORBIDDEN_WRAPPER_MARKERS = (
    "RESULT",
    "CLARIFICATION_NEEDED",
    "Format:",
    "Assumptions:",
    "Code:",
    "Notes:",
    "```",
)

FORBIDDEN_LUA_REGEX_RULES = (
    (
        r":[ \t]*(filter|map|reduce)\s*\(",
        "unsupported_collection_method",
        "Method-based collection helpers like :filter/:map/:reduce are not allowed.",
    ),
    (
        r"\brequire\s*\(",
        "external_module_forbidden",
        "External modules via require(...) are not allowed in this runtime.",
    ),
    (
        r"\bstring\.split\s*\(",
        "unsupported_string_split",
        "string.split(...) is not part of the allowed plain-Lua runtime.",
    ),
)


def _make_issue(code: str, message: str, severity: str = "error") -> dict[str, str]:
    return {
        "code": code,
        "message": message,
        "severity": severity,
    }


def normalize_output(output: str) -> str:
    stripped = output.strip()
    lines = stripped.splitlines()

    # Модель иногда все равно заворачивает ответ в markdown fence.
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1] == "```":
        return "\n".join(lines[1:-1]).strip()

    return stripped


def detect_output_format(output: str) -> str:
    stripped = normalize_output(output)

    if stripped.startswith("lua{") and stripped.endswith("}lua"):
        return "lowcode_lua_fragment"

    # Если ответ выглядит как JSON-объект, считаем его JSON-режимом даже при
    # битом синтаксисе. Так repair loop сможет увидеть, что модель пыталась
    # вернуть JSON не в том формате.
    if stripped.startswith("{") and stripped.endswith("}"):
        return "json_with_lua_fields"

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return "raw_lua"

    if isinstance(parsed, dict):
        return "json_with_lua_fields"

    return "raw_lua"


def _unwrap_lua_fragment(fragment: str) -> str:
    stripped = fragment.strip()
    if stripped.startswith("lua{") and stripped.endswith("}lua"):
        return stripped[4:-4].strip()
    return stripped


def _collect_json_lua_snippets(
    node: Any,
    path: str,
    snippets: list[str],
    issues: list[dict[str, str]],
) -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            _collect_json_lua_snippets(value, f"{path}.{key}", snippets, issues)
        return

    if isinstance(node, list):
        for index, value in enumerate(node):
            _collect_json_lua_snippets(value, f"{path}[{index}]", snippets, issues)
        return

    if isinstance(node, str):
        stripped = node.strip()
        if stripped.startswith("lua{") and stripped.endswith("}lua"):
            snippets.append(_unwrap_lua_fragment(stripped))
        elif "lua{" in stripped or "}lua" in stripped:
            issues.append(
                _make_issue(
                    "broken_lua_wrapper",
                    f"JSON field at {path} contains an incomplete lua{{...}}lua wrapper.",
                )
            )


def extract_lua_snippets(output: str, detected_output_format: str) -> tuple[list[str], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    stripped = normalize_output(output)

    if detected_output_format == "raw_lua":
        return [stripped], issues

    if detected_output_format == "lowcode_lua_fragment":
        return [_unwrap_lua_fragment(stripped)], issues

    if detected_output_format == "json_with_lua_fields":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            snippets = [
                match.group(1).strip()
                for match in re.finditer(r"lua\{([\s\S]*?)\}lua", stripped)
            ]
            issues.append(
                _make_issue(
                    "invalid_json",
                    f"Output is not valid JSON: {exc.msg}.",
                )
            )
            return snippets, issues

        snippets: list[str] = []
        _collect_json_lua_snippets(parsed, "$", snippets, issues)

        if not snippets:
            issues.append(
                _make_issue(
                    "missing_lua_snippets",
                    "JSON output does not contain any lua{...}lua values.",
                )
            )

        return snippets, issues

    issues.append(
        _make_issue(
            "unknown_output_format",
            f"Unknown detected output format: {detected_output_format}.",
        )
    )
    return [], issues


def _run_lua_syntax_check(snippets: list[str]) -> dict[str, Any]:
    luac_path = (
        os.getenv("LUAC_BIN")
        or shutil.which("luac")
        or shutil.which("luac5.4")
        or shutil.which("luac5.3")
    )
    if luac_path is None:
        return {
            "supported": False,
            "ok": None,
            "errors": [],
            "engine": None,
        }

    errors: list[str] = []
    for index, snippet in enumerate(snippets, start=1):
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".lua",
                delete=False,
                encoding="utf-8",
            ) as handle:
                handle.write(snippet)
                temp_path = Path(handle.name)

            completed = subprocess.run(
                [luac_path, "-p", str(temp_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                details = (completed.stderr or completed.stdout).strip() or "Unknown syntax error"
                errors.append(f"Snippet {index}: {details}")
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    return {
        "supported": True,
        "ok": not errors,
        "errors": errors,
        "engine": luac_path,
    }


def _check_output_wrappers(output: str, issues: list[dict[str, str]]) -> None:
    if not output.strip():
        issues.append(_make_issue("empty_output", "Output must not be empty."))

    if "TODO" in output or "FIXME" in output:
        issues.append(_make_issue("unfinished_placeholder", "Output contains TODO or FIXME."))

    if "$." in output or "jsonpath" in output.lower():
        issues.append(_make_issue("jsonpath_forbidden", "JsonPath syntax is forbidden in this domain."))

    for marker in FORBIDDEN_WRAPPER_MARKERS:
        if marker in output:
            issues.append(
                _make_issue(
                    "forbidden_wrapper_marker",
                    f"Output contains forbidden wrapper marker: {marker}",
                )
            )


def _check_expected_format(
    expected_output_format: str,
    detected_output_format: str,
    output: str,
    issues: list[dict[str, str]],
) -> None:
    if expected_output_format == "auto":
        return

    if expected_output_format != detected_output_format:
        issues.append(
            _make_issue(
                "unexpected_output_format",
                f"Expected {expected_output_format}, got {detected_output_format}.",
            )
        )

    stripped = output.strip()
    if expected_output_format == "raw_lua" and ("lua{" in stripped or "}lua" in stripped):
        issues.append(
            _make_issue(
                "raw_lua_contains_wrapper",
                "raw_lua output must not contain lua{...}lua wrappers.",
            )
        )

    if expected_output_format == "lowcode_lua_fragment":
        if not (stripped.startswith("lua{") and stripped.endswith("}lua")):
            issues.append(
                _make_issue(
                    "invalid_fragment_wrapper",
                    "lowcode_lua_fragment must be exactly one lua{...}lua fragment.",
                )
            )

    if expected_output_format == "json_with_lua_fields":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            issues.append(_make_issue("invalid_json", f"Output is not valid JSON: {exc.msg}."))
        else:
            if not isinstance(parsed, dict):
                issues.append(
                    _make_issue(
                        "json_root_not_object",
                        "json_with_lua_fields must return a JSON object.",
                    )
                )


def _check_allowed_helpers(lua_source: str, issues: list[dict[str, str]]) -> None:
    helpers = set(re.findall(r"_utils\.array\.([A-Za-z_]\w*)", lua_source))
    for helper_name in sorted(helpers - ALLOWED_ARRAY_HELPERS):
        issues.append(
            _make_issue(
                "unknown_array_helper",
                f"Unsupported helper detected: _utils.array.{helper_name}.",
            )
        )

    for pattern, code, message in FORBIDDEN_LUA_REGEX_RULES:
        if re.search(pattern, lua_source, flags=re.IGNORECASE | re.MULTILINE):
            issues.append(_make_issue(code, message))


def _extract_or_field_pair(user_text: str) -> tuple[str, str] | None:
    patterns = (
        r"полях?\s+([A-Za-z_]\w*)\s+или\s+([A-Za-z_]\w*)",
        r"fields?\s+([A-Za-z_]\w*)\s+or\s+([A-Za-z_]\w*)",
    )

    for pattern in patterns:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if match:
            return match.group(1), match.group(2)

    return None


def _instruction_prefix(user_text: str) -> str:
    return user_text.split("{", 1)[0]


def _extract_cleanup_keys(user_text: str) -> list[str]:
    instruction = _instruction_prefix(user_text)
    scope_match = re.search(
        r"(?:переменных|fields?)\s+([^.\n]+)",
        instruction,
        flags=re.IGNORECASE,
    )
    scope = scope_match.group(1) if scope_match else instruction
    candidates = re.findall(r"\b[A-Z][A-Z0-9_]+\b", scope)
    # Сохраняем порядок появления, но убираем дубликаты.
    unique_keys: list[str] = []
    for item in candidates:
        if item not in unique_keys:
            unique_keys.append(item)
    return unique_keys


def _extract_variable_name(user_text: str) -> str | None:
    instruction = _instruction_prefix(user_text)
    patterns = (
        r"переменн(?:ой|ую)\s+([A-Za-z_]\w*)",
        r"variable\s+([A-Za-z_]\w*)",
    )
    for pattern in patterns:
        match = re.search(pattern, instruction, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _check_task_specific_rules(
    user_text: str,
    lua_source: str,
    output_artifact: str,
    issues: list[dict[str, str]],
) -> None:
    lowered_prompt = user_text.lower()
    lowered_code = lua_source.lower()
    lowered_output = output_artifact.lower()

    if (
        "увелич" in lowered_prompt
        or "increment" in lowered_prompt
        or "следующее значение" in lowered_prompt
        or "next value" in lowered_prompt
    ):
        variable_name = _extract_variable_name(user_text)
        if "+ 1" not in lowered_code and "+1" not in lowered_code:
            issues.append(
                _make_issue(
                    "missing_increment_logic",
                    "Increment task should add 1 to the target variable.",
                )
            )

        if variable_name and variable_name.lower() not in lowered_code:
            issues.append(
                _make_issue(
                    "wrong_increment_target",
                    f"Increment task should use the target variable {variable_name}.",
                )
            )

    if re.search(r"return\s+function\b", lowered_code):
        issues.append(
            _make_issue(
                "returned_function_object",
                "Generated code returns a function object instead of the computed result.",
            )
        )

    # Для запросов с "или" хотим хотя бы грубую проверку на AND/OR,
    # чтобы ловить типичные семантические ошибки.
    if (" или " in lowered_prompt or " or " in lowered_prompt) and " or " not in lowered_code and " and " in lowered_code:
        issues.append(
            _make_issue(
                "suspicious_boolean_logic",
                "Task mentions OR logic, but generated code seems to use only AND.",
            )
        )

    if "отфильт" in lowered_prompt or "filter" in lowered_prompt:
        if "_utils.array.new()" not in lua_source and "table.insert" not in lua_source:
            issues.append(
                _make_issue(
                    "weak_filter_implementation",
                    "Filtering task does not appear to build a result array explicitly.",
                )
            )

        if "table.insert" in lua_source and "_utils.array.new()" not in lua_source and "_utils.array.markAsArray" not in lua_source:
            issues.append(
                _make_issue(
                    "array_helper_missing",
                    "Filtering code builds a new array but does not use _utils.array.new() or _utils.array.markAsArray(arr).",
                )
            )

        if ("null" in lowered_prompt or "nil" in lowered_prompt) and " ~= nil" not in lowered_code and " == nil" not in lowered_code:
            issues.append(
                _make_issue(
                    "missing_nil_guard",
                    "Context contains null values, but filtering code does not check for nil explicitly.",
                )
            )

        if ("null" in lowered_prompt or "nil" in lowered_prompt) and (" или " in lowered_prompt or " or " in lowered_prompt):
            field_pair = _extract_or_field_pair(user_text)
            if field_pair is not None:
                left_field, right_field = (field.lower() for field in field_pair)

                left_grouped = re.search(
                    rf"\([^)]*{left_field}[^)]*~=\s*nil[^)]*and[^)]*{left_field}[^)]*~=\s*\"\"[^)]*\)",
                    lowered_code,
                ) or re.search(
                    rf"\([^)]*{left_field}[^)]*~=\s*\"\"[^)]*and[^)]*{left_field}[^)]*~=\s*nil[^)]*\)",
                    lowered_code,
                )

                right_grouped = re.search(
                    rf"\([^)]*{right_field}[^)]*~=\s*nil[^)]*and[^)]*{right_field}[^)]*~=\s*\"\"[^)]*\)",
                    lowered_code,
                ) or re.search(
                    rf"\([^)]*{right_field}[^)]*~=\s*\"\"[^)]*and[^)]*{right_field}[^)]*~=\s*nil[^)]*\)",
                    lowered_code,
                )

                if not (left_grouped and right_grouped):
                    issues.append(
                        _make_issue(
                            "suspicious_or_nil_grouping",
                            "OR filtering with nullable fields should check nil and empty-string per field group.",
                        )
                    )

    if "очист" in lowered_prompt or "clean" in lowered_prompt:
        if " = nil" not in lowered_code or "pairs(" not in lowered_code:
            issues.append(
                _make_issue(
                    "cleanup_logic_incomplete",
                    "Cleanup task should iterate over entries and clear keys dynamically.",
                )
            )

        if re.search(r"\[\d+\]\.[A-Za-z_]\w*", lua_source):
            issues.append(
                _make_issue(
                    "hardcoded_index_patch",
                    "Cleanup task should not hard-code array indices when transforming a collection.",
                )
            )

        cleanup_keys = _extract_cleanup_keys(user_text)
        missing_cleanup_keys = [
            key
            for key in cleanup_keys
            if f"key ~= '{key.lower()}'" not in lowered_code
            and f'key ~= "{key.lower()}"' not in lowered_code
        ]
        if missing_cleanup_keys:
            issues.append(
                _make_issue(
                    "cleanup_keep_keys_missing",
                    "Cleanup task should preserve the keys named in the prompt while clearing other keys.",
                )
            )

    if "iso 8601" in lowered_prompt:
        has_sub_logic = "safe_sub" in lowered_code or "string.sub" in lowered_code or ":sub(" in lowered_code
        has_date_parts = all(
            token in lowered_code
            for token in ("year", "month", "day", "hour", "minute", "second")
        )
        if not (has_sub_logic and has_date_parts and "string.format" in lowered_code):
            issues.append(
                _make_issue(
                    "iso8601_conversion_too_weak",
                    "ISO 8601 conversion should split the incoming date and time into parts before formatting.",
                )
            )

        if ".00000z" not in lowered_code:
            issues.append(
                _make_issue(
                    "iso8601_suffix_missing",
                    "ISO 8601 conversion should use the canonical .00000Z suffix.",
                )
            )

    if "всегда были представлены в виде массивов" in lowered_prompt or "always were arrays" in lowered_prompt:
        if "ensurearray" not in lowered_code and "math.floor(" not in lowered_code and "_utils.array.markasarray" not in lowered_code:
            issues.append(
                _make_issue(
                    "array_normalization_too_weak",
                    "Array-normalization task should explicitly normalize non-array tables.",
                )
            )

    if "последн" in lowered_prompt or "last" in lowered_prompt:
        if "#" not in lua_source:
            issues.append(
                _make_issue(
                    "missing_last_element_pattern",
                    "Task asks for the last element, but code does not use a last-element access pattern.",
                    severity="warning",
                )
            )

    if "квадрат" in lowered_prompt and "числ" in lowered_prompt:
        if '"squared"' not in lowered_output:
            issues.append(
                _make_issue(
                    "missing_squared_field",
                    "Square task should return a field named squared in the final artifact.",
                )
            )

    if "unix" in lowered_prompt and "recalltime" in lowered_prompt:
        if '"unix_time"' not in lowered_output:
            issues.append(
                _make_issue(
                    "missing_unix_time_field",
                    "Unix-time task should use the canonical output field unix_time.",
                )
            )

        if "os.time" in lowered_code:
            issues.append(
                _make_issue(
                    "os_time_forbidden",
                    "Unix-time conversion should not depend on os.time in this LowCode runtime.",
                )
            )

        if (
            "days_since_epoch" not in lowered_code
            or "parse_iso8601_to_epoch" not in lowered_code
            or "days_in_month" not in lowered_code
            or "is_leap_year" not in lowered_code
        ):
            issues.append(
                _make_issue(
                    "unix_conversion_too_weak",
                    "Unix-time conversion should be implemented with explicit pure-Lua parsing logic.",
                )
            )


def validate_lua(
    user_text: str,
    output: str,
    expected_output_format: str = "auto",
) -> dict[str, Any]:
    if expected_output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported expected_output_format: {expected_output_format}")

    normalized_output = normalize_output(output)
    detected_output_format = detect_output_format(normalized_output)

    issues: list[dict[str, str]] = []
    _check_output_wrappers(normalized_output, issues)
    _check_expected_format(
        expected_output_format=expected_output_format,
        detected_output_format=detected_output_format,
        output=normalized_output,
        issues=issues,
    )

    lua_snippets, extraction_issues = extract_lua_snippets(
        normalized_output,
        detected_output_format=detected_output_format,
    )
    issues.extend(extraction_issues)

    combined_lua_source = "\n\n".join(lua_snippets)
    if combined_lua_source:
        _check_allowed_helpers(combined_lua_source, issues)
        _check_task_specific_rules(user_text, combined_lua_source, normalized_output, issues)

    syntax = _run_lua_syntax_check(lua_snippets)
    if syntax["supported"] and syntax["ok"] is False:
        for error_message in syntax["errors"]:
            issues.append(_make_issue("lua_syntax_error", error_message))

    return {
        "ok": not any(issue["severity"] == "error" for issue in issues),
        "expected_output_format": expected_output_format,
        "detected_output_format": detected_output_format,
        "normalized_output": normalized_output,
        "lua_snippets": lua_snippets,
        "syntax": syntax,
        "issues": issues,
    }


__all__ = [
    "OUTPUT_FORMATS",
    "ALLOWED_ARRAY_HELPERS",
    "normalize_output",
    "detect_output_format",
    "extract_lua_snippets",
    "validate_lua",
]
