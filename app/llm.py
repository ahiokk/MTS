from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ollama import Client

from app.retrieval import (
    DEFAULT_FEW_SHOT_LIMIT,
    RetrievedExample,
    render_retrieved_examples,
    retrieve_few_shot_examples,
)
from app.validator import validate_lua

DEFAULT_MODEL = os.getenv("LOCALSCRIPT_MODEL", "qwen2.5-coder:7b")
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_NUM_CTX = int(os.getenv("LOCALSCRIPT_NUM_CTX", "4096"))
DEFAULT_NUM_PREDICT = int(os.getenv("LOCALSCRIPT_NUM_PREDICT", "256"))
DEFAULT_TEMPERATURE = float(os.getenv("LOCALSCRIPT_TEMPERATURE", "0"))
DEFAULT_SEED = int(os.getenv("LOCALSCRIPT_SEED", "42"))
MAX_REPAIR_ATTEMPTS = 2

AgentMode = Literal["generate", "modify"]
AgentStatus = Literal["completed", "needs_clarification"]

# Эти режимы нужны, чтобы система не "угадывала" формат ответа каждый раз.
# Мы либо задаем формат явно, либо разрешаем policy-слою выбрать его по правилам.
OUTPUT_FORMATS = {
    "auto",
    "raw_lua",
    "lowcode_lua_fragment",
    "json_with_lua_fields",
}

MODIFICATION_HINTS = (
    "доработай код",
    "измени код",
    "исправь код",
    "дополни код",
    "patch the code",
    "modify the code",
    "fix the code",
    "update the code",
    "existing code",
    "текущий код",
)

LOWCODE_HINTS = (
    "lowcode",
    "low-code",
    "lua{",
    "фрагмент",
    "script expression",
)

JSON_OUTPUT_HINTS = (
    "json",
    "json object",
    "объект",
    "поле",
    "fields",
    "field",
    "named output",
    "output field",
    "сформируй поле",
    "заполни поле",
    "добавь переменную",
    "return an object",
    "build an object",
)

CONTEXT_DEPENDENT_HINTS = (
    "предыдущ",
    "полученн",
    "этот ",
    "эта ",
    "эти ",
    "данных",
    "контекст",
    "rest response",
    "previous",
    "received",
    "this ",
    "these ",
    "current data",
)


@dataclass(slots=True)
class TaskAnalysis:
    mode: AgentMode
    resolved_output_format: str
    needs_clarification: bool
    clarification_question: str | None
    referenced_paths: list[str]
    context_attached: bool
    existing_code_attached: bool
    feedback_attached: bool


@dataclass(slots=True)
class AgentResult:
    status: AgentStatus
    code: str | None
    clarification_question: str | None
    analysis: TaskAnalysis


def _contains_inline_context(prompt: str) -> bool:
    return "{" in prompt and ("\"wf\"" in prompt or "'wf'" in prompt or "wf." in prompt)


def _contains_cyrillic(text: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", text))


def _serialize_context(context: Any | None) -> str | None:
    if context is None:
        return None

    if isinstance(context, str):
        serialized = context.strip()
        return serialized or None

    return json.dumps(context, ensure_ascii=False, indent=2)


def compose_user_text(user_text: str, context: Any | None = None) -> str:
    cleaned_text = user_text.strip()
    if not cleaned_text:
        raise ValueError("user_text must not be empty")

    serialized_context = _serialize_context(context)
    if not serialized_context:
        return cleaned_text

    # Контекст передаем отдельным блоком, чтобы API мог принимать prompt и context
    # раздельно, но модель все равно видела одну цельную задачу.
    return f"{cleaned_text}\n\nContext:\n{serialized_context}"


def _dedupe_keep_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def extract_referenced_paths(text: str) -> list[str]:
    matches = re.findall(r"\bwf\.(?:vars|initVariables)(?:\.[A-Za-z_]\w*)+\b", text)
    return _dedupe_keep_order(matches)


def extract_paths_from_context(context: Any | None) -> list[str]:
    paths: list[str] = []

    def walk(node: Any, current_path: str | None = None) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                next_path = key if current_path is None else f"{current_path}.{key}"
                if next_path.startswith("wf.vars") or next_path.startswith("wf.initVariables"):
                    paths.append(next_path)
                walk(value, next_path)
            return

        if isinstance(node, list):
            if current_path and (
                current_path.startswith("wf.vars") or current_path.startswith("wf.initVariables")
            ):
                paths.append(current_path)
            for item in node:
                walk(item, current_path)

    walk(context)
    return _dedupe_keep_order(paths)


def resolve_output_format(
    user_text: str,
    *,
    requested_output_format: str,
    context_attached: bool,
    existing_code_attached: bool,
) -> str:
    if requested_output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported target_output_format: {requested_output_format}")

    if requested_output_format != "auto":
        return requested_output_format

    lowered = user_text.lower()

    if any(hint in lowered for hint in LOWCODE_HINTS):
        return "lowcode_lua_fragment"

    if any(hint in lowered for hint in JSON_OUTPUT_HINTS):
        return "json_with_lua_fields"

    if existing_code_attached and "return {" in lowered:
        return "json_with_lua_fields"

    if context_attached:
        return "raw_lua"

    return "raw_lua"


def _needs_context_clarification(user_text: str, context_attached: bool) -> bool:
    if context_attached:
        return False

    lowered = user_text.lower()
    return any(hint in lowered for hint in CONTEXT_DEPENDENT_HINTS)


def _build_clarification_question(
    user_text: str,
    *,
    needs_existing_code: bool,
    needs_context: bool,
) -> str | None:
    if not needs_existing_code and not needs_context:
        return None

    if _contains_cyrillic(user_text):
        if needs_existing_code:
            return "Пришлите текущий Lua-код, который нужно доработать или исправить."
        if needs_context:
            return "Пришлите входной контекст: JSON с wf.vars / wf.initVariables или пример текущих данных."
        return None

    if needs_existing_code:
        return "Please send the current Lua code that should be modified or fixed."
    if needs_context:
        return "Please send the input context: JSON with wf.vars / wf.initVariables or a sample of the current data."
    return None


def analyze_task(
    user_text: str,
    *,
    context: Any | None = None,
    existing_code: str | None = None,
    feedback: str | None = None,
    target_output_format: str = "auto",
) -> TaskAnalysis:
    combined_user_text = compose_user_text(user_text, context=context)
    lowered = user_text.lower()
    existing_code_attached = bool(existing_code and existing_code.strip())
    context_attached = bool(_serialize_context(context)) or _contains_inline_context(user_text)
    feedback_attached = bool(feedback and feedback.strip())
    referenced_paths = _dedupe_keep_order(
        extract_referenced_paths(combined_user_text) + extract_paths_from_context(context)
    )

    mode: AgentMode = "generate"
    if existing_code_attached or feedback_attached or any(hint in lowered for hint in MODIFICATION_HINTS):
        mode = "modify"

    resolved_output_format = resolve_output_format(
        user_text=combined_user_text,
        requested_output_format=target_output_format,
        context_attached=context_attached,
        existing_code_attached=existing_code_attached,
    )

    needs_existing_code = mode == "modify" and not existing_code_attached
    needs_context = _needs_context_clarification(user_text, context_attached=context_attached)
    clarification_question = _build_clarification_question(
        user_text,
        needs_existing_code=needs_existing_code,
        needs_context=needs_context and not needs_existing_code,
    )

    return TaskAnalysis(
        mode=mode,
        resolved_output_format=resolved_output_format,
        needs_clarification=clarification_question is not None,
        clarification_question=clarification_question,
        referenced_paths=referenced_paths,
        context_attached=context_attached,
        existing_code_attached=existing_code_attached,
        feedback_attached=feedback_attached,
    )


def _extract_init_variable_name(user_text: str) -> str | None:
    path_match = re.search(r"\bwf\.initVariables\.([A-Za-z_]\w*)\b", user_text)
    if path_match:
        return path_match.group(1)

    instruction_match = re.search(
        r"(?:переменн(?:ой|ую)|variable)\s+([A-Za-z_]\w*)",
        user_text,
        flags=re.IGNORECASE,
    )
    if instruction_match:
        return instruction_match.group(1)

    return None


def _build_unix_time_template(init_variable_name: str, target_output_format: str) -> str:
    lua_code = f"""local iso_time = wf.initVariables.{init_variable_name}
local days_in_month = {{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}}
if not iso_time or not iso_time:match("^%d%d%d%d%-%d%d%-%d%dT") then
 return nil
end
local function is_leap_year(year)
 return (year % 4 == 0 and year % 100 ~= 0) or (year % 400 == 0)
end
local function days_since_epoch(year, month, day)
 local days = 0
 for y = 1970, year - 1 do
  days = days + (is_leap_year(y) and 366 or 365)
 end
 for m = 1, month - 1 do
  days = days + days_in_month[m]
  if m == 2 and is_leap_year(year) then
   days = days + 1
  end
 end
 days = days + (day - 1)
 return days
end
local function parse_iso8601_to_epoch(iso_str)
 if not iso_str then
  error("Date is nil")
 end
 local year, month, day, hour, min, sec, ms, offset_sign, offset_hour, offset_min =
  iso_str:match("(%d+)-(%d+)-(%d+)T(%d+):(%d+):(%d+)%.(%d+)([+-])(%d+):(%d+)")
 if not year then
  year, month, day, hour, min, sec, offset_sign, offset_hour, offset_min =
   iso_str:match("(%d+)-(%d+)-(%d+)T(%d+):(%d+):(%d+)([+-])(%d+):(%d+)")
  ms = 0
 end
 if not year then
  error("Cannot parse date: " .. tostring(iso_str))
 end
 year = tonumber(year); month = tonumber(month); day = tonumber(day)
 hour = tonumber(hour); min = tonumber(min); sec = tonumber(sec)
 ms = tonumber(ms) or 0
 offset_hour = tonumber(offset_hour); offset_min = tonumber(offset_min)
 local total_days = days_since_epoch(year, month, day)
 local total_seconds = total_days * 86400 + hour * 3600 + min * 60 + sec
 local offset_seconds = offset_hour * 3600 + offset_min * 60
 if offset_sign == "-" then
  offset_seconds = -offset_seconds
 end
 return total_seconds - offset_seconds
end
local epoch_seconds = parse_iso8601_to_epoch(iso_time)
return epoch_seconds""".strip()

    if target_output_format == "raw_lua":
        return lua_code

    if target_output_format == "lowcode_lua_fragment":
        return f"lua{{{lua_code}}}lua"

    if target_output_format == "json_with_lua_fields":
        return json.dumps({"unix_time": f"lua{{{lua_code}}}lua"}, ensure_ascii=False)

    return lua_code


def maybe_resolve_template(
    user_text: str,
    *,
    target_output_format: str,
) -> str | None:
    lowered = user_text.lower()

    if "unix" in lowered and "initvariables" in lowered:
        init_variable_name = _extract_init_variable_name(user_text)
        if init_variable_name:
            return _build_unix_time_template(
                init_variable_name,
                target_output_format=target_output_format,
            )

    return None


def select_few_shot_examples(
    user_text: str,
    *,
    analysis: TaskAnalysis,
    limit: int = DEFAULT_FEW_SHOT_LIMIT,
) -> list[RetrievedExample]:
    return retrieve_few_shot_examples(
        user_text=user_text,
        mode=analysis.mode,
        output_format=analysis.resolved_output_format,
        referenced_paths=analysis.referenced_paths,
        context_attached=analysis.context_attached,
        existing_code_attached=analysis.existing_code_attached,
        feedback_attached=analysis.feedback_attached,
        limit=limit,
    )


def build_system_prompt(
    target_output_format: str = "auto",
    *,
    mode: AgentMode = "generate",
    retrieved_examples: Sequence[RetrievedExample] | None = None,
) -> str:
    if target_output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported target_output_format: {target_output_format}")

    mode_rules = """
Generation mode rules:
1. Solve the task from scratch using only the provided prompt and context.
2. Choose the simplest correct Lua implementation that matches the target format.
""".strip()

    if mode == "modify":
        mode_rules = """
Modification mode rules:
1. When existing Lua code is provided, modify that code instead of rewriting from scratch unless the request requires a larger rewrite.
2. Preserve correct variable names, output structure, and working logic that is unrelated to the requested change.
3. Apply the smallest correct change that satisfies the task and any user feedback.
4. Do not remove working fields from a JSON object unless the request explicitly asks for that.
""".strip()

    retrieval_block = render_retrieved_examples(list(retrieved_examples or ()))

    # Главная идея промпта: модель должна возвращать только финальный код,
    # без "агентских" оберток вроде RESULT / Code / Notes.
    prompt = f"""
You are LocalScript, a local AI assistant that generates Lua code for a LowCode runtime.

Your only task is to return the final code artifact.

Hard output rules:
1. Return only the final code.
2. Do not return explanations, comments, notes, assumptions, headers, markdown, or code fences.
3. Do not return words like RESULT, FORMAT, CODE, NOTES, or CLARIFICATION_NEEDED.
4. Do not wrap the answer in triple backticks.
5. Do not use JsonPath. Access data directly through wf.vars or wf.initVariables.
6. Do not invent unavailable fields or helper variables outside the provided context.
7. Do not output pseudocode, TODO, FIXME, or placeholders.
8. Do not use require(...), imported modules, or hidden runtime libraries.
9. Do not use method-based collection helpers like :filter(...), :map(...), or :reduce(...).
10. Never return a function object like `return function() ... end`; return the computed result directly.

LowCode environment rules:
1. Workflow variables are stored in wf.vars.
2. Initial input variables passed at startup are stored in wf.initVariables.
3. Use lua{{ ... }}lua only when the required output format needs a LowCode script string.
4. When creating arrays, prefer _utils.array.new() when appropriate.
5. Allowed control flow is basic Lua: if, while, for, repeat-until.
6. Keep the solution simple and compatible with the examples.
7. When filtering arrays, build the result explicitly with loops and table.insert.
8. When transforming arrays or objects, return the transformed collection or object itself, not a set of hard-coded index patches.
9. For date and time conversion tasks, implement parsing in plain Lua.
10. In json_with_lua_fields mode, do not return an already evaluated JSON array or object as the field value; return a lua{{...}}lua string that computes it.

{mode_rules}

Target output format: {target_output_format}

Format rules:
- raw_lua: return plain Lua code only, such as `return wf.vars.emails[#wf.vars.emails]`
- lowcode_lua_fragment: return only one string in the form `lua{{ ... }}lua`
- json_with_lua_fields: return only one valid JSON object; values that contain code must be strings in the form `lua{{ ... }}lua`
- auto: choose exactly one format using the rules below

Auto-selection rules:
1. If the user explicitly asks for raw Lua or asks to compute/return one value, prefer raw_lua.
2. If the user explicitly asks for a LowCode fragment or script expression, use lowcode_lua_fragment.
3. If the user asks to produce named output fields, build an object, or fill a field in a JSON-like result, use json_with_lua_fields.
4. If the request contains a context example and no explicit output field name, prefer raw_lua.

Canonical examples:
User request: "Из полученного списка email получи последний."
Correct raw_lua:
return wf.vars.emails[#wf.vars.emails]

Correct json_with_lua_fields:
{{"lastEmail":"lua{{return wf.vars.emails[#wf.vars.emails]}}lua"}}

User request: "Отфильтруй элементы из массива, чтобы включить только те, у которых есть значения в полях Discount или Markdown."
Correct raw_lua:
local result = _utils.array.new()
local items = wf.vars.parsedCsv
for _, item in ipairs(items) do
 if (item.Discount ~= "" and item.Discount ~= nil) or (item.Markdown ~= "" and item.Markdown ~= nil) then
  table.insert(result, item)
 end
end
return result

User request: "Добавь переменную с квадратом числа."
Correct json_with_lua_fields:
{{"num":"lua{{return tonumber('5')}}lua","squared":"lua{{local n = tonumber('5'); return n * n}}lua"}}

User request: "Для полученных данных из предыдущего REST запроса очисти значения переменных ID, ENTITY_ID, CALL."
Correct json_with_lua_fields:
{{"result":"lua{{result = wf.vars.RESTbody.result; for _, filteredEntry in pairs(result) do for key, value in pairs(filteredEntry) do if key ~= 'ID' and key ~= 'ENTITY_ID' and key ~= 'CALL' then filteredEntry[key] = nil end end end return result}}lua"}}

User request: "For the previous REST response clear values in fields ORDER_ID, STATE, COMMENT and remove every other key."
Correct json_with_lua_fields:
{{"result":"lua{{result = wf.vars.RESTbody.result; for _, filteredEntry in pairs(result) do for key, value in pairs(filteredEntry) do if key ~= 'ORDER_ID' and key ~= 'STATE' and key ~= 'COMMENT' then filteredEntry[key] = nil end end end return result}}lua"}}

User request: "Преобразуй время из формата 'YYYYMMDD' и 'HHMMSS' в строку в формате ISO 8601 с использованием Lua."
Correct raw_lua:
DATUM = wf.vars.json.IDOC.ZCDF_HEAD.DATUM
TIME = wf.vars.json.IDOC.ZCDF_HEAD.TIME
local function safe_sub(str, start, finish)
 local s = string.sub(str, start, math.min(finish, #str))
 return s ~= "" and s or "00"
end
year = safe_sub(DATUM, 1, 4)
month = safe_sub(DATUM, 5, 6)
day = safe_sub(DATUM, 7, 8)
hour = safe_sub(TIME, 1, 2)
minute = safe_sub(TIME, 3, 4)
second = safe_sub(TIME, 5, 6)
iso_date = string.format('%s-%s-%sT%s:%s:%s.00000Z', year, month, day, hour, minute, second)
return iso_date

User request: "Как преобразовать структуру данных так, чтобы все элементы items всегда были представлены в виде массивов?"
Correct raw_lua:
function ensureArray(t)
 if type(t) ~= "table" then
  return {{t}}
 end
 local isArray = true
 for k, v in pairs(t) do
  if type(k) ~= "number" or math.floor(k) ~= k then
   isArray = false
   break
  end
 end
 return isArray and t or {{t}}
end

function ensureAllItemsAreArrays(objectsArray)
 if type(objectsArray) ~= "table" then
  return objectsArray
 end
 for _, obj in ipairs(objectsArray) do
  if type(obj) == "table" and obj.items then
   obj.items = ensureArray(obj.items)
  end
 end
 return objectsArray
end

return ensureAllItemsAreArrays(wf.vars.json.IDOC.ZCDF_HEAD.ZCDF_PACKAGES)

User request: "Конвертируй время в переменной recallTime в unix-формат."
Correct json_with_lua_fields:
{{"unix_time":"lua{{local iso_time = wf.initVariables.recallTime
local days_in_month = {{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}}
if not iso_time or not iso_time:match('^%d%d%d%d%-%d%d%-%d%dT') then
 return nil
end
local function is_leap_year(year)
 return (year % 4 == 0 and year % 100 ~= 0) or (year % 400 == 0)
end
local function days_since_epoch(year, month, day)
 local days = 0
 for y = 1970, year - 1 do
  days = days + (is_leap_year(y) and 366 or 365)
 end
 for m = 1, month - 1 do
  days = days + days_in_month[m]
  if m == 2 and is_leap_year(year) then
   days = days + 1
  end
 end
 days = days + (day - 1)
 return days
end
local function parse_iso8601_to_epoch(iso_str)
 local year, month, day, hour, min, sec, ms, offset_sign, offset_hour, offset_min =
  iso_str:match('(%d+)-(%d+)-(%d+)T(%d+):(%d+):(%d+)%.(%d+)([+-])(%d+):(%d+)')
 if not year then
  year, month, day, hour, min, sec, offset_sign, offset_hour, offset_min =
   iso_str:match('(%d+)-(%d+)-(%d+)T(%d+):(%d+):(%d+)([+-])(%d+):(%d+)')
  ms = 0
 end
 year = tonumber(year); month = tonumber(month); day = tonumber(day)
 hour = tonumber(hour); min = tonumber(min); sec = tonumber(sec)
 offset_hour = tonumber(offset_hour); offset_min = tonumber(offset_min)
 local total_days = days_since_epoch(year, month, day)
 local total_seconds = total_days * 86400 + hour * 3600 + min * 60 + sec
 local offset_seconds = offset_hour * 3600 + offset_min * 60
 if offset_sign == '-' then
  offset_seconds = -offset_seconds
 end
 return total_seconds - offset_seconds
end
local epoch_seconds = parse_iso8601_to_epoch(iso_time)
return epoch_seconds}}lua"}}

Modification example:
Task: "Extend the existing JSON output with a squared value."
Existing code:
{{"num":"lua{{return tonumber('5')}}lua"}}
Correct modified output:
{{"num":"lua{{return tonumber('5')}}lua","squared":"lua{{local n = tonumber('5'); return n * n}}lua"}}

Task-specific guidance:
- For increment tasks, add exactly 1 to the variable named in the request and return the incremented result.
- For cleanup tasks over wf.vars.RESTbody.result, iterate over entries and keys, remove all keys except the allowed ones, and return one final result object or collection.
- For cleanup tasks, preserve exactly the field names named in the prompt and clear all other keys.
- For ISO 8601 conversion from DATUM and TIME, split the source into year/month/day/hour/minute/second first, then build iso_date with string.format.
- For tasks that must ensure items are arrays, implement explicit normalization logic for both arrays and non-array tables instead of only wrapping scalar values.
- For unix timestamp conversion, use the canonical output field unix_time and implement pure Lua helpers like days_since_epoch and parse_iso8601_to_epoch.

You must return only one final artifact in the selected target format.
""".strip()

    if retrieval_block:
        prompt = f"{prompt}\n\n{retrieval_block}"

    return prompt


def build_task_payload(
    user_text: str,
    *,
    context: Any | None = None,
    existing_code: str | None = None,
    feedback: str | None = None,
) -> str:
    sections = [f"Task:\n{user_text.strip()}"]

    serialized_context = _serialize_context(context)
    if serialized_context:
        sections.append(f"Structured context:\n{serialized_context}")

    if existing_code and existing_code.strip():
        sections.append(f"Existing Lua code to modify:\n{existing_code.strip()}")

    if feedback and feedback.strip():
        sections.append(f"User feedback to address:\n{feedback.strip()}")

    return "\n\n".join(sections)


def build_initial_messages(
    user_text: str,
    target_output_format: str = "auto",
    *,
    mode: AgentMode = "generate",
    context: Any | None = None,
    existing_code: str | None = None,
    feedback: str | None = None,
    retrieved_examples: Sequence[RetrievedExample] | None = None,
) -> list[dict[str, str]]:
    cleaned_text = user_text.strip()
    if not cleaned_text:
        raise ValueError("user_text must not be empty")

    # Ollama chat API ожидает список сообщений в формате system/user/assistant.
    return [
        {
            "role": "system",
            "content": build_system_prompt(
                target_output_format,
                mode=mode,
                retrieved_examples=retrieved_examples,
            ),
        },
        {
            "role": "user",
            "content": build_task_payload(
                cleaned_text,
                context=context,
                existing_code=existing_code,
                feedback=feedback,
            ),
        },
    ]


def _extract_content(response: Any) -> str:
    # У Ollama ответ может прийти как объект SDK или как dict-подобная структура.
    # Здесь аккуратно достаем текст, не завязываясь на один формат.
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content

    if isinstance(response, Mapping):
        maybe_message = response.get("message")
        if isinstance(maybe_message, Mapping):
            content = maybe_message.get("content")
            if isinstance(content, str):
                return content

    raise TypeError("Could not extract message content from Ollama response")


def normalize_model_output(output: str) -> str:
    stripped = output.strip()
    lines = stripped.splitlines()

    # Даже если мы запретили markdown в промпте, модель иногда все равно
    # добавляет ```lua ... ```. Здесь убираем такую обертку автоматически.
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1] == "```":
        return "\n".join(lines[1:-1]).strip()

    return stripped


def _normalize_embedded_lua_string(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_embedded_lua_string(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_normalize_embedded_lua_string(item) for item in value]

    if isinstance(value, str) and value.startswith("lua{") and value.endswith("}lua"):
        return (
            value.replace("\\r\\n", "\n")
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace('\\"', '"')
        )

    return value


def canonicalize_output_for_target(output: str, target_output_format: str) -> str:
    normalized = normalize_model_output(output)
    if target_output_format != "json_with_lua_fields":
        return normalized

    try:
        parsed = json.loads(normalized)
        normalized_parsed = _normalize_embedded_lua_string(parsed)
        return json.dumps(normalized_parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    # Модель иногда возвращает JSON-объект с lua{...}lua-значениями,
    # но кладет внутрь строк "сырые" переводы строк. Для API это все еще
    # почти правильный артефакт, поэтому аккуратно собираем его заново.
    pairs = list(
        re.finditer(
            r'"(?P<key>[^"\\]+)"\s*:\s*"lua\{(?P<body>[\s\S]*?)\}lua"',
            normalized,
        )
    )
    if not pairs:
        return normalized

    repaired: dict[str, str] = {}
    for match in pairs:
        repaired[match.group("key")] = f"lua{{{match.group('body').strip()}}}lua"

    return json.dumps(repaired, ensure_ascii=False)


def build_ollama_client(host: str = DEFAULT_OLLAMA_HOST) -> Client:
    return Client(host=host)


def call_llm(model: str, messages: Sequence[Mapping[str, str]]) -> str:
    try:
        client = build_ollama_client()
        response = client.chat(
            model=model,
            messages=list(messages),
            options={
                "temperature": DEFAULT_TEMPERATURE,
                "num_ctx": DEFAULT_NUM_CTX,
                "num_predict": DEFAULT_NUM_PREDICT,
                "seed": DEFAULT_SEED,
            },
        )
    except Exception as exc:
        # Оборачиваем низкоуровневую ошибку в понятное сообщение для ноутбука/API.
        raise RuntimeError(
            "Failed to call local Ollama. Make sure Ollama is running and the model is pulled."
        ) from exc

    return normalize_model_output(_extract_content(response))


def validate_agent_output(
    output: str,
    target_output_format: str = "auto",
    user_text: str = "",
) -> list[str]:
    # Старый guardrail заменен на полноценный валидатор.
    # Здесь оставляем только совместимый интерфейс: list[str].
    report = validate_lua(
        user_text=user_text,
        output=output,
        expected_output_format=target_output_format,
    )
    return [
        issue["message"]
        for issue in report["issues"]
        if issue["severity"] == "error"
    ]


def build_repair_messages(
    user_text: str,
    bad_output: str,
    issues: Sequence[str],
    target_output_format: str = "auto",
    *,
    mode: AgentMode = "generate",
    context: Any | None = None,
    existing_code: str | None = None,
    feedback: str | None = None,
    retrieved_examples: Sequence[RetrievedExample] | None = None,
) -> list[dict[str, str]]:
    diagnostics = "\n".join(f"- {issue}" for issue in issues)
    repair_request = (
        "Repair the previous answer.\n"
        "Keep the same user intent, fix only the listed problems, and return only the corrected final code artifact.\n"
        f"{diagnostics}"
    )

    # Во втором проходе даем модели:
    # 1) исходную задачу
    # 2) плохой ответ
    # 3) список конкретных проблем от валидатора
    # Это намного стабильнее, чем просто просить "попробуй еще раз".
    return [
        {
            "role": "system",
            "content": build_system_prompt(
                target_output_format,
                mode=mode,
                retrieved_examples=retrieved_examples,
            ),
        },
        {
            "role": "user",
            "content": build_task_payload(
                user_text,
                context=context,
                existing_code=existing_code,
                feedback=feedback,
            ),
        },
        {"role": "assistant", "content": bad_output.strip()},
        {"role": "user", "content": repair_request},
    ]


def run_agent_flow(
    user_text: str,
    *,
    model: str = DEFAULT_MODEL,
    context: Any | None = None,
    existing_code: str | None = None,
    feedback: str | None = None,
    target_output_format: str = "auto",
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    allow_clarification: bool = True,
) -> AgentResult:
    analysis = analyze_task(
        user_text,
        context=context,
        existing_code=existing_code,
        feedback=feedback,
        target_output_format=target_output_format,
    )

    if allow_clarification and analysis.needs_clarification:
        return AgentResult(
            status="needs_clarification",
            code=None,
            clarification_question=analysis.clarification_question,
            analysis=analysis,
        )

    validation_user_text = compose_user_text(user_text, context=context)
    retrieved_examples = select_few_shot_examples(
        validation_user_text,
        analysis=analysis,
    )
    current_output = maybe_resolve_template(
        validation_user_text,
        target_output_format=analysis.resolved_output_format,
    )
    used_template = current_output is not None

    if current_output is None:
        messages = build_initial_messages(
            user_text=user_text,
            target_output_format=analysis.resolved_output_format,
            mode=analysis.mode,
            context=context,
            existing_code=existing_code,
            feedback=feedback,
            retrieved_examples=retrieved_examples,
        )
        current_output = call_llm(model=model, messages=messages)
        current_output = canonicalize_output_for_target(
            current_output,
            analysis.resolved_output_format,
        )

    # Небольшой управляемый цикл: после каждой генерации проверяем ответ и,
    # если нужно, просим модель исправить только конкретные проблемы.
    for attempt in range(max_repair_attempts + 1):
        issues = validate_agent_output(
            current_output,
            target_output_format=analysis.resolved_output_format,
            user_text=validation_user_text,
        )
        if not issues:
            return AgentResult(
                status="completed",
                code=current_output,
                clarification_question=None,
                analysis=analysis,
            )

        if used_template:
            used_template = False
            messages = build_initial_messages(
                user_text=user_text,
                target_output_format=analysis.resolved_output_format,
                mode=analysis.mode,
                context=context,
                existing_code=existing_code,
                feedback=feedback,
                retrieved_examples=retrieved_examples,
            )
            current_output = call_llm(model=model, messages=messages)
            current_output = canonicalize_output_for_target(
                current_output,
                analysis.resolved_output_format,
            )
            continue

        if attempt == max_repair_attempts:
            break

        repair_messages = build_repair_messages(
            user_text=user_text,
            bad_output=current_output,
            issues=issues,
            target_output_format=analysis.resolved_output_format,
            mode=analysis.mode,
            context=context,
            existing_code=existing_code,
            feedback=feedback,
            retrieved_examples=retrieved_examples,
        )
        current_output = call_llm(model=model, messages=repair_messages)
        current_output = canonicalize_output_for_target(
            current_output,
            analysis.resolved_output_format,
        )

    return AgentResult(
        status="completed",
        code=current_output,
        clarification_question=None,
        analysis=analysis,
    )


def run_agent(
    user_text: str,
    model: str = DEFAULT_MODEL,
    target_output_format: str = "auto",
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    *,
    context: Any | None = None,
    existing_code: str | None = None,
    feedback: str | None = None,
) -> str:
    # Обратная совместимость для ноутбука и старых скриптов:
    # старый run_agent по-прежнему возвращает только код.
    result = run_agent_flow(
        user_text=user_text,
        model=model,
        context=context,
        existing_code=existing_code,
        feedback=feedback,
        target_output_format=target_output_format,
        max_repair_attempts=max_repair_attempts,
        allow_clarification=False,
    )
    return result.code or ""


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_NUM_CTX",
    "DEFAULT_NUM_PREDICT",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_SEED",
    "MAX_REPAIR_ATTEMPTS",
    "OUTPUT_FORMATS",
    "TaskAnalysis",
    "AgentResult",
    "compose_user_text",
    "extract_referenced_paths",
    "extract_paths_from_context",
    "resolve_output_format",
    "analyze_task",
    "select_few_shot_examples",
    "maybe_resolve_template",
    "build_system_prompt",
    "build_task_payload",
    "build_initial_messages",
    "build_ollama_client",
    "call_llm",
    "normalize_model_output",
    "canonicalize_output_for_target",
    "validate_agent_output",
    "build_repair_messages",
    "run_agent_flow",
    "run_agent",
]
