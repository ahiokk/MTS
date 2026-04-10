from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from ollama import Client
from app.validator import validate_lua

DEFAULT_MODEL = os.getenv("LOCALSCRIPT_MODEL", "qwen2.5-coder:7b")
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MAX_REPAIR_ATTEMPTS = 2

# Эти режимы нужны, чтобы система не "угадывала" формат ответа каждый раз.
# Мы либо задаем формат явно, либо разрешаем модели выбрать его по простым правилам.
OUTPUT_FORMATS = {
    "auto",
    "raw_lua",
    "lowcode_lua_fragment",
    "json_with_lua_fields",
}


def build_system_prompt(target_output_format: str = "auto") -> str:
    if target_output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported target_output_format: {target_output_format}")

    # Главная идея промпта: модель должна возвращать только финальный код,
    # без "агентских" оберток вроде RESULT / Code / Notes.
    return f"""
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

LowCode environment rules:
1. Workflow variables are stored in wf.vars.
2. Initial input variables passed at startup are stored in wf.initVariables.
3. Use lua{{ ... }}lua only when the required output format needs a LowCode script string.
4. When creating arrays, prefer _utils.array.new() when appropriate.
5. Allowed control flow is basic Lua: if, while, for, repeat-until.
6. Keep the solution simple and compatible with the examples.

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

You must return only one final artifact in the selected target format.
""".strip()


def build_initial_messages(
    user_text: str,
    target_output_format: str = "auto",
) -> list[dict[str, str]]:
    cleaned_text = user_text.strip()
    if not cleaned_text:
        raise ValueError("user_text must not be empty")

    # Ollama chat API ожидает список сообщений в формате system/user/assistant.
    return [
        {"role": "system", "content": build_system_prompt(target_output_format)},
        {"role": "user", "content": cleaned_text},
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


def build_ollama_client(host: str = DEFAULT_OLLAMA_HOST) -> Client:
    return Client(host=host)


def call_llm(model: str, messages: Sequence[Mapping[str, str]]) -> str:
    try:
        client = build_ollama_client()
        response = client.chat(
            model=model,
            messages=list(messages),
            options={"temperature": 0},
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
        {"role": "system", "content": build_system_prompt(target_output_format)},
        {"role": "user", "content": user_text.strip()},
        {"role": "assistant", "content": bad_output.strip()},
        {"role": "user", "content": repair_request},
    ]


def run_agent(
    user_text: str,
    model: str = DEFAULT_MODEL,
    target_output_format: str = "auto",
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
) -> str:
    messages = build_initial_messages(
        user_text=user_text,
        target_output_format=target_output_format,
    )
    current_output = call_llm(model=model, messages=messages)

    # Небольшой управляемый цикл: после каждой генерации проверяем ответ и,
    # если нужно, просим модель исправить только конкретные проблемы.
    for _ in range(max_repair_attempts + 1):
        issues = validate_agent_output(
            current_output,
            target_output_format=target_output_format,
            user_text=user_text,
        )
        if not issues:
            return current_output

        repair_messages = build_repair_messages(
            user_text=user_text,
            bad_output=current_output,
            issues=issues,
            target_output_format=target_output_format,
        )
        current_output = call_llm(model=model, messages=repair_messages)

    return current_output


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_HOST",
    "MAX_REPAIR_ATTEMPTS",
    "OUTPUT_FORMATS",
    "build_system_prompt",
    "build_initial_messages",
    "build_ollama_client",
    "call_llm",
    "normalize_model_output",
    "validate_agent_output",
    "build_repair_messages",
    "run_agent",
]
