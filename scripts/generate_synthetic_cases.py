from __future__ import annotations

import argparse
import json
from pathlib import Path


def _prompt(instruction: str, context: dict) -> str:
    return f"{instruction}\n{json.dumps(context, ensure_ascii=False, indent=2)}"


def _raw_case(
    case_id: str,
    title: str,
    instruction: str,
    context: dict,
    required_regexes: list[str],
) -> dict:
    return {
        "id": case_id,
        "title": title,
        "source": "Synthetic templates v1",
        "target_output_format": "raw_lua",
        "prompt": _prompt(instruction, context),
        "required_regexes": required_regexes,
    }


def _json_case(
    case_id: str,
    title: str,
    instruction: str,
    context: dict | None,
    required_regexes: list[str],
) -> dict:
    prompt = instruction
    if context is not None:
        prompt = _prompt(instruction, context)

    return {
        "id": case_id,
        "title": title,
        "source": "Synthetic templates v1",
        "target_output_format": "json_with_lua_fields",
        "prompt": prompt,
        "required_regexes": required_regexes,
    }


def build_synthetic_cases_v1() -> list[dict]:
    return [
        _raw_case(
            case_id="last_tracking_code_raw",
            title="Последний код отслеживания",
            instruction="Из массива trackingCodes получи последний код отслеживания.",
            context={
                "wf": {
                    "vars": {
                        "trackingCodes": ["TR-001", "TR-002", "TR-003", "TR-004"]
                    }
                }
            },
            required_regexes=[
                r"return\s+wf\.vars\.trackingCodes\s*\[\s*#wf\.vars\.trackingCodes\s*\]",
            ],
        ),
        _json_case(
            case_id="last_customer_json",
            title="Последний customer id в JSON",
            instruction="Сформируй поле lastCustomerId из последнего элемента массива customerIds.",
            context={
                "wf": {
                    "vars": {
                        "customerIds": ["C-100", "C-200", "C-300"]
                    }
                }
            },
            required_regexes=[
                r'"lastCustomerId"\s*:',
                r"wf\.vars\.customerIds\s*\[\s*#wf\.vars\.customerIds\s*\]",
            ],
        ),
        _json_case(
            case_id="retry_counter_json",
            title="Инкремент retry counter",
            instruction="Увеличивай значение переменной retryCounter на каждой итерации.",
            context={
                "wf": {
                    "vars": {
                        "retryCounter": 7
                    }
                }
            },
            required_regexes=[
                r'"retryCounter"\s*:',
                r"wf\.vars\.retryCounter\s*\+\s*1",
            ],
        ),
        _raw_case(
            case_id="attempt_index_raw",
            title="Следующая попытка raw",
            instruction="Верни следующее значение переменной attempt_index.",
            context={
                "wf": {
                    "vars": {
                        "attempt_index": 12
                    }
                }
            },
            required_regexes=[
                r"return\s+wf\.vars\.attempt_index\s*\+\s*1",
            ],
        ),
        _json_case(
            case_id="cleanup_partner_payload_json",
            title="Очистка payload с новыми ключами",
            instruction="Для полученных данных из предыдущего REST запроса очисти значения переменных DOC_ID, STATUS, MESSAGE.",
            context={
                "wf": {
                    "vars": {
                        "RESTbody": {
                            "result": [
                                {
                                    "DOC_ID": "A-101",
                                    "STATUS": "READY",
                                    "MESSAGE": "ok",
                                    "PRICE": "1500",
                                    "CURRENCY": "RUB",
                                },
                                {
                                    "DOC_ID": "A-102",
                                    "STATUS": "FAILED",
                                    "MESSAGE": "retry",
                                    "ERROR_CODE": "E42",
                                    "REASON": "timeout",
                                },
                            ]
                        }
                    }
                }
            },
            required_regexes=[
                r'"result"\s*:',
                r"wf\.vars\.RESTbody\.result",
                r'key\s*~=\s*["\']DOC_ID["\']',
                r'key\s*~=\s*["\']STATUS["\']',
                r'key\s*~=\s*["\']MESSAGE["\']',
                r"filteredEntry\s*\[\s*key\s*\]\s*=\s*nil",
            ],
        ),
        _raw_case(
            case_id="iso_8601_variant_raw",
            title="ISO 8601 вариант",
            instruction="Преобразуй время из формата 'YYYYMMDD' и 'HHMMSS' в строку в формате ISO 8601 с использованием Lua.",
            context={
                "wf": {
                    "vars": {
                        "json": {
                            "IDOC": {
                                "ZCDF_HEAD": {
                                    "DATUM": "20251201",
                                    "TIME": "081530",
                                }
                            }
                        }
                    }
                }
            },
            required_regexes=[
                r"wf\.vars\.json\.IDOC\.ZCDF_HEAD\.DATUM",
                r"wf\.vars\.json\.IDOC\.ZCDF_HEAD\.TIME",
                r"string\.format\(",
                r"\.00000Z",
            ],
        ),
        _raw_case(
            case_id="ensure_items_arrays_variant_raw",
            title="Нормализация items в массивы",
            instruction="Как преобразовать структуру данных так, чтобы все элементы items в ZCDF_PACKAGES всегда были представлены в виде массивов, даже если они изначально не являются массивами?",
            context={
                "wf": {
                    "vars": {
                        "json": {
                            "IDOC": {
                                "ZCDF_HEAD": {
                                    "ZCDF_PACKAGES": [
                                        {"items": [{"sku": "X1"}]},
                                        {"items": {"sku": "X2"}},
                                        {"items": {"sku": "X3", "qty": 2}},
                                    ]
                                }
                            }
                        }
                    }
                }
            },
            required_regexes=[
                r"function\s+ensureArray",
                r"type\(t\)\s*~=\s*[\"']table[\"']",
                r"math\.floor\(k\)\s*~=\s*k",
                r"obj\.items\s*=\s*ensureArray\(obj\.items\)",
                r"return\s+ensureAllItemsAreArrays\(wf\.vars\.json\.IDOC\.ZCDF_HEAD\.ZCDF_PACKAGES\)",
            ],
        ),
        _raw_case(
            case_id="filter_bonus_or_promoprice_raw",
            title="Фильтрация по Bonus или PromoPrice",
            instruction="Отфильтруй элементы из массива, чтобы включить только те, у которых есть значения в полях Bonus или PromoPrice.",
            context={
                "wf": {
                    "vars": {
                        "catalogRows": [
                            {"SKU": "B001", "Bonus": "yes", "PromoPrice": ""},
                            {"SKU": "B002", "Bonus": "", "PromoPrice": "799"},
                            {"SKU": "B003", "Bonus": None, "PromoPrice": None},
                            {"SKU": "B004", "Bonus": "", "PromoPrice": ""},
                        ]
                    }
                }
            },
            required_regexes=[
                r"_utils\.array\.new\(\)",
                r"wf\.vars\.catalogRows",
                r"item\.Bonus\s*~=\s*\"\"\s*and\s*item\.Bonus\s*~=\s*nil",
                r"item\.PromoPrice\s*~=\s*\"\"\s*and\s*item\.PromoPrice\s*~=\s*nil",
                r"table\.insert\(",
            ],
        ),
        _json_case(
            case_id="square_nine_json",
            title="Квадрат числа 9",
            instruction="Добавь переменную с квадратом числа 9. Сохрани исходное число в поле num.",
            context=None,
            required_regexes=[
                r'"num"\s*:',
                r'"squared"\s*:',
                r"(tonumber\(['\"]9['\"]\)|\b9\b)",
            ],
        ),
        _json_case(
            case_id="unix_time_variant_json",
            title="Unix timestamp с другим значением",
            instruction="Конвертируй время в переменной recallTime в unix-формат.",
            context={
                "wf": {
                    "initVariables": {
                        "recallTime": "2024-02-29T23:59:58+03:00"
                    }
                }
            },
            required_regexes=[
                r'"unix_time"\s*:',
                r"wf\.initVariables\.recallTime",
                r"local\s+function\s+days_since_epoch",
                r"local\s+function\s+parse_iso8601_to_epoch",
            ],
        ),
    ]


def build_synthetic_cases_v2() -> list[dict]:
    return [
        _raw_case(
            case_id="last_shipment_id_raw_init",
            title="Последний shipment id из initVariables",
            instruction="Из массива shipmentIds в initVariables верни последний shipment id.",
            context={
                "wf": {
                    "initVariables": {
                        "shipmentIds": ["SHP-01", "SHP-02", "SHP-03", "SHP-04"]
                    }
                }
            },
            required_regexes=[
                r"return\s+wf\.initVariables\.shipmentIds\s*\[\s*#wf\.initVariables\.shipmentIds\s*\]",
            ],
        ),
        _json_case(
            case_id="latest_invoice_json_en",
            title="Latest invoice code in JSON",
            instruction="Build field latestInvoiceCode from the last element of invoiceCodes.",
            context={
                "wf": {
                    "vars": {
                        "invoiceCodes": ["INV-001", "INV-002", "INV-003"]
                    }
                }
            },
            required_regexes=[
                r'"latestInvoiceCode"\s*:',
                r"wf\.vars\.invoiceCodes\s*\[\s*#wf\.vars\.invoiceCodes\s*\]",
            ],
        ),
        _raw_case(
            case_id="increment_fail_count_raw_en",
            title="Increment failCount raw",
            instruction="Increment variable failCount by 1 and return the next value.",
            context={
                "wf": {
                    "vars": {
                        "failCount": 2
                    }
                }
            },
            required_regexes=[
                r"return\s+wf\.vars\.failCount\s*\+\s*1",
            ],
        ),
        _json_case(
            case_id="increment_reprocess_json_ru",
            title="Инкремент reprocess_count в JSON",
            instruction="Увеличивай значение переменной reprocess_count на каждой итерации и верни результат в поле reprocess_count.",
            context={
                "wf": {
                    "vars": {
                        "reprocess_count": 4
                    }
                }
            },
            required_regexes=[
                r'"reprocess_count"\s*:',
                r"wf\.vars\.reprocess_count\s*\+\s*1",
            ],
        ),
        _json_case(
            case_id="cleanup_rows_json_en",
            title="Cleanup payload with English instruction",
            instruction="For the previous REST response clear values in fields ORDER_ID, STATE, COMMENT and remove every other key.",
            context={
                "wf": {
                    "vars": {
                        "RESTbody": {
                            "result": [
                                {
                                    "ORDER_ID": "O-1001",
                                    "STATE": "created",
                                    "COMMENT": "ok",
                                    "AMOUNT": "500",
                                    "CURRENCY": "USD"
                                },
                                {
                                    "ORDER_ID": "O-1002",
                                    "STATE": "failed",
                                    "COMMENT": "retry",
                                    "ERROR_CODE": "E17",
                                    "DETAILS": "network"
                                }
                            ]
                        }
                    }
                }
            },
            required_regexes=[
                r'"result"\s*:',
                r"wf\.vars\.RESTbody\.result",
                r'key\s*~=\s*["\']ORDER_ID["\']',
                r'key\s*~=\s*["\']STATE["\']',
                r'key\s*~=\s*["\']COMMENT["\']',
                r"filteredEntry\s*\[\s*key\s*\]\s*=\s*nil",
            ],
        ),
        _raw_case(
            case_id="iso_8601_noisy_raw",
            title="ISO 8601 with noisy context",
            instruction="Convert DATUM and TIME into an ISO 8601 string using Lua.",
            context={
                "wf": {
                    "vars": {
                        "json": {
                            "IDOC": {
                                "ZCDF_HEAD": {
                                    "DATUM": "20261130",
                                    "TIME": "235959",
                                    "STATUS": "READY",
                                    "ROUTE": "Night line"
                                }
                            }
                        },
                        "meta": {
                            "traceId": "T-900"
                        }
                    }
                }
            },
            required_regexes=[
                r"wf\.vars\.json\.IDOC\.ZCDF_HEAD\.DATUM",
                r"wf\.vars\.json\.IDOC\.ZCDF_HEAD\.TIME",
                r"string\.format\(",
                r"\.00000Z",
            ],
        ),
        _raw_case(
            case_id="ensure_items_arrays_noisy_raw",
            title="Ensure items arrays with noisy payload",
            instruction="Как преобразовать структуру данных так, чтобы все элементы items в ZCDF_PACKAGES всегда были представлены в виде массивов, даже если они изначально не являются массивами?",
            context={
                "wf": {
                    "vars": {
                        "json": {
                            "IDOC": {
                                "ZCDF_HEAD": {
                                    "ZCDF_PACKAGES": [
                                        {"id": "P1", "items": [{"sku": "A1"}]},
                                        {"id": "P2", "items": {"sku": "A2"}},
                                        {"id": "P3", "items": {"sku": "A3", "qty": 4}}
                                    ],
                                    "DELIVERY": "D-100"
                                }
                            }
                        }
                    }
                }
            },
            required_regexes=[
                r"function\s+ensureArray",
                r"type\(t\)\s*~=\s*[\"']table[\"']",
                r"math\.floor\(k\)\s*~=\\s*k|math\.floor\(k\)\s*~=\s*k",
                r"obj\.items\s*=\s*ensureArray\(obj\.items\)",
                r"return\s+ensureAllItemsAreArrays\(wf\.vars\.json\.IDOC\.ZCDF_HEAD\.ZCDF_PACKAGES\)",
            ],
        ),
        {
            "id": "filter_bonus_fragment_v2",
            "title": "LowCode fragment for Bonus or PromoPrice filter",
            "source": "Synthetic templates v2",
            "target_output_format": "lowcode_lua_fragment",
            "prompt": _prompt(
                "Сформируй LowCode fragment, который фильтрует catalogRows и оставляет только элементы с заполненными полями Bonus или PromoPrice.",
                {
                    "wf": {
                        "vars": {
                            "catalogRows": [
                                {"SKU": "B001", "Bonus": "yes", "PromoPrice": ""},
                                {"SKU": "B002", "Bonus": "", "PromoPrice": "799"},
                                {"SKU": "B003", "Bonus": None, "PromoPrice": None},
                                {"SKU": "B004", "Bonus": "", "PromoPrice": ""}
                            ]
                        }
                    }
                },
            ),
            "required_regexes": [
                r"^lua\{[\s\S]*\}lua$",
                r"(_utils\.array\.new\(\)|table\.remove\()",
                r"wf\.vars\.catalogRows",
                r"(table\.insert\(|table\.remove\()",
            ],
        },
        _json_case(
            case_id="square_twelve_json_en",
            title="Square of number 12",
            instruction="Add a variable with the square of number 12 and keep the source value in field num.",
            context=None,
            required_regexes=[
                r'"num"\s*:',
                r'"squared"\s*:',
                r"(tonumber\(['\"]12['\"]\)|\b12\b)",
            ],
        ),
        _json_case(
            case_id="unix_time_millis_json",
            title="Unix time with milliseconds and offset",
            instruction="Конвертируй время в переменной recallTime в unix-формат.",
            context={
                "wf": {
                    "initVariables": {
                        "recallTime": "2024-07-01T08:15:30.123-05:00"
                    }
                }
            },
            required_regexes=[
                r'"unix_time"\s*:',
                r"wf\.initVariables\.recallTime",
                r"local\s+function\s+days_since_epoch",
                r"local\s+function\s+parse_iso8601_to_epoch",
            ],
        ),
    ]


def build_synthetic_cases() -> list[dict]:
    return build_synthetic_cases_v1()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic LocalScript regression cases."
    )
    parser.add_argument(
        "--output",
        help="Path to save the generated synthetic cases JSON.",
    )
    parser.add_argument(
        "--variant",
        choices=("v1", "v2", "all"),
        default="v1",
        help="Synthetic case set to generate.",
    )
    args = parser.parse_args()

    if args.variant == "v1":
        cases = build_synthetic_cases_v1()
        default_output = Path("eval/synthetic_cases.json")
    elif args.variant == "v2":
        cases = build_synthetic_cases_v2()
        default_output = Path("eval/synthetic_cases_v2.json")
    else:
        cases = build_synthetic_cases_v1() + build_synthetic_cases_v2()
        default_output = Path("eval/synthetic_cases_all.json")

    output_path = Path(args.output) if args.output else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"cases": cases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Generated {len(cases)} synthetic cases at {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
