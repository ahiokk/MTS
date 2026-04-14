from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request


COMMON_FORBIDDEN_SUBSTRINGS = (
    "RESULT",
    "CLARIFICATION_NEEDED",
    "```",
    "Code:",
    "Notes:",
    "JsonPath",
    "TODO",
    "FIXME",
)


def _post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {path}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to reach {base_url}{path}: {exc.reason}") from exc


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No cases found in {path}")
    return cases


def _check_required_regexes(output: str, patterns: list[str]) -> list[str]:
    missing: list[str] = []
    for pattern in patterns:
        if re.search(pattern, output, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL) is None:
            missing.append(pattern)
    return missing


def _check_forbidden_substrings(output: str, forbidden: list[str]) -> list[str]:
    return [item for item in forbidden if item in output]


def run_case(
    case: dict[str, Any],
    *,
    base_url: str,
    max_repair_attempts: int,
    request_timeout_seconds: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    prompt = case["prompt"]
    target_output_format = case["target_output_format"]

    try:
        generation_payload: dict[str, Any] = {
            "prompt": prompt,
            "target_output_format": target_output_format,
            "max_repair_attempts": max_repair_attempts,
        }
        for optional_key in ("context", "existing_code", "feedback", "allow_clarification"):
            if optional_key in case:
                generation_payload[optional_key] = case[optional_key]

        generation = _post_json(
            base_url,
            "/generate",
            generation_payload,
            timeout_seconds=request_timeout_seconds,
        )
        if generation.get("status") == "needs_clarification":
            raise RuntimeError(
                f"Agent asked for clarification instead of code: {generation.get('clarification_question')}"
            )
        output = generation["code"]

        validation = _post_json(
            base_url,
            "/validate",
            {
                "prompt": prompt,
                "output": output,
                "expected_output_format": target_output_format,
            },
            timeout_seconds=request_timeout_seconds,
        )
    except Exception as exc:
        duration_seconds = round(time.perf_counter() - started_at, 2)
        return {
            "id": case["id"],
            "title": case.get("title", case["id"]),
            "source": case.get("source"),
            "passed": False,
            "duration_seconds": duration_seconds,
            "target_output_format": target_output_format,
            "output": "",
            "validation_ok": False,
            "detected_output_format": None,
            "missing_patterns": [],
            "forbidden_hits": [],
            "issues": [
                {
                    "code": "request_failed",
                    "message": str(exc),
                    "severity": "error",
                }
            ],
        }

    missing_patterns = _check_required_regexes(output, case.get("required_regexes", []))
    forbidden_hits = _check_forbidden_substrings(
        output,
        list(COMMON_FORBIDDEN_SUBSTRINGS) + case.get("forbidden_substrings", []),
    )
    duration_seconds = round(time.perf_counter() - started_at, 2)

    passed = bool(validation.get("ok")) and not missing_patterns and not forbidden_hits
    return {
        "id": case["id"],
        "title": case.get("title", case["id"]),
        "source": case.get("source"),
        "passed": passed,
        "duration_seconds": duration_seconds,
        "target_output_format": target_output_format,
        "output": output,
        "validation_ok": bool(validation.get("ok")),
        "detected_output_format": validation.get("detected_output_format"),
        "missing_patterns": missing_patterns,
        "forbidden_hits": forbidden_hits,
        "issues": validation.get("issues", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run public LocalScript regression cases against the local API."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Base URL of the running LocalScript API.",
    )
    parser.add_argument(
        "--cases",
        default=str(Path("eval/public_cases.json")),
        help="Path to the JSON file with curated regression cases.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the full JSON report.",
    )
    parser.add_argument(
        "--max-repair-attempts",
        type=int,
        default=2,
        help="Repair attempts to pass through to /generate.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=300,
        help="Per-request HTTP timeout in seconds for /generate and /validate.",
    )
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    results = []
    for case in cases:
        results.append(
            run_case(
                case,
                base_url=args.base_url,
                max_repair_attempts=args.max_repair_attempts,
                request_timeout_seconds=args.request_timeout_seconds,
            )
        )

    passed_count = sum(1 for result in results if result["passed"])
    print(f"Regression summary: {passed_count}/{len(results)} passed")
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{status}] {result['id']} "
            f"({result['target_output_format']}, {result['duration_seconds']}s)"
        )
        if not result["passed"]:
            if not result["validation_ok"]:
                print("  validator: failed")
            if result["missing_patterns"]:
                print("  missing patterns:")
                for pattern in result["missing_patterns"]:
                    print(f"    - {pattern}")
            if result["forbidden_hits"]:
                print("  forbidden hits:")
                for hit in result["forbidden_hits"]:
                    print(f"    - {hit}")
            validation_errors = [
                issue["message"]
                for issue in result["issues"]
                if issue.get("severity") == "error"
            ]
            if validation_errors:
                print("  validator issues:")
                for message in validation_errors:
                    print(f"    - {message}")

    report = {
        "summary": {
            "passed": passed_count,
            "total": len(results),
            "base_url": args.base_url,
            "max_repair_attempts": args.max_repair_attempts,
            "request_timeout_seconds": args.request_timeout_seconds,
        },
        "results": results,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved full report to {output_path}")

    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
