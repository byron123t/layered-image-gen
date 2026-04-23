#!/usr/bin/env python3
"""Query an OpenAI LLM with a schema and prompt; save JSON or raw text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai_api import OpenAIAPI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Query an OpenAI LLM with a JSON schema and a prompt. "
            "The response is written to --output: pretty-printed JSON if parseable, "
            "otherwise the raw model text."
        )
    )
    parser.add_argument(
        "prompt",
        help="Prompt describing the JSON instance to generate.",
    )
    parser.add_argument(
        "--schema",
        default="image_composition.schema.json",
        help="Path to the JSON schema file.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-pro",
        help="Model/deployment name configured in openai_api.py.",
    )
    parser.add_argument(
        "--output",
        default="generated_image_composition.json",
        help="Path to write the generated JSON.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum output tokens for the completion.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterative refinement rounds (>=1).",
    )
    parser.add_argument(
        "--auto-stop",
        action="store_true",
        help=(
            "Let the model decide when to stop refining. "
            "Uses --max-iterations as a hard safety cap."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="Hard cap for refinement steps when --auto-stop is enabled.",
    )
    return parser


def _read_schema(schema_path: Path) -> str:
    with schema_path.open("r", encoding="utf-8") as f:
        return f.read()


def _extract_json_text(raw_text: str) -> str:
    """Accept plain JSON or JSON wrapped in markdown fences."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _call_llm_for_draft(
    prompt: str,
    schema_text: str,
    model: str,
    max_tokens: int,
    previous_draft: str | None = None,
) -> str:
    """Call the LLM and return raw text response."""
    client = OpenAIAPI(model)
    system_prompt = (
        "You are a JSON generator. "
        "Return only a valid JSON object, with no markdown, no comments, and no extra text."
        "The JSON must satisfy the provided JSON Schema."
    )
    if previous_draft is None:
        user_prompt = (
            "Generate one JSON object from the schema below.\n\n"
            "JSON Schema:\n"
            f"{schema_text}\n\n"
            "Generation instructions:\n"
            f"{prompt}. Be detailed when filling out the description.\n\n"
            "Output: Return only the JSON object."
        )
    else:
        user_prompt = (
            "Refine the previous JSON draft so it better satisfies the schema and instructions.\n\n"
            "JSON Schema:\n"
            f"{schema_text}\n\n"
            "Generation instructions:\n"
            f"{prompt}. Be detailed when filling out the description.\n\n"
            "Previous draft to refine:\n"
            f"{previous_draft}\n\n"
            "Output: Return only the improved JSON object."
        )
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _call_llm_for_autostop_step(
    prompt: str,
    schema_text: str,
    model: str,
    max_tokens: int,
    previous_draft: str | None = None,
) -> tuple[str, bool, str]:
    """Return (draft, done, reason) from a model-controlled refinement step."""
    client = OpenAIAPI(model)
    system_prompt = (
        "You are a JSON refinement controller. "
        "Respond with exactly one JSON object and no extra text. "
        'Required keys: "done" (boolean), "json_draft" (string), "reason" (string). '
        '"json_draft" must contain only a JSON object text, without markdown fences.'
    )
    if previous_draft is None:
        user_prompt = (
            "Create an initial JSON draft from the provided schema and instructions.\n\n"
            "JSON Schema:\n"
            f"{schema_text}\n\n"
            "Generation instructions:\n"
            f"{prompt}\n\n"
            "Set done=true only if this draft is already strong and complete. "
            "Otherwise set done=false."
        )
    else:
        user_prompt = (
            "Refine the previous JSON draft.\n\n"
            "JSON Schema:\n"
            f"{schema_text}\n\n"
            "Generation instructions:\n"
            f"{prompt}\n\n"
            "Previous draft:\n"
            f"{previous_draft}\n\n"
            "Set done=true only when no further meaningful refinement is needed."
        )

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content or ""
    candidate = _extract_json_text(raw)
    if not candidate.strip():
        candidate = raw.strip()
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            draft = str(obj.get("json_draft", "")).strip()
            done = bool(obj.get("done", False))
            reason = str(obj.get("reason", "")).strip()
            if draft:
                return draft, done, reason
    except json.JSONDecodeError:
        pass
    fallback = candidate if candidate.strip() else raw
    return fallback, False, "Control output parse failed; continuing with fallback draft."


def _normalize_output(raw_content: str) -> tuple[str, bool]:
    """Return (text to save, whether valid JSON)."""
    candidate = _extract_json_text(raw_content)
    if not candidate.strip():
        candidate = raw_content.strip()
    try:
        parsed = json.loads(candidate)
        body = json.dumps(parsed, indent=2) + "\n"
        return body, True
    except json.JSONDecodeError:
        body = candidate if candidate.strip() else raw_content
        if not body.endswith("\n"):
            body = body + "\n"
        return body, False


def generate_output_text(
    prompt: str,
    schema_text: str,
    model: str,
    max_tokens: int,
    iterations: int,
    auto_stop: bool,
    max_iterations: int,
) -> tuple[str, bool]:
    """Run iterative generation/refinement and return final text."""
    latest_raw = ""
    latest_normalized = ""
    latest_is_json = False
    if auto_stop:
        rounds = max(1, max_iterations)
        for i in range(rounds):
            previous = latest_raw if i > 0 else None
            latest_raw, done, reason = _call_llm_for_autostop_step(
                prompt=prompt,
                schema_text=schema_text,
                model=model,
                max_tokens=max_tokens,
                previous_draft=previous,
            )
            latest_normalized, latest_is_json = _normalize_output(latest_raw)
            print(
                f"Completed iteration {i + 1}/{rounds} (done={done})",
                flush=True,
            )
            if reason:
                print(f"  reason: {reason}", flush=True)
            if done:
                break
    else:
        rounds = max(1, iterations)
        for i in range(rounds):
            previous = latest_raw if i > 0 else None
            latest_raw = _call_llm_for_draft(
                prompt=prompt,
                schema_text=schema_text,
                model=model,
                max_tokens=max_tokens,
                previous_draft=previous,
            )
            latest_normalized, latest_is_json = _normalize_output(latest_raw)
            print(f"Completed iteration {i + 1}/{rounds}", flush=True)
    return latest_normalized, latest_is_json


def main() -> None:
    args = build_arg_parser().parse_args()
    schema_path = Path(args.schema)
    output_path = Path(args.output)
    schema_text = _read_schema(schema_path)
    body, was_json = generate_output_text(
        prompt=args.prompt,
        schema_text=schema_text,
        model=args.model,
        max_tokens=args.max_tokens,
        iterations=args.iterations,
        auto_stop=args.auto_stop,
        max_iterations=args.max_iterations,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(body)
    print(output_path.resolve())
    if not was_json:
        print("Note: response was not valid JSON; saved raw model output.", flush=True)


if __name__ == "__main__":
    main()
