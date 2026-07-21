from __future__ import annotations

import argparse
from collections.abc import Collection, Mapping, Sequence
import dataclasses
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
import urllib.request

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONCORDIA_ROOT = ROOT / "external" / "concordia"
sys.path.insert(0, str(CONCORDIA_ROOT))

from concordia.language_model import language_model  # noqa: E402
from examples.conversation_with_ai_companion import (  # noqa: E402
    scenario_01_philosophy_student_exam_prep as scenario,
)
from examples.conversation_with_ai_companion import shared_utils  # noqa: E402


class LlamaCppLanguageModel(language_model.LanguageModel):
    """Minimal Concordia adapter for VivaTrace's local OpenAI-compatible server."""

    def __init__(self, base_url: str, model_name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def _chat(self, prompt: str, max_tokens: int, temperature: float) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Continue the requested simulation text directly. Be concise. "
                        "Do not discuss these instructions."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": min(max_tokens, 220),
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            result = json.loads(response.read().decode("utf-8"))
        return str(result["choices"][0]["message"]["content"]).strip()

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        top_p: float = language_model.DEFAULT_TOP_P,
        top_k: int = language_model.DEFAULT_TOP_K,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        del top_p, top_k, timeout, seed
        result = self._chat(prompt, max_tokens=max_tokens, temperature=min(temperature, 0.7))
        for terminator in terminators:
            if terminator and terminator in result:
                result = result.split(terminator, 1)[0]
        return result

    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, Mapping[str, Any]]:
        del seed
        options = "\n".join(f"{index}: {value}" for index, value in enumerate(responses))
        answer = self._chat(
            f"{prompt}\n\nOptions:\n{options}\nReturn only the option number.",
            max_tokens=12,
            temperature=0.0,
        )
        match = re.search(r"\b(\d+)\b", answer)
        if match and int(match.group(1)) < len(responses):
            index = int(match.group(1))
            return index, responses[index], {"raw": answer}
        for index, response in enumerate(responses):
            if response.lower() in answer.lower():
                return index, response, {"raw": answer}
        return 0, responses[0], {"raw": answer, "fallback": True}


def hash_embedder(text: str, dimensions: int = 64) -> np.ndarray:
    """Small offline embedder sufficient for a smoke-test memory store."""
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in re.findall(r"[a-z0-9']+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dimensions
        vector[index] += 1.0 if digest[4] % 2 else -1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run a short Concordia demo on local llama.cpp.")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="qwen2.5-3b-instruct-q4_k_m.gguf")
    args = parser.parse_args()

    config = dataclasses.replace(scenario.create_config(), default_max_steps=args.steps)
    output_dir = CONCORDIA_ROOT / "local-results"
    result = shared_utils.run_simulation(
        config=config,
        scenario_name="Concordia local education smoke test",
        model=LlamaCppLanguageModel(args.url, args.model),
        embedder=hash_embedder,
        output_dir=str(output_dir),
    )
    print(f"dialog_path={result.get('dialog_path')}")


if __name__ == "__main__":
    main()
