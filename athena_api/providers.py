from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from athena_api.agent import DeterministicProvider, LanguageProvider
from athena_api.settings import Settings, get_settings

SYSTEM_GROUNDING = """You explain Athena Baseball predictions.
Use only the supplied structured prediction. Do not calculate, infer, or invent a number.
Include the stored value, confidence, lineup/starter status, update time, model version,
main evidence, main uncertainty, and every data warning. Never claim certainty or profit.
If a field is absent, say it is unavailable."""


@dataclass
class OpenAIProvider:
    api_key: str
    model: str
    timeout: float
    max_tokens: int
    name: str = "openai"

    def explain(self, question: str, evidence: dict[str, Any], detail_level: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, timeout=self.timeout, max_retries=1)
        response = client.responses.create(
            model=self.model,
            instructions=SYSTEM_GROUNDING,
            input=json.dumps(
                {"question": question, "detail_level": detail_level, "evidence": evidence}
            ),
            max_output_tokens=self.max_tokens,
        )
        return response.output_text


@dataclass
class AnthropicProvider:
    api_key: str
    model: str
    timeout: float
    max_tokens: int
    name: str = "anthropic"

    def explain(self, question: str, evidence: dict[str, Any], detail_level: str) -> str:
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key, timeout=self.timeout, max_retries=1)
        response = client.messages.create(
            model=self.model,
            system=SYSTEM_GROUNDING,
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "detail_level": detail_level,
                            "evidence": evidence,
                        }
                    ),
                }
            ],
            max_tokens=self.max_tokens,
        )
        return "".join(block.text for block in response.content if block.type == "text")


def provider_from_settings(settings: Settings | None = None) -> LanguageProvider:
    settings = settings or get_settings()
    requested = settings.ai_provider.lower()
    if requested in {"auto", "openai"} and settings.openai_api_key:
        return OpenAIProvider(
            settings.openai_api_key,
            settings.ai_model or "gpt-5-mini",
            settings.ai_timeout_seconds,
            settings.ai_max_output_tokens,
        )
    if requested in {"auto", "anthropic"} and settings.anthropic_api_key:
        return AnthropicProvider(
            settings.anthropic_api_key,
            settings.ai_model or "claude-sonnet-4-5",
            settings.ai_timeout_seconds,
            settings.ai_max_output_tokens,
        )
    return DeterministicProvider()
