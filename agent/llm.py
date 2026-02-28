"""OpenAI LLM client with conversation history and structured data extraction."""
from __future__ import annotations
import asyncio
import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

MODEL = "gpt-4o"
EXTRACTION_MODEL = "gpt-4o-mini"
MAX_TOKENS = 1024

VALID_CATEGORIES = {"billing", "technical", "shipping", "returns", "account", "product_info", "other"}
VALID_URGENCY = {"low", "medium", "high", "critical"}


@dataclass
class SupportTicketData:
    order_number: Optional[str] = None
    problem_category: Optional[str] = None
    problem_description: Optional[str] = None
    urgency_level: Optional[str] = None
    resolved: bool = False

    def is_complete(self) -> bool:
        """Return True when all required fields are collected."""
        return all([
            self.order_number,
            self.problem_category,
            self.problem_description and len(self.problem_description) >= 10,
            self.urgency_level,
        ])

    def missing_fields(self) -> list[str]:
        missing = []
        if not self.order_number:
            missing.append("order_number")
        if not self.problem_category:
            missing.append("problem_category")
        if not self.problem_description or len(self.problem_description) < 10:
            missing.append("problem_description")
        if not self.urgency_level:
            missing.append("urgency_level")
        return missing

    def to_dict(self) -> dict:
        return asdict(self)


EXTRACTION_PROMPT = """You are a data extractor. Given a conversation history, extract support ticket data into JSON.
Return ONLY valid JSON with these fields (use null for unknown):
{
  "order_number": "string or null (alphanumeric, 6-12 characters)",
  "problem_category": "one of: billing, technical, shipping, returns, account, product_info, other — or null",
  "problem_description": "string or null (brief description of the issue, at least 10 characters)",
  "urgency_level": "one of: low, medium, high, critical — or null"
}
Return ONLY the JSON object, no other text."""


class LLMClient:
    def __init__(self, api_key: Optional[str] = None, system_prompt: str = ""):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.system_prompt = system_prompt
        self.history: list[dict] = []
        self.ticket = SupportTicketData()
        self._sync_client = None
        self._extraction_task: Optional[asyncio.Task] = None

    def _get_sync_client(self):
        if self._sync_client is None:
            from openai import OpenAI
            self._sync_client = OpenAI(api_key=self._api_key)
        return self._sync_client

    async def stream_chat(self, user_text: str):
        """Async generator yielding LLM response tokens. Updates history when complete."""
        from openai import AsyncOpenAI

        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": self.system_prompt}] + self.history

        async_client = AsyncOpenAI(api_key=self._api_key)
        full_response = ""

        response = await async_client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                yield delta

        # Runs only if generator was fully consumed (not cancelled by barge-in)
        self.history.append({"role": "assistant", "content": full_response})
        logger.info("LLM response: %.100s...", full_response)
        self._extraction_task = asyncio.create_task(self._extract_ticket_data())

    async def await_extraction(self) -> None:
        """Wait for the latest extraction task to complete."""
        if self._extraction_task:
            await self._extraction_task
            self._extraction_task = None

    async def _extract_ticket_data(self) -> None:
        """Use gpt-4o-mini to extract structured support ticket data from history."""
        client = self._get_sync_client()
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self.history
        )

        def _extract():
            response = client.chat.completions.create(
                model=EXTRACTION_MODEL,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": history_text},
                ],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        loop = asyncio.get_event_loop()
        try:
            raw = await loop.run_in_executor(None, _extract)
            data = json.loads(raw)
            for field_name, value in data.items():
                if value is not None and hasattr(self.ticket, field_name):
                    # Normalize category and urgency to lowercase
                    if field_name == "problem_category":
                        value = value.lower().strip()
                        if value not in VALID_CATEGORIES:
                            value = "other"
                    elif field_name == "urgency_level":
                        value = value.lower().strip()
                        if value not in VALID_URGENCY:
                            continue
                    setattr(self.ticket, field_name, value)
            logger.info("Ticket data: %s", self.ticket.to_dict())
        except Exception as e:
            logger.warning("Extraction failed: %s", e)

    def get_summary(self) -> str:
        """Generate a one-line summary of the support ticket."""
        t = self.ticket
        parts = []
        if t.order_number:
            parts.append(f"Order #{t.order_number}")
        if t.problem_category:
            parts.append(f"{t.problem_category} issue")
        if t.urgency_level:
            parts.append(f"urgency: {t.urgency_level}")
        if t.problem_description:
            desc = t.problem_description[:80]
            if len(t.problem_description) > 80:
                desc += "..."
            parts.append(f'"{desc}"')
        return " — ".join(parts) if parts else "Support ticket details collected"
