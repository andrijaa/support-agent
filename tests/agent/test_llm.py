"""Tests for agent.llm — SupportTicketData + LLMClient."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.llm import SupportTicketData, LLMClient, VALID_CATEGORIES, VALID_URGENCY


class AsyncIterFromList:
    """Wrap a list into a proper async iterator for use with `async for`."""
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


class TestSupportTicketData:
    def test_empty_ticket_not_complete(self):
        ticket = SupportTicketData()
        assert not ticket.is_complete()

    def test_all_fields_set_is_complete(self):
        ticket = SupportTicketData(
            order_number="ABC123",
            problem_category="billing",
            problem_description="I was charged twice for the same order",
            urgency_level="high",
        )
        assert ticket.is_complete()

    def test_short_description_not_complete(self):
        ticket = SupportTicketData(
            order_number="ABC123",
            problem_category="billing",
            problem_description="short",  # < 10 chars
            urgency_level="high",
        )
        assert not ticket.is_complete()

    def test_missing_fields_returns_all_when_empty(self):
        ticket = SupportTicketData()
        missing = ticket.missing_fields()
        assert set(missing) == {"order_number", "problem_category", "problem_description", "urgency_level"}

    def test_missing_fields_returns_only_unfilled(self):
        ticket = SupportTicketData(
            order_number="ABC123",
            problem_category="billing",
        )
        missing = ticket.missing_fields()
        assert "order_number" not in missing
        assert "problem_category" not in missing
        assert "problem_description" in missing
        assert "urgency_level" in missing

    def test_to_dict_includes_all_fields(self):
        ticket = SupportTicketData(
            order_number="ABC123",
            problem_category="billing",
            problem_description="double charge on account",
            urgency_level="high",
        )
        d = ticket.to_dict()
        assert d["order_number"] == "ABC123"
        assert d["problem_category"] == "billing"
        assert d["problem_description"] == "double charge on account"
        assert d["urgency_level"] == "high"
        assert "resolved" in d

    def test_resolved_defaults_false(self):
        ticket = SupportTicketData()
        assert ticket.resolved is False
        assert ticket.to_dict()["resolved"] is False

    def test_is_complete_ignores_resolved(self):
        ticket = SupportTicketData(
            order_number="ABC123",
            problem_category="billing",
            problem_description="I was charged twice for the same order",
            urgency_level="high",
            resolved=False,
        )
        assert ticket.is_complete()
        ticket.resolved = True
        assert ticket.is_complete()


class TestLLMClientGetSummary:
    def test_empty_ticket_fallback(self):
        client = LLMClient(api_key="test", system_prompt="test")
        assert client.get_summary() == "Support ticket details collected"

    def test_summary_includes_fields(self):
        client = LLMClient(api_key="test", system_prompt="test")
        client.ticket = SupportTicketData(
            order_number="ORD456",
            problem_category="shipping",
            urgency_level="medium",
            problem_description="Package arrived damaged",
        )
        summary = client.get_summary()
        assert "Order #ORD456" in summary
        assert "shipping issue" in summary
        assert "urgency: medium" in summary
        assert "Package arrived damaged" in summary

    def test_summary_truncates_long_description(self):
        client = LLMClient(api_key="test", system_prompt="test")
        long_desc = "A" * 100
        client.ticket = SupportTicketData(problem_description=long_desc)
        summary = client.get_summary()
        assert "..." in summary
        # The truncated part should be 80 chars
        assert f'"{"A" * 80}..."' in summary


class TestLLMClientStreamChat:
    @patch("openai.AsyncOpenAI")
    async def test_stream_chat_appends_user_message(self, MockAsyncOpenAI):
        # Set up mock streaming response
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Hello"

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " there"

        mock_response = AsyncIterFromList([mock_chunk, mock_chunk2])

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        MockAsyncOpenAI.return_value = mock_client

        llm = LLMClient(api_key="test-key", system_prompt="You are helpful")

        tokens = []
        async for token in llm.stream_chat("What is Python?"):
            tokens.append(token)

        assert tokens == ["Hello", " there"]
        # User message should be in history
        assert llm.history[0] == {"role": "user", "content": "What is Python?"}

    @patch("openai.AsyncOpenAI")
    async def test_stream_chat_appends_assistant_after_consume(self, MockAsyncOpenAI):
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Response"

        mock_response = AsyncIterFromList([mock_chunk])

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        MockAsyncOpenAI.return_value = mock_client

        llm = LLMClient(api_key="test-key", system_prompt="test")

        # Fully consume the generator
        async for _ in llm.stream_chat("Hi"):
            pass

        assert len(llm.history) == 2
        assert llm.history[1]["role"] == "assistant"
        assert llm.history[1]["content"] == "Response"


class TestLLMClientExtraction:
    @patch("agent.llm.LLMClient._get_sync_client")
    async def test_extract_ticket_data_updates_fields(self, mock_get_client):
        mock_client = MagicMock()
        extraction_response = json.dumps({
            "order_number": "ORD789",
            "problem_category": "Billing",
            "problem_description": "Charged twice for item, need refund ASAP",
            "urgency_level": "HIGH",
        })
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=extraction_response))]
        )
        mock_get_client.return_value = mock_client

        llm = LLMClient(api_key="test", system_prompt="test")
        llm.history = [
            {"role": "user", "content": "Order ORD789, charged twice"},
            {"role": "assistant", "content": "I understand."},
        ]

        await llm._extract_ticket_data()

        assert llm.ticket.order_number == "ORD789"
        assert llm.ticket.problem_category == "billing"  # normalized lowercase
        assert llm.ticket.urgency_level == "high"  # normalized lowercase
        assert llm.ticket.problem_description == "Charged twice for item, need refund ASAP"

    @patch("agent.llm.LLMClient._get_sync_client")
    async def test_extract_rejects_invalid_category(self, mock_get_client):
        mock_client = MagicMock()
        extraction_response = json.dumps({
            "order_number": "ORD789",
            "problem_category": "InvalidCategory",
            "problem_description": None,
            "urgency_level": "INVALID_URGENCY",
        })
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=extraction_response))]
        )
        mock_get_client.return_value = mock_client

        llm = LLMClient(api_key="test", system_prompt="test")
        llm.history = [{"role": "user", "content": "test"}]

        await llm._extract_ticket_data()

        assert llm.ticket.order_number == "ORD789"
        # Invalid category → normalized to "other"
        assert llm.ticket.problem_category == "other"
        # Invalid urgency → skipped (stays None)
        assert llm.ticket.urgency_level is None
