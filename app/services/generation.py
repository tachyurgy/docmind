import json
from collections.abc import AsyncGenerator

import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.config import settings
from app.services.retrieval import SearchResult

logger = structlog.get_logger()

SYSTEM_PROMPT = """You are a document Q&A assistant. Answer the user's question based only on the provided context chunks from their uploaded documents.

Rules:
- Only use information from the provided context to answer.
- If the context does not contain enough information to answer, say so clearly.
- Cite your sources by referencing the document name and page number when available.
- Use the format [Source: document_name, p.X] for citations inline.
- Be concise and direct."""


def _build_context(chunks: list[SearchResult]) -> str:
    """Format retrieved chunks into a context string for the generation model."""
    parts: list[str] = []
    for i, result in enumerate(chunks):
        page_info = f", page {result.chunk.page_number}" if result.chunk.page_number else ""
        header = f"[{i + 1}] {result.document_name}{page_info} (relevance: {result.combined_score:.2f})"
        parts.append(f"{header}\n{result.chunk.content}")
    return "\n\n---\n\n".join(parts)


class GenerationService:
    """Generates answers using retrieved context chunks.

    Uses Anthropic Claude as the primary model and falls back to OpenAI
    if Claude is unavailable or not configured.
    """

    def __init__(self) -> None:
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

        if settings.anthropic_api_key:
            self._anthropic = AsyncAnthropic(api_key=settings.anthropic_api_key)
        if settings.openai_api_key:
            self._openai = AsyncOpenAI(api_key=settings.openai_api_key)

    async def generate_answer(
        self,
        query: str,
        context_chunks: list[SearchResult],
        stream: bool = True,
    ) -> AsyncGenerator[str, None] | tuple[str, str]:
        """Generate an answer from the query and context.

        If stream=True, returns an async generator yielding SSE-formatted strings.
        If stream=False, returns a tuple of (answer_text, model_used).
        """
        context = _build_context(context_chunks)
        user_message = f"Context:\n{context}\n\nQuestion: {query}"

        if stream:
            return self._stream_answer(user_message)
        else:
            return await self._complete_answer(user_message)

    async def _stream_answer(self, user_message: str) -> AsyncGenerator[str, None]:
        """Stream the answer as SSE events, trying Claude first then OpenAI."""
        if self._anthropic:
            try:
                async for chunk in self._stream_claude(user_message):
                    yield chunk
                return
            except Exception as e:
                logger.warning("claude_stream_failed", error=str(e))

        if self._openai:
            async for chunk in self._stream_openai(user_message):
                yield chunk
            return

        yield self._sse_event({"error": "No generation model configured"}, event="error")

    async def _stream_claude(self, user_message: str) -> AsyncGenerator[str, None]:
        """Stream response from Claude."""
        yield self._sse_event({"model": "claude-sonnet-4-20250514"}, event="metadata")

        async with self._anthropic.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for text in stream.text_stream:
                yield self._sse_event({"token": text}, event="token")

        yield self._sse_event({"done": True}, event="done")

    async def _stream_openai(self, user_message: str) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI as fallback."""
        yield self._sse_event({"model": "gpt-4o-mini"}, event="metadata")

        response = await self._openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=2048,
            stream=True,
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                yield self._sse_event({"token": token}, event="token")

        yield self._sse_event({"done": True}, event="done")

    async def _complete_answer(self, user_message: str) -> tuple[str, str]:
        """Generate a complete (non-streaming) answer."""
        if self._anthropic:
            try:
                response = await self._anthropic.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
                return response.content[0].text, "claude-sonnet-4-20250514"
            except Exception as e:
                logger.warning("claude_completion_failed", error=str(e))

        if self._openai:
            response = await self._openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=2048,
            )
            return response.choices[0].message.content, "gpt-4o-mini"

        raise RuntimeError("No generation model configured")

    @staticmethod
    def _sse_event(data: dict, event: str = "message") -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"
