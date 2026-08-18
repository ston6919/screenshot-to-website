"""Generates code by shelling out to the local `claude` CLI in print mode,
so a run bills against the CLI's own login (subscription) instead of a
metered ANTHROPIC_API_KEY.

ponytail: single-shot only, no tool loop (the app's image-gen /
asset-extraction / screenshot-preview tools aren't wired up here) — add
a tools bridge if those are ever needed through this path.
"""

import asyncio
import base64
import json
import os
import tempfile
from typing import List, Optional

from openai.types.chat import ChatCompletionMessageParam

from agent.providers.base import (
    EventSink,
    ExecutedToolCall,
    ProviderSession,
    ProviderTurn,
    StreamEvent,
)
from llm import Llm

CLI_TIMEOUT_SECONDS = 300


def _flatten_prompt(
    prompt_messages: List[ChatCompletionMessageParam],
) -> tuple[str, List[bytes]]:
    """Turns the OpenAI-style chat messages into one prompt string plus the
    raw bytes of any embedded (data-URL) images."""
    text_parts: List[str] = []
    images: List[bytes] = []
    for message in prompt_messages:
        content = message.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:") and "," in url:
                    images.append(base64.b64decode(url.split(",", 1)[1]))
    return "\n\n".join(p for p in text_parts if p), images


class CliProviderSession(ProviderSession):
    def __init__(self, model: Llm, prompt_messages: List[ChatCompletionMessageParam]):
        self._model = model
        self._prompt_messages = prompt_messages

    async def stream_turn(self, on_event: EventSink) -> ProviderTurn:
        text, images = _flatten_prompt(self._prompt_messages)

        with tempfile.TemporaryDirectory(prefix="s2c-cli-") as tmp_dir:
            image_paths = []
            for i, img_bytes in enumerate(images):
                path = os.path.join(tmp_dir, f"screenshot_{i}.png")
                with open(path, "wb") as f:
                    f.write(img_bytes)
                image_paths.append(path)

            prompt = text
            if image_paths:
                prompt += "\n\nReference screenshot(s) (open with the Read tool):\n" + "\n".join(
                    image_paths
                )
            # The instructions above mention tools (create_file, edit_file,
            # generate_images, ...) that don't exist for you here — you only
            # have Read/Write/Edit. Without this note, models sometimes stop
            # to ask for confirmation instead of just writing the file.
            prompt += (
                "\n\nNote: you don't have create_file/edit_file/image tools "
                "in this environment. Use your own Write tool to create "
                "index.html directly in the current directory — don't ask "
                "for confirmation, just write it."
            )

            # The app's system prompt talks about "create_file" / "edit_file"
            # tools that don't exist for this CLI — it has its own Write/Edit
            # instead. Running with cwd=tmp_dir lets it write index.html
            # there like it would for any other coding task, which we then
            # read back rather than trying to parse code out of chat text.
            proc = await asyncio.create_subprocess_exec(
                "claude",
                "-p",
                prompt,
                "--output-format",
                "json",
                "--allowedTools",
                "Read Write Edit",
                cwd=tmp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=CLI_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                raise RuntimeError(
                    f"claude CLI timed out after {CLI_TIMEOUT_SECONDS}s"
                )

            if proc.returncode != 0:
                raise RuntimeError(
                    f"claude CLI failed: {stderr.decode(errors='replace')}"
                )

            payload = json.loads(stdout.decode())
            written_html = os.path.join(tmp_dir, "index.html")
            if os.path.isfile(written_html):
                with open(written_html, "r") as f:
                    assistant_text = f.read()
            else:
                assistant_text = payload.get("result", "")

        await on_event(StreamEvent(type="assistant_delta", text=assistant_text))
        return ProviderTurn(assistant_text=assistant_text, tool_calls=[])

    async def append_tool_results(
        self,
        turn: ProviderTurn,
        executed_tool_calls: list[ExecutedToolCall],
    ) -> None:
        # No tools are offered in this provider, so this is never called.
        return

    def total_cost_usd(self) -> Optional[float]:
        return None  # billed via the CLI's own login, not per-token

    async def close(self) -> None:
        return
