import base64
import json
from typing import Any

import pytest

from agent.providers.cli import CliProviderSession
from llm import Llm


class _FakeProcess:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


@pytest.mark.asyncio
async def test_stream_turn_runs_cli_and_returns_result(monkeypatch: Any) -> None:
    captured_args: list[str] = []

    async def fake_create_subprocess_exec(
        *args: str, **kwargs: Any
    ) -> _FakeProcess:
        captured_args.extend(args)
        return _FakeProcess(json.dumps({"result": "<div>hi</div>"}).encode())

    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", fake_create_subprocess_exec
    )

    png_bytes = b"\x89PNG\r\n\x1a\nfake"
    data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Build this page"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    session = CliProviderSession(
        model=Llm.CLAUDE_CODE_CLI, prompt_messages=prompt_messages  # type: ignore[arg-type]
    )

    events: list[str] = []

    async def on_event(event: Any) -> None:
        events.append(event.text)

    turn = await session.stream_turn(on_event)

    assert turn.assistant_text == "<div>hi</div>"
    assert turn.tool_calls == []
    assert events == ["<div>hi</div>"]
    assert captured_args[0] == "claude"
    assert "-p" in captured_args
    assert "Build this page" in captured_args[captured_args.index("-p") + 1]
    assert "--allowedTools" in captured_args
    assert session.total_cost_usd() is None


@pytest.mark.asyncio
async def test_stream_turn_reads_file_written_by_cli(monkeypatch: Any) -> None:
    """When the CLI writes index.html (its normal Write-tool behavior),
    that file wins over the chat-text "result" field."""
    import os

    async def fake_create_subprocess_exec(
        *args: str, **kwargs: Any
    ) -> _FakeProcess:
        cwd = kwargs["cwd"]
        with open(os.path.join(cwd, "index.html"), "w") as f:
            f.write("<html><body>from file</body></html>")
        return _FakeProcess(json.dumps({"result": "some chat text"}).encode())

    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", fake_create_subprocess_exec
    )

    session = CliProviderSession(model=Llm.CLAUDE_CODE_CLI, prompt_messages=[])

    async def on_event(event: Any) -> None:
        pass

    turn = await session.stream_turn(on_event)
    assert turn.assistant_text == "<html><body>from file</body></html>"


@pytest.mark.asyncio
async def test_stream_turn_raises_on_nonzero_exit(monkeypatch: Any) -> None:
    async def fake_create_subprocess_exec(*args: str, **kwargs: Any) -> _FakeProcess:
        return _FakeProcess(b"", b"boom", returncode=1)

    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", fake_create_subprocess_exec
    )

    session = CliProviderSession(model=Llm.CLAUDE_CODE_CLI, prompt_messages=[])

    async def on_event(event: Any) -> None:
        pass

    with pytest.raises(RuntimeError, match="boom"):
        await session.stream_turn(on_event)
