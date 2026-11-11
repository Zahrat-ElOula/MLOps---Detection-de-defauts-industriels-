"""Tests for GitHub feedback synchronisation (no real network calls)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.github_sync import GitHubSync
from src.api.main import app


def _write_pair(
    feedback_dir: Path, stem: str, label: str = "good"
) -> Path:
    image = feedback_dir / f"{stem}.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    image.with_suffix(".txt").write_text(label, encoding="utf-8")
    return image


def _make_sync(
    feedback_dir: Path, handler, **kwargs
) -> tuple[GitHubSync, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sync = GitHubSync(
        repo="owner/repo",
        token="secret-token",
        feedback_dir=feedback_dir,
        client=client,
        sleep=asyncio.sleep,
        backoff_seconds=0,
        **kwargs,
    )
    return sync, client


def _pushed_payload(feedback_dir: Path) -> dict:
    return json.loads((feedback_dir / ".pushed.json").read_text(encoding="utf-8"))


def test_network_failure_then_success_retries_and_marks_pushed(tmp_path: Path) -> None:
    """A transient PUT failure is retried, then the pair is marked pushed."""
    put_attempts: list[str] = []
    dispatches: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(404, request=request)
        if request.method == "PUT":
            put_attempts.append(request.url.path)
            if len(put_attempts) == 1:
                raise httpx.ConnectError("network down", request=request)
            return httpx.Response(201, json={"content": {}}, request=request)
        if request.method == "POST" and request.url.path.endswith("/dispatches"):
            dispatches.append(request)
            return httpx.Response(204, request=request)
        return httpx.Response(404, request=request)

    sync, client = _make_sync(tmp_path, handler)
    _write_pair(tmp_path, "sample")

    async def scenario() -> None:
        result = await sync.sync_pending()
        await client.aclose()
        assert result.pushed == ("sample.png",)
        assert result.auth_error is False

    asyncio.run(scenario())

    assert len(put_attempts) == 3  # one retry for the image, then the label
    pushed = _pushed_payload(tmp_path)["pushed"]
    assert "sample.png" in pushed
    assert pushed["sample.png"]["files"] == ["sample.png", "sample.txt"]
    assert len(dispatches) == 1
    assert json.loads(dispatches[0].content)["event_type"] == "feedback-received"


def test_auth_error_is_not_retried_and_health_reports_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """401 keeps the feedback pending, does not retry, and surfaces in /health."""
    put_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal put_calls
        if request.method == "GET":
            return httpx.Response(404, request=request)
        if request.method == "PUT":
            put_calls += 1
            return httpx.Response(401, json={"message": "Bad credentials"}, request=request)
        return httpx.Response(404, request=request)

    sync, client = _make_sync(tmp_path, handler)
    _write_pair(tmp_path, "unauthorised")

    async def scenario() -> None:
        result = await sync.sync_pending()
        await client.aclose()
        assert result.auth_error is True

    asyncio.run(scenario())

    assert put_calls == 1
    assert sync.pending_count() == 1
    assert not (tmp_path / ".pushed.json").exists()

    status = sync.status()
    assert status["status"] == "error"
    assert status["pending_feedback"] == 1
    assert "401" in status["last_error"]
    assert "secret-token" not in json.dumps(status)

    monkeypatch.setattr(app.state, "github_sync", sync, raising=False)
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["github_sync"]["status"] == "error"
    assert payload["github_sync"]["pending_feedback"] == 1
    assert "secret-token" not in response.text


def test_startup_resumes_only_pending_feedback(tmp_path: Path) -> None:
    """The startup loop uploads pending pairs but skips already pushed ones."""
    dispatched: dict[str, asyncio.Event] = {}
    put_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(404, request=request)
        if request.method == "PUT":
            put_paths.append(request.url.path)
            return httpx.Response(201, json={"content": {}}, request=request)
        if request.method == "POST" and request.url.path.endswith("/dispatches"):
            dispatched["event"].set()
            return httpx.Response(204, request=request)
        return httpx.Response(404, request=request)

    sync, client = _make_sync(tmp_path, handler, retry_seconds=3600)
    _write_pair(tmp_path, "already-done")
    _write_pair(tmp_path, "still-pending")

    (tmp_path / ".pushed.json").write_text(
        json.dumps(
            {
                "pushed": {
                    "already-done.png": {
                        "pushed_at": "2026-01-01T00:00:00+00:00",
                        "label": "good",
                        "files": ["already-done.png", "already-done.txt"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    async def scenario() -> None:
        dispatched["event"] = asyncio.Event()
        await sync.start()
        await asyncio.wait_for(dispatched["event"].wait(), timeout=2)
        await sync.stop()
        await client.aclose()

    asyncio.run(scenario())

    assert any(path.endswith("/still-pending.png") for path in put_paths)
    assert not any("already-done" in path for path in put_paths)
    pushed = _pushed_payload(tmp_path)["pushed"]
    assert "still-pending.png" in pushed
    assert "already-done.png" in pushed


def test_dispatch_is_sent_once_per_batch(tmp_path: Path) -> None:
    """Two pairs in one pass produce one repository dispatch, not two."""
    put_count = 0
    dispatch_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal put_count, dispatch_count
        if request.method == "GET":
            return httpx.Response(404, request=request)
        if request.method == "PUT":
            put_count += 1
            return httpx.Response(201, json={"content": {}}, request=request)
        if request.method == "POST" and request.url.path.endswith("/dispatches"):
            dispatch_count += 1
            return httpx.Response(204, request=request)
        return httpx.Response(404, request=request)

    sync, client = _make_sync(tmp_path, handler)
    _write_pair(tmp_path, "first")
    _write_pair(tmp_path, "second")

    async def scenario() -> None:
        await sync.sync_pending()
        await sync.sync_pending()  # nothing pending: no second dispatch
        await client.aclose()

    asyncio.run(scenario())

    assert put_count == 4
    assert dispatch_count == 1


@pytest.mark.parametrize(
    ("pushed", "expected_status", "expected_text"),
    [
        (False, "pending", "nouvelle tentative"),
        (True, "success", "envoyé à GitHub"),
    ],
)
def test_feedback_reports_whether_the_pair_was_pushed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pushed: bool,
    expected_status: str,
    expected_text: str,
) -> None:
    """The frontend can distinguish a successful push from a pending retry."""
    from src.api import main as api_main

    class FakeSync:
        enabled = True

        def __init__(self) -> None:
            self.synced = False

        async def sync_pending(self) -> None:
            self.synced = True

        def is_pushed(self, _image: Path) -> bool:
            return pushed

        def status(self) -> dict:
            return {
                "status": "ok",
                "pending_feedback": 0 if pushed else 1,
                "last_success": None,
                "last_error": None,
            }

    fake = FakeSync()
    monkeypatch.setattr(api_main, "FEEDBACK_DIR", tmp_path)
    monkeypatch.setattr(app.state, "github_sync", fake, raising=False)

    response = TestClient(app).post(
        "/feedback",
        files={"file": ("sample.png", b"\x89PNG\r\n", "image/png")},
        data={"label": "good"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == expected_status
    assert expected_text in response.json()["message"]
    assert fake.synced is True
    assert len(list(tmp_path.glob("*.png"))) == 1
    assert len(list(tmp_path.glob("*.txt"))) == 1


def test_health_stays_200_when_github_is_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GitHub outage is reported but never turns /health into an error."""
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    sync, client = _make_sync(tmp_path, handler)
    _write_pair(tmp_path, "offline")

    async def scenario() -> None:
        await sync.sync_pending()
        await client.aclose()

    asyncio.run(scenario())

    monkeypatch.setattr(app.state, "github_sync", sync, raising=False)
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["github_sync"]["status"] == "error"
    assert payload["github_sync"]["pending_feedback"] == 1
    assert payload["github_sync"]["last_error"]
    assert "secret-token" not in response.text
