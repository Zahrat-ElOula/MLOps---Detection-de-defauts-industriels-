"""GitHub synchronisation for user feedback and CT dispatch.

The API stores feedback locally first. This module uploads each image/label
pair with the GitHub Contents API, marks the pair as pushed only when both
files succeeded, and sends one ``feedback-received`` repository dispatch per
successful batch. A background task resumes pending feedback at startup and
then retries periodically. The daily CT schedule is the final safety net.

The GitHub token is never logged or exposed through status endpoints.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote

import httpx

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_API_URL = "https://api.github.com"
DEFAULT_RETRY_SECONDS = 300
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 1.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SyncResult:
    """Outcome of one synchronisation pass."""

    pushed: tuple[str, ...] = ()
    error: str | None = None
    auth_error: bool = False


class GitHubSync:
    """Upload feedback pairs to GitHub and trigger CT once per batch."""

    def __init__(
        self,
        *,
        repo: str,
        token: str,
        feedback_dir: Path | None = None,
        enabled: bool | None = None,
        branch: str = "main",
        api_url: str = DEFAULT_API_URL,
        repo_path: str = "data/feedback",
        retry_seconds: float = DEFAULT_RETRY_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.repo = repo.strip()
        self.token = token.strip()
        self.feedback_dir = Path(feedback_dir or (PROJECT_ROOT / "data" / "feedback"))
        self.branch = branch.strip() or "main"
        self.api_url = api_url.rstrip("/")
        self.repo_path = repo_path.strip("/")
        self.retry_seconds = max(1.0, float(retry_seconds))
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_seconds = max(0.0, float(backoff_seconds))

        repo_parts = self.repo.split("/")
        repo_is_valid = len(repo_parts) == 2 and all(part.strip() for part in repo_parts)
        self.enabled = (
            repo_is_valid and bool(self.token) if enabled is None else bool(enabled)
        )

        self._client = client
        self._owns_client = client is None
        self._sleep = sleep or asyncio.sleep
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._last_success_at: datetime | None = None
        self._last_error_at: datetime | None = None
        self._last_error: str | None = None

    @classmethod
    def from_env(cls, feedback_dir: Path | None = None) -> "GitHubSync":
        """Build a synchroniser from environment variables.

        Missing ``GITHUB_TOKEN`` or ``GITHUB_REPO`` disables synchronisation
        without preventing the API from starting.
        """
        retry_seconds = _positive_float(
            os.getenv("FEEDBACK_RETRY_SECONDS"), DEFAULT_RETRY_SECONDS
        )
        timeout_seconds = _positive_float(
            os.getenv("GITHUB_TIMEOUT_SECONDS"), DEFAULT_TIMEOUT_SECONDS
        )
        sync = cls(
            repo=os.getenv("GITHUB_REPO", ""),
            token=os.getenv("GITHUB_TOKEN", ""),
            feedback_dir=feedback_dir,
            branch=os.getenv("GITHUB_BRANCH", "main"),
            api_url=os.getenv("GITHUB_API_URL", DEFAULT_API_URL),
            retry_seconds=retry_seconds,
            timeout_seconds=timeout_seconds,
        )
        if not sync.enabled:
            logger.info(
                "GitHub feedback sync disabled: set GITHUB_TOKEN and GITHUB_REPO to enable it"
            )
        return sync

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Start the background loop (first pass runs immediately)."""
        if not self.enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._run_loop(), name="github-feedback-sync")

    async def stop(self) -> None:
        """Cancel the background loop, if any."""
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def aclose(self) -> None:
        """Close the HTTP client when this instance created it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _run_loop(self) -> None:
        while True:
            try:
                await self.sync_pending()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # keep the background task alive
                self._record_error(
                    f"feedback sync loop failure: {type(exc).__name__}: {exc}"
                )
            await self._sleep(self.retry_seconds)

    # ------------------------------------------------------------------
    # Pending feedback state
    # ------------------------------------------------------------------
    @property
    def pushed_path(self) -> Path:
        return self.feedback_dir / ".pushed.json"

    def _load_pushed(self) -> dict[str, Any]:
        path = self.pushed_path
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Cannot read %s: %s", path, exc)
            return {}
        if not isinstance(data, dict):
            return {}
        pushed = data.get("pushed", data)
        return pushed if isinstance(pushed, dict) else {}

    def _save_pushed(self, pushed: dict[str, Any]) -> None:
        path = self.pushed_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"pushed": pushed}, indent=2, sort_keys=True) + "\n"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)

    def _feedback_images(self) -> list[Path]:
        if not self.feedback_dir.is_dir():
            return []
        return sorted(
            path
            for path in self.feedback_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and not path.name.startswith(".")
        )

    def pending_images(self) -> list[Path]:
        """Return images whose image/label pair has not been marked pushed."""
        pushed = self._load_pushed()
        return [path for path in self._feedback_images() if path.name not in pushed]

    def pending_count(self) -> int:
        return len(self.pending_images())

    def is_pushed(self, image: Path | str) -> bool:
        return Path(image).name in self._load_pushed()

    @staticmethod
    def _label_path(image: Path) -> Path:
        return image.with_suffix(".txt")

    def _mark_pushed(self, image: Path, label: str) -> None:
        pushed = self._load_pushed()
        pushed[image.name] = {
            "pushed_at": _utc_now().isoformat(),
            "label": label,
            "files": [image.name, self._label_path(image).name],
        }
        self._save_pushed(pushed)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "industrial-defect-detection-api",
        }

    def _safe(self, message: str) -> str:
        if self.token:
            return message.replace(self.token, "***")
        return message

    def _record_success(self) -> None:
        self._last_success_at = _utc_now()
        logger.info("GitHub feedback sync succeeded")

    def _record_error(self, message: str) -> None:
        message = self._safe(message)
        self._last_error = message
        self._last_error_at = _utc_now()
        logger.warning("GitHub feedback sync error: %s", message)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds)
            )
        return self._client

    def _contents_path(self, repo_file: str) -> str:
        return f"/repos/{self.repo}/contents/{quote(repo_file, safe='/')}"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        accept_status: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> tuple[httpx.Response | None, str | None, bool]:
        """Perform one GitHub request.

        Returns ``(response, error, auth_error)``. Network errors, timeouts,
        429 and 5xx responses are retried with exponential backoff. A 401 or
        403 is never retried.
        """
        url = f"{self.api_url}{path}"
        last_error: str | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self._get_client().request(
                    method, url, headers=self._headers(), **kwargs
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = self._safe(
                    f"{operation}: {type(exc).__name__}: {exc}"
                )
                if attempt < self.max_attempts:
                    await self._sleep(self.backoff_seconds * (2 ** (attempt - 1)))
                    continue
                self._record_error(last_error)
                return None, last_error, False

            if response.status_code in (401, 403):
                error = (
                    f"{operation}: GitHub authentication/permission error "
                    f"({response.status_code})"
                )
                self._record_error(error)
                return response, error, True

            if response.status_code in accept_status:
                return response, None, False

            if response.status_code == 429 or response.status_code >= 500:
                last_error = (
                    f"{operation}: GitHub API transient error "
                    f"({response.status_code})"
                )
                if attempt < self.max_attempts:
                    await self._sleep(self.backoff_seconds * (2 ** (attempt - 1)))
                    continue
                self._record_error(last_error)
                return response, last_error, False

            if response.status_code >= 400:
                error = f"{operation}: GitHub API error ({response.status_code})"
                self._record_error(error)
                return response, error, False

            return response, None, False

        error = last_error or f"{operation}: GitHub request failed"
        self._record_error(error)
        return None, error, False

    async def _get_sha(self, repo_file: str) -> tuple[str | None, str | None, bool]:
        response, error, auth = await self._request(
            "GET",
            self._contents_path(repo_file),
            operation=f"look up {repo_file}",
            accept_status=(404,),
        )
        if auth or error:
            return None, error, auth
        if response is None or response.status_code == 404:
            return None, None, False
        try:
            data = response.json()
            sha = data.get("sha")
        except (ValueError, AttributeError):
            error = f"look up {repo_file}: invalid GitHub response"
            self._record_error(error)
            return None, error, False
        return sha if isinstance(sha, str) else None, None, False

    async def _put_file(self, path: Path) -> tuple[bool, str | None, bool]:
        repo_file = f"{self.repo_path}/{path.name}"
        content = base64.b64encode(path.read_bytes()).decode("ascii")
        sha, error, auth = await self._get_sha(repo_file)
        if auth or error:
            return False, error, auth

        body: dict[str, Any] = {
            "message": f"feedback: add {path.name}",
            "content": content,
            "branch": self.branch,
        }
        if sha:
            body["sha"] = sha

        response, error, auth = await self._request(
            "PUT",
            self._contents_path(repo_file),
            operation=f"upload {path.name}",
            json=body,
        )
        if auth or error:
            # A concurrent upload may have created the file after our lookup.
            if response is not None and response.status_code in (409, 422) and sha is None:
                new_sha, refresh_error, refresh_auth = await self._get_sha(repo_file)
                if not refresh_auth and not refresh_error and new_sha:
                    body["sha"] = new_sha
                    response, error, auth = await self._request(
                        "PUT",
                        self._contents_path(repo_file),
                        operation=f"upload {path.name}",
                        json=body,
                    )
                    if (
                        not auth
                        and not error
                        and response is not None
                        and response.status_code < 400
                    ):
                        return True, None, False
            return False, error, auth

        if response is not None and response.status_code in (200, 201):
            return True, None, False

        status = response.status_code if response is not None else "none"
        error = f"upload {path.name}: unexpected GitHub status {status}"
        self._record_error(error)
        return False, error, False

    async def _dispatch(self, names: list[str]) -> str | None:
        body = {
            "event_type": "feedback-received",
            "client_payload": {
                "files": [f"{self.repo_path}/{name}" for name in names],
            },
        }
        response, error, auth = await self._request(
            "POST",
            f"/repos/{self.repo}/dispatches",
            operation="dispatch CT",
            json=body,
            accept_status=(204,),
        )
        if auth or error:
            return error or "dispatch CT: GitHub authentication error"
        if response is None or response.status_code not in (200, 204):
            status = response.status_code if response is not None else "none"
            error = f"dispatch CT: unexpected GitHub status {status}"
            self._record_error(error)
            return error
        return None

    # ------------------------------------------------------------------
    # Synchronisation
    # ------------------------------------------------------------------
    async def sync_pending(self) -> SyncResult:
        """Push every complete pending pair, then dispatch CT once."""
        if not self.enabled:
            return SyncResult()
        async with self._lock:
            return await self._sync_pending_locked()

    async def _sync_pending_locked(self) -> SyncResult:
        pushed_names: list[str] = []
        auth_error = False
        last_error: str | None = None

        for image in self.pending_images():
            label_path = self._label_path(image)
            if not label_path.is_file():
                message = f"{image.name}: missing label file {label_path.name}"
                logger.warning("%s; keeping feedback pending", message)
                last_error = message
                continue

            try:
                ok, error, auth = await self._put_file(image)
            except OSError as exc:
                message = f"{image.name}: cannot read image: {exc}"
                self._record_error(message)
                last_error = message
                continue
            if auth:
                auth_error = True
                last_error = error
                break
            if not ok:
                last_error = error
                continue

            try:
                ok, error, auth = await self._put_file(label_path)
            except OSError as exc:
                message = f"{label_path.name}: cannot read label: {exc}"
                self._record_error(message)
                last_error = message
                continue
            if auth:
                auth_error = True
                last_error = error
                break
            if not ok:
                last_error = error
                continue

            try:
                label = label_path.read_text(encoding="utf-8").strip()
            except OSError:
                label = ""
            self._mark_pushed(image, label)
            pushed_names.append(image.name)

        if pushed_names:
            self._record_success()
            dispatch_error = await self._dispatch(pushed_names)
            if dispatch_error:
                last_error = dispatch_error

        return SyncResult(tuple(pushed_names), last_error, auth_error)

    def status(self) -> dict[str, Any]:
        """Return the public health payload (never contains the token)."""
        pending_error: str | None = None
        try:
            pending = self.pending_count()
        except OSError as exc:
            pending = 0
            pending_error = f"cannot scan feedback directory: {exc}"

        if not self.enabled:
            state = "disabled"
        elif pending_error:
            state = "error"
        elif self._last_error_at is not None and (
            self._last_success_at is None or self._last_error_at >= self._last_success_at
        ):
            state = "error"
        else:
            state = "ok"

        error = pending_error or self._last_error
        return {
            "status": state,
            "pending_feedback": pending,
            "last_success": (
                self._last_success_at.isoformat() if self._last_success_at else None
            ),
            "last_error": self._safe(error) if error else None,
        }


def _positive_float(raw: str | None, default: float) -> float:
    try:
        value = float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


__all__ = ["GitHubSync", "SyncResult", "IMAGE_EXTENSIONS"]
