import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional


class CancellationToken:
    def __init__(self) -> None:
        self._cancelled = False
        self._reason: Optional[str] = None
        self._listeners: list[Callable[[str], None]] = []

    def cancel(self, reason: str = "cancelled") -> None:
        if not self._cancelled:
            self._cancelled = True
            self._reason = reason
            for listener in self._listeners:
                try:
                    listener(reason)
                except Exception:
                    pass

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def add_listener(self, listener: Callable[[str], None]) -> None:
        self._listeners.append(listener)

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise asyncio.CancelledError(self._reason or "cancelled")


@dataclass(frozen=True)
class PolicyResult:
    allowed: bool
    reason: Optional[str] = None


@dataclass
class RetryPolicy:
    max_retries: int = 3
    delay_seconds: float = 1.0
    backoff_factor: float = 2.0

    def delays(self) -> Iterator[float]:
        for attempt in range(self.max_retries + 1):
            if attempt == 0:
                yield 0.0
            else:
                yield self.delay_seconds * (self.backoff_factor ** (attempt - 1))


@dataclass
class RuntimePolicy:
    max_iterations: int = 10
    timeout_seconds: float = 300.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    def allow_continue(
        self, iteration: int, elapsed_time: float, token: CancellationToken
    ) -> PolicyResult:
        if token.cancelled:
            return PolicyResult(allowed=False, reason=token.reason or "cancelled")
        if iteration >= self.max_iterations:
            return PolicyResult(
                allowed=False, reason=f"达到最大迭代次数 {self.max_iterations}"
            )
        if elapsed_time >= self.timeout_seconds:
            return PolicyResult(
                allowed=False, reason=f"执行超时 {self.timeout_seconds}s"
            )
        return PolicyResult(allowed=True)
