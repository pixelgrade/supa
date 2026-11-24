from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any


class RecordingLease:
    def __init__(self, factory: "RecordingLeaseFactory", conversation_id: int, owner: str):
        self.factory = factory
        self.conversation_id = int(conversation_id)
        self.owner = owner
        self.active = False

    async def __aenter__(self):
        assert self.conversation_id not in self.factory.active_ids
        self.active = True
        self.factory.active_ids.add(self.conversation_id)
        self.factory.events.append(("enter", self.conversation_id, self.owner))
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.factory.events.append(("exit", self.conversation_id, self.owner))
        self.factory.active_ids.remove(self.conversation_id)
        self.active = False
        return False

    async def verify_owned(self) -> None:
        assert self.active
        self.factory.events.append(("verify", self.conversation_id, self.owner))

    async def run_guarded(
        self,
        callback: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        await self.verify_owned()
        name = getattr(callback, "_mock_name", None) or getattr(
            callback, "__name__", type(callback).__name__
        )
        self.factory.events.append(("mutation", self.conversation_id, name))
        return await callback(*args, **kwargs)


class RecordingLeaseFactory:
    def __init__(self):
        self.events: list[tuple[str, int, str]] = []
        self.active_ids: set[int] = set()

    def __call__(self, _db_path, conversation_id, *, owner: str, **_kwargs):
        return RecordingLease(self, int(conversation_id), owner)

    def assert_mutation_guarded(self, conversation_id: int, name: str) -> None:
        assert ("mutation", int(conversation_id), name) in self.events, self.events

    def assert_guarded_count(self, conversation_id: int, count: int) -> None:
        guarded = [
            event
            for event in self.events
            if event[0] == "mutation" and event[1] == int(conversation_id)
        ]
        assert len(guarded) == count, self.events
