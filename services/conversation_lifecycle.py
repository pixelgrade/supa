"""Narrow outbound creation and unsent-draft discard through the agent ledger."""

from __future__ import annotations

import asyncio
import hashlib
import re

from services.agent_auth import AgentIdentity
from services.conversation_gateway import (
    ConversationGateway, ConversationGatewayError, normalize_conversation,
    normalize_draft_body,
)
from services.conversation_locks import conversation_lease
from services.helpscout import HelpScoutError, _is_draft_thread


def refuse(code: str, status: int = 409) -> None:
    raise ConversationGatewayError("Conversation operation refused", code=code, status=status)


def require_scope(identity: AgentIdentity, scope: str) -> None:
    if scope not in identity.scopes:
        refuse("insufficient_scope", 403)


def require_id(value: int) -> None:
    if type(value) is not int or not 1 <= value <= (1 << 63) - 1:
        refuse("invalid_conversation", 400)


async def _execute(gateway, *, identity, scope, name, target, request, key, request_id,
                   callback):
    """Never resubmit after the single durable upstream-attempt marker."""
    args = dict(idempotency_key=key, identity=identity, scope=scope, request_id=request_id)
    reservation = await gateway._operation_reserve(
        **args, operation_name=name, target_id=target, request=request,
    )
    if reservation.replayed:
        if reservation.operation.state == "succeeded":
            response = dict(reservation.operation.response)
            response["conversation_id"] = int(response["conversation_id"])
            return response
        gateway._raise_replayed_operation(reservation.operation)
    await gateway._operation_begin(**args)
    state = {"submitted": False, "response": None}
    async def mark():
        await gateway._operation_mark_upstream(**args)
        state["submitted"] = True
    try:
        response = await callback(mark, state)
        await gateway._operation_finish(**args, outcome="succeeded", response=response)
        return response
    except BaseException as error:
        # Cancellation, transport errors and readback failures after submission
        # are all unknown. Never retry them with a new key.
        code = getattr(error, "code", "outcome_unknown" if state["submitted"] else "upstream_read_failed")
        unknown = state["submitted"]
        # Only a definitive upstream rejection can safely terminalize as failed.
        if code == "upstream_rejected":
            unknown = False
        await gateway._operation_finish(
            **args, outcome="outcome_unknown" if unknown else "failed",
            response=state["response"], error={"code": "outcome_unknown" if unknown else code,
                                              "retryable": False},
        )
        if isinstance(error, asyncio.CancelledError):
            raise
        if unknown:
            refuse("outcome_unknown")
        raise


def _check_mutation(result, expected_status):
    if result.ok:
        if result.status_code != expected_status:
            refuse("outcome_unknown")
        return
    if result.outcome_unknown or result.status_code is None or result.status_code >= 500:
        refuse("outcome_unknown")
    refuse("upstream_rejected", 502)


async def create_conversation(
    gateway: ConversationGateway, *, mailbox_id: int, customer_id: int, subject: str,
    body: str, body_format: str, draft: bool, idempotency_key: str,
    identity: AgentIdentity, request_id: str,
) -> dict:
    require_scope(identity, "create")
    require_id(mailbox_id)
    require_id(customer_id)
    if type(draft) is not bool:
        refuse("invalid_body", 400)
    # Named existing conversations cannot authorize a not-yet-existing resource.
    if identity.is_conversation_restricted:
        refuse("conversation_not_allowed", 403)
    if not draft:
        require_scope(identity, "send")
    normalized = normalize_draft_body(body, body_format)
    if not subject.strip() or len(subject) > 255 or any(ord(c) < 32 for c in subject):
        refuse("invalid_subject", 400)
    request = dict(mailbox_id=mailbox_id, customer_id=customer_id, subject=subject,
                   body=body, body_format=body_format, draft=draft)

    async def submit(mark, state):
        # A mailbox/customer lease serializes this gateway's outbound creations;
        # the ledger key remains the actual duplicate-prevention boundary.
        async with conversation_lease(gateway.db_path, f"create:{mailbox_id}:{customer_id}",
                                      owner=hashlib.sha256(idempotency_key.encode()).hexdigest()) as lease:
            await lease.verify_owned()
            async def mutate():
                await mark()
                return await gateway.client.create_conversation_result(
                    mailbox_id=mailbox_id, customer_id=customer_id, subject=subject,
                    text=normalized, draft=draft,
                )
            result = await lease.run_guarded(mutate)
            _check_mutation(result, 201)
            if type(result.resource_id) is not int or result.resource_id <= 0:
                refuse("outcome_unknown")
            cid = result.resource_id
            response = dict(action="created", conversation_id=cid,
                            status="created" if draft else "sent",
                            body_sha256=hashlib.sha256(normalized.encode()).hexdigest())
            state["response"] = response
            raw = await gateway.client.get_complete_conversation(cid)
            threads = [t for t in gateway._raw_threads(raw) if t.get("type") != "lineitem"]
            if (gateway._resolved_id(raw) != cid or raw.get("subject") != subject
                or normalize_conversation(raw)["mailbox"]["id"] != mailbox_id
                or (normalize_conversation(raw).get("customer") or {}).get("id") != customer_id
                or not threads or len(threads) != 1
                or threads[0].get("type") not in {"reply", "message"}
                or threads[0].get("createdBy", {}).get("type") != "user"
                or threads[0].get("state") != ("draft" if draft else "published")
                or _is_draft_thread(threads[0]) != draft
                or ("draft" in threads[0] and threads[0]["draft"] is not draft)
                or normalize_draft_body(threads[0].get("body", ""), "html") != normalized):
                refuse("verification_mismatch")
            return response

    return await _execute(gateway, identity=identity, scope="create", name="conversation.create",
                          target=f"mailbox:{mailbox_id}:customer:{customer_id}", request=request,
                          key=idempotency_key, request_id=request_id, callback=submit)


async def discard_conversation(
    gateway: ConversationGateway, conversation_id: int, *, expected_version: str,
    idempotency_key: str, identity: AgentIdentity, request_id: str,
) -> dict:
    require_scope(identity, "delete")
    require_id(conversation_id)
    gateway._require_requested_allowed(identity, conversation_id)
    if re.fullmatch(r"[0-9a-f]{64}", expected_version) is None:
        refuse("invalid_draft_version", 400)

    async def submit(mark, state):
        async with conversation_lease(gateway.db_path, conversation_id,
                                      owner=hashlib.sha256(idempotency_key.encode()).hexdigest()) as lease:
            raw = await gateway.client.get_complete_conversation(conversation_id)
            # NEVER follow a merge/redirect for destructive operations.
            if gateway._resolved_id(raw) != conversation_id:
                refuse("conversation_resolution_changed")
            # Activity entries (lineitems) are not content: Help Scout adds one to an
            # API-created draft conversation, exactly as creation's readback allows.
            threads = [t for t in gateway._raw_threads(raw) if t.get("type") != "lineitem"]
            if normalize_conversation(raw)["version"] != expected_version:
                refuse("conversation_changed")
            # Empty *thread collection* is allowed only for an upstream draft.
            # Every existing thread must be an unsent agent reply, never a note
            # or a customer message even if marked draft by the upstream API.
            if ((not threads and raw.get("status") != "draft") or any(
                not _is_draft_thread(t) or t.get("type") not in {"message", "reply"}
                or t.get("state") != "draft"
                or ("draft" in t and t["draft"] is not True)
                or t.get("createdBy", {}).get("type") != "user" for t in threads
            )):
                refuse("conversation_not_discardable")
            await lease.verify_owned()
            async def mutate():
                await mark()
                return await gateway.client.delete_conversation_result(conversation_id)
            result = await lease.run_guarded(mutate)
            _check_mutation(result, 204)
            # 204 alone is not readback proof; 404 without the successful DELETE
            # would equally be a nonexistent/merged resource and is never success.
            try:
                await gateway.client.get_complete_conversation(conversation_id)
            except HelpScoutError as error:
                if (error.status != 404 or error.details.get("endpoint") !=
                        f"/conversations/{conversation_id}?embed=threads"):
                    raise
            else:
                refuse("verification_mismatch")
            return dict(action="discarded", conversation_id=conversation_id, status="discarded")

    return await _execute(gateway, identity=identity, scope="delete", name="conversation.discard",
                          target=conversation_id, request={"expected_version": expected_version},
                          key=idempotency_key, request_id=request_id, callback=submit)
