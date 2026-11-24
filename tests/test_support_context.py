import json
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest


def test_exact_duplicate_question_has_one_canonical_conversation():
    from services.support_context import duplicate_question_ids
    question = "Can I copy one index into a new journal while preserving the page numbers?"
    related = "=== SAME CUSTOMER: OTHER CURRENT CONVERSATIONS ===\n" + json.dumps([
        {"conversation_id": 20, "published_turns": [{"role": "customer", "message": question}]}])
    assert duplicate_question_ids(question, related, conversation_id=21) == [20]
    assert duplicate_question_ids(question, related, conversation_id=19) == []
    assert duplicate_question_ids("Different question about fonts", related, conversation_id=21) == []
    assert duplicate_question_ids("Thanks!", related, conversation_id=21) == []


def conversation(cid, customer=12, email='reader@woodland.example', tags=None, threads=None):
    return {'id':cid, 'number':cid+100, 'subject':'Plan selection', 'status':'active',
            'primaryCustomer':{'id':customer,'email':email}, 'tags':tags or [],
            '_embedded':{'threads': threads or []}}


def thread(tid, body, *, kind='customer', state='published', created='2024-03-01T10:00:00Z'):
    return {'id':tid,'type':kind,'state':state,'body':body,'createdAt':created}


@pytest.mark.asyncio
async def test_sibling_context_includes_current_customer_facts_but_never_drafts_or_notes():
    from services.support_context import related_customer_context
    full = conversation(2,threads=[thread(20,'The glossary appears twice in the index.'),
        thread(21,'Unapproved diagnosis',kind='message',state='draft'),
        thread(22,'Internal instruction',kind='note'),
        thread(23,'Published guidance',kind='message')])
    client=SimpleNamespace(search_conversations=AsyncMock(return_value=[conversation(1),full]),
                           get_complete_conversation=AsyncMock(return_value=full))
    result=await related_customer_context(helpscout=client,customer_id=12,customer_email='reader@woodland.example',exclude_id=1)
    assert 'The glossary appears twice in the index.' in result
    assert 'Published guidance' in result
    assert 'Unapproved diagnosis' not in result
    assert 'Internal instruction' not in result
    row = json.loads(result.splitlines()[-1])[0]
    assert row['last_customer_at'] == '2024-03-01T10:00:00Z'
    assert row['last_support_reply_at'] == '2024-03-01T10:00:00Z'
    assert client.get_complete_conversation.await_args.args == (2,)


@pytest.mark.asyncio
async def test_sibling_search_is_not_identity_proof_and_manual_cases_are_excluded():
    from services.support_context import related_customer_context
    wrong=conversation(2,customer=99)
    manual=conversation(3,tags=[{'tag':'supa-ignore'}])
    client=SimpleNamespace(search_conversations=AsyncMock(return_value=[wrong,manual]),
                           get_complete_conversation=AsyncMock(return_value=wrong))
    assert await related_customer_context(helpscout=client,customer_id=12,customer_email='reader@woodland.example',exclude_id=1) == ''


@pytest.mark.asyncio
async def test_related_lookup_failure_does_not_block_support():
    from services.support_context import related_customer_context
    client=SimpleNamespace(search_conversations=AsyncMock(side_effect=TimeoutError()),
                           get_complete_conversation=AsyncMock())
    assert await related_customer_context(helpscout=client,customer_id=12,customer_email='reader@woodland.example',exclude_id=1) == ''
    client.search_conversations.assert_awaited_once()


def test_current_notes_require_all_terms_in_the_selected_pack(tmp_path, monkeypatch):
    import re
    from services import support_context

    monkeypatch.setattr(support_context, "_CONTEXT_NOTES", [
        {"file": "index.md", "patterns": [re.compile("journal"), re.compile("index")]},
        {"file": "maps.md", "patterns": [re.compile("atlas")]},
    ])
    (tmp_path / "index.md").write_text("Index entries follow page order.")
    (tmp_path / "maps.md").write_text("Every map has a scale.")
    def context(subject):
        return support_context.current_product_context(subject=subject, message="", knowledge_dir=tmp_path)
    assert context("journal index") == "Index entries follow page order."
    assert context("journal") == ""
    assert context("atlas") == "Every map has a scale."
