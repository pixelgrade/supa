from scripts.list_open_conversations import summarize_conversations


def _convo(threads, **over):
    base = {
        "id": 5001,
        "subject": "Help with Field Journal",
        "primaryCustomer": {"id": 42},
        "tags": [{"tag": "field-journal"}],
        "_embedded": {"threads": threads},
    }
    base.update(over)
    return base


def test_latest_customer_thread_marks_waiting():
    convo = _convo([
        {"type": "customer", "createdAt": "2026-06-01T10:00:00Z"},
        {"type": "message", "createdAt": "2026-06-02T10:00:00Z"},
        {"type": "customer", "createdAt": "2026-06-03T10:00:00Z"},
    ])
    [out] = summarize_conversations([convo])
    assert out["id"] == 5001
    assert out["latest_thread_type"] == "customer"
    assert out["customer_id"] == 42
    assert out["existing_tags"] == ["field-journal"]


def test_latest_agent_reply_marks_answered():
    convo = _convo([
        {"type": "customer", "createdAt": "2026-06-01T10:00:00Z"},
        {"type": "message", "createdAt": "2026-06-04T10:00:00Z"},
    ])
    [out] = summarize_conversations([convo])
    assert out["latest_thread_type"] == "agent"


def test_notes_and_lineitems_are_ignored_when_choosing_latest():
    convo = _convo([
        {"type": "customer", "createdAt": "2026-06-01T10:00:00Z"},
        {"type": "note", "createdAt": "2026-06-09T10:00:00Z"},
        {"type": "lineitem", "createdAt": "2026-06-10T10:00:00Z"},
    ])
    [out] = summarize_conversations([convo])
    assert out["latest_thread_type"] == "customer"


def test_no_relevant_threads_returns_unknown():
    convo = _convo([
        {"type": "note", "createdAt": "2026-06-09T10:00:00Z"},
        {"type": "phone", "createdAt": "2026-06-10T10:00:00Z"},
    ])
    [out] = summarize_conversations([convo])
    assert out["latest_thread_type"] == "unknown"


def test_tie_on_timestamp_resolves_to_agent():
    convo = _convo([
        {"type": "customer", "createdAt": "2026-06-03T10:00:00Z"},
        {"type": "message", "createdAt": "2026-06-03T10:00:00Z"},
    ])
    [out] = summarize_conversations([convo])
    assert out["latest_thread_type"] == "agent"
