from services import conversation_policy, extensions


def test_manual_route_preserves_a_registered_target_and_the_manual_tag(monkeypatch):
    def route(payload, *args, **kwargs):
        return ("manual", 72) if payload["id"] == 52 else None

    monkeypatch.setattr(extensions, "manual_route", route)
    assert conversation_policy.manual_route({"id": 52, "tags": []}) == ("manual", 72)
    assert conversation_policy.manual_route({"id": 42, "tags": [{"tag": "supa-ignore"}]}) == ("supa_ignore", None)
    assert conversation_policy.manual_route({"id": 42, "tags": []}) == (None, None)
