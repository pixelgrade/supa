"""Fail-closed admission for bounded, draft-only support recovery."""
import json
from datetime import datetime, timezone

from services.conversation_gateway import normalize_conversation, note_version
from services.helpscout import get_supa_owned_draft_thread_ids, get_draft_thread_ids, strip_html

EVIDENCE_BLOCKERS = {'missing_critical_evidence', 'missing_repo_evidence', 'unread_attachments', 'unreviewed_screenshots'}
TECHNICAL_BLOCKERS = EVIDENCE_BLOCKERS | {'manual_review', 'malformed_provider_output', 'semantic_failure_match', 'contradiction_detected', 'repo_evidence_contradiction', 'dead_css_selector'}


def _list(value):
    try:
        value = json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError):
        return []
    return value if isinstance(value, list) else []


def human_note_versions(threads):
    return {str(t.get('id')): note_version(t) for t in threads if t.get('type') == 'note'
            and (t.get('source') or {}).get('type') != 'api'}


def _time(value):
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except (ValueError, AttributeError):
        return None


def substantive_draft_ids(threads):
    ids = set(get_draft_thread_ids(threads))
    return [t['id'] for t in threads if t.get('id') in ids
            and strip_html(t.get('body') or '') != 'Manual review required before sending a reply.']


def recovery_metadata(raw, row):
    result = {'eligible': False, 'reason': 'human_review'}
    if not row or row.get('delivery_mode') != 'draft' or not row.get('response_posted'):
        return result
    current = normalize_conversation(raw)
    draft = current.get('draft')
    threads = raw.get('_embedded', {}).get('threads', [])
    if (raw.get('status') != 'active' or current['waiting_on'] != 'agent' or not draft
            or len(substantive_draft_ids(threads)) != 1
            or draft['thread_id'] not in get_supa_owned_draft_thread_ids(threads)
            or draft['body_html'] != row.get('draft_response_html')):
        return result
    for thread in threads:
        if (thread.get('type') == 'note' and (thread.get('source') or {}).get('type') != 'api'
                and (not _time(thread.get('createdAt')) or not _time(draft.get('created_at'))
                     or _time(thread['createdAt']) >= _time(draft['created_at']))):
            return result
    flags = set(_list(row.get('risk_flags')))
    trace = _list(row.get('confidence_trace'))
    blockers = set()
    # All observed hard gates remain protective; only explicitly technical holds may recover.
    for step in trace:
        if isinstance(step, dict):
            blockers.update(step.get('hard_blockers') or [])
    if blockers - TECHNICAL_BLOCKERS:
        return result
    if 'evaluator_failure' in flags:
        reason = 'evaluator_failure'
    elif blockers & EVIDENCE_BLOCKERS:
        reason = 'missing_evidence'
    else:
        return result
    return {'eligible': True, 'reason': reason,
            'expected_thread_id': str(current['latest_actionable_thread_id']),
            'expected_draft_version': draft['version'],
            'expected_conversation_version': current['version']}


def guard_matches(raw, row, expected):
    current = recovery_metadata(raw, row)
    return current['eligible'] and all(current.get(key) == expected.get(key) for key in (
        'expected_thread_id', 'expected_draft_version', 'expected_conversation_version'))


def verified_replacement(raw, expected, body):
    current = normalize_conversation(raw)
    draft = current.get('draft')
    threads = raw.get('_embedded', {}).get('threads', [])
    human_notes = human_note_versions(threads)
    if (not draft or raw.get('status') != 'active' or current['waiting_on'] != 'agent'
            or len(substantive_draft_ids(threads)) != 1
            or draft['thread_id'] not in get_supa_owned_draft_thread_ids(threads)
            or human_notes != expected.get('baseline_human_notes', {})
            or str(current['latest_actionable_thread_id']) != expected['expected_thread_id']
            or draft['body_html'] != body or draft['version'] == expected['expected_draft_version']):
        return None
    return {'draft_thread_id': draft['thread_id'], 'draft_version': draft['version'],
            'customer_thread_id': current['latest_actionable_thread_id']}
