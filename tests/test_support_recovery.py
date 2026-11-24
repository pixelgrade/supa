import pytest
from services import support_recovery as recovery
from services.conversation_gateway import normalize_conversation


def fixture():
    raw = {'id': 123, 'status': 'active', '_embedded': {'threads': [
        {'id': 1, 'type': 'customer', 'state': 'published', 'createdBy': {'type': 'customer'}, 'createdAt': '2026-10-20T00:00:00Z', 'body': 'Help'},
        {'id': 2, 'type': 'message', 'state': 'draft', 'source': {'type': 'api'}, 'createdAt': '2026-10-20T01:00:00Z', 'body': '<p>Draft</p>'},
    ]}}
    row = {'delivery_mode': 'draft', 'response_posted': 1, 'draft_response_html': '<p>Draft</p>', 'risk_flags': '["evaluator_failure"]', 'confidence_trace': '[]'}
    return raw, row


def test_machine_timeout_is_recoverable_with_pinned_versions():
    raw, row = fixture()
    result = recovery.recovery_metadata(raw, row)
    assert result['eligible'] is True
    assert result['reason'] == 'evaluator_failure'
    assert result['expected_thread_id'] == '1'
    assert result['expected_draft_version'] == normalize_conversation(raw)['draft']['version']


@pytest.mark.parametrize('change', ['human_source', 'edited_body', 'closed', 'answered', 'multiple_drafts', 'money_gate'])
def test_recovery_preserves_real_holds(change):
    raw, row = fixture()
    threads = raw['_embedded']['threads']
    if change == 'human_source': threads[-1]['source']['type'] = 'web'
    if change == 'edited_body': threads[-1]['body'] = 'Human edit'
    if change == 'closed': raw['status'] = 'closed'
    if change == 'answered': threads.append({'id': 3, 'type': 'message', 'state': 'published', 'createdBy': {'type': 'user'}, 'createdAt': '2026-10-20T02:00:00Z', 'body': 'Sent'})
    if change == 'multiple_drafts': threads.append(dict(threads[-1], id=4))
    if change == 'money_gate': row['confidence_trace'] = '[{"hard_blockers":["financial_commitment"]}]'
    assert recovery.recovery_metadata(raw, row)['eligible'] is False


def test_missing_evidence_can_recover_but_generic_manual_review_cannot():
    raw, row = fixture()
    row['risk_flags'] = '[]'
    row['confidence_trace'] = '[{"hard_blockers":["manual_review","missing_critical_evidence"]}]'
    assert recovery.recovery_metadata(raw, row)['reason'] == 'missing_evidence'
    row['confidence_trace'] = '[{"hard_blockers":["manual_review"]}]'
    assert not recovery.recovery_metadata(raw, row)['eligible']


def test_guard_rejects_note_added_during_investigation():
    raw, row = fixture()
    expected = recovery.recovery_metadata(raw, row)
    assert recovery.guard_matches(raw, row, expected)
    raw['_embedded']['threads'].append({'id': 3, 'type': 'note', 'body': 'Hold this', 'state': 'published'})
    assert not recovery.guard_matches(raw, row, expected)


def test_recover_task_requires_all_fingerprints_and_never_send():
    from services.agent_tasks import prepare_task_request, AgentTaskRequestError
    args = {'conversation_id': 123, 'expected_thread_id': '1', 'expected_draft_version': 'a'*64, 'expected_conversation_version': 'b'*64}
    task = prepare_task_request('ticket.recover', args, scopes={'task','read','draft'})
    assert task.delivery_ceiling == 'draft'
    with pytest.raises(AgentTaskRequestError):
        prepare_task_request('ticket.recover', dict(args, allow_send=True), scopes={'task','read','draft','send'})
    with pytest.raises(AgentTaskRequestError):
        prepare_task_request('ticket.recover', {'conversation_id':123}, scopes={'task','read','draft'})


@pytest.mark.asyncio
async def test_recovery_is_bounded_and_passes_guard(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from db.database import init_db, log_ticket
    from services.agent_tasks import AgentTaskRepository, run_process_task
    path = str(tmp_path/'audit.db'); init_db(path)
    raw, row = fixture()
    log_ticket(path, helpscout_id='123', **row)
    expected = recovery.recovery_metadata(raw, row)
    args = {'conversation_id':123, **{k:v for k,v in expected.items() if k.startswith('expected_')}}
    processor = SimpleNamespace(helpscout=SimpleNamespace(get_complete_conversation=AsyncMock(return_value=raw)), process_ticket=AsyncMock(return_value={'status':'processed','delivery_mode':'draft'}))
    monkeypatch.setattr('services.agent_tasks.manual_route', lambda _: (None, None))
    repository = AgentTaskRepository(path)
    for index in range(3):
        run = repository.create_task_run(token_id='a',token_name='A',task_name='ticket.recover',arguments=args,scopes={'read','draft','task'},idempotency_key=f'recover-{index}')
        await run_process_task(repository, run.run_id, processor=processor)
    assert processor.process_ticket.await_count == 2, repository.get_run(run.run_id,token_id="a")
    assert processor.process_ticket.call_args.kwargs['delivery_ceiling'] == 'draft'
    assert processor.process_ticket.call_args.kwargs['recovery_guard']['expected_thread_id'] == '1'
    assert repository.get_run(run.run_id,token_id='a').result['reason'] == 'recovery_exhausted'


def test_timeout_with_content_contradiction_is_researchable_without_send():
    raw, row = fixture()
    row['confidence_trace'] = '[{"hard_blockers":["malformed_provider_output","semantic_failure_match"]}]'
    assert recovery.recovery_metadata(raw,row)['eligible']


def test_cli_recover_requires_fingerprints():
    from supa_cli.main import _parser
    args = _parser().parse_args(['ticket','recover','123','--expected-thread-id','1','--expected-draft-version','a'*64,'--expected-conversation-version','b'*64])
    assert args.ticket_command == 'recover'


def test_readback_requires_new_matching_draft_and_same_customer_obligation():
    raw, row = fixture()
    expected = recovery.recovery_metadata(raw,row)
    assert recovery.verified_replacement(raw, expected, row['draft_response_html']) is None
    raw['_embedded']['threads'][-1]['id'] = 3
    result = recovery.verified_replacement(raw, expected, row['draft_response_html'])
    assert result['draft_thread_id'] == 3
    raw['_embedded']['threads'][0]['id'] = 4
    assert recovery.verified_replacement(raw, expected, row['draft_response_html']) is None


def test_existing_human_note_after_draft_is_not_automatically_recovered():
    raw,row=fixture()
    raw['_embedded']['threads'].append({'id':3,'type':'note','state':'published','source':{'type':'web'},'body':'Hold pending my review','createdAt':'2026-10-20T02:00:00Z'})
    assert recovery.recovery_metadata(raw,row)['eligible'] is False


def test_replacement_readback_rejects_competing_human_draft():
    raw,row=fixture(); expected=recovery.recovery_metadata(raw,row)
    raw['_embedded']['threads'].append(dict(raw['_embedded']['threads'][-1],id=3,createdAt='2026-10-20T02:00:00Z'))
    raw['_embedded']['threads'][1]['body']='Human edited older draft'
    assert recovery.verified_replacement(raw,expected,row['draft_response_html']) is None


def test_placeholder_leftover_does_not_block_second_recovery():
    raw,row=fixture()
    raw['_embedded']['threads'].append(dict(raw['_embedded']['threads'][-1],id=3,createdAt='2026-10-19T01:00:00Z',body='<p>Manual review required before sending a reply.</p>'))
    assert recovery.recovery_metadata(raw,row)['eligible']


def test_readback_rejects_new_human_note():
    raw,row=fixture();expected=recovery.recovery_metadata(raw,row)
    raw['_embedded']['threads'][-1]['id']=3
    raw['_embedded']['threads'].append({'id':4,'type':'note','state':'published','source':{'type':'web'},'body':'Hold'})
    assert recovery.verified_replacement(raw,expected,row['draft_response_html']) is None
