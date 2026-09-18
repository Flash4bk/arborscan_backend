import json
from unittest.mock import Mock
import pytest
from arborscan_v4 import quality_worker as w


@pytest.mark.parametrize('state,exit_code,terminal', [('cancel_requested',None,'cancelled'),('running',1,'failed'),('running',0,'completed')])
def test_worker_terminal_states_are_not_training_evidence(tmp_path,monkeypatch,state,exit_code,terminal):
    monkeypatch.setenv('MODEL_QUALITY_DIR',str(tmp_path))
    job={'id':'00000000-0000-4000-8000-000000000001','params':{'max_seconds':1800}}
    store=Mock();calls=[]
    def transition(action,key,**data):
        calls.append((action,data))
        if action=='claim':return job
        if action=='heartbeat':return {'state':state}
    store.transition.side_effect=transition
    monkeypatch.setattr(w,'QualityStore',lambda:store)
    process=Mock();process.poll.return_value=exit_code
    def start(*args,**kwargs):
        assert args[0][1:3]==['-m','arborscan_v4.quality_train']
        assert kwargs['env']['OMP_NUM_THREADS']=='1'
        root=w.model_path(job['id']).parent
        (root/'result.json').write_text(json.dumps({'synthetic_test':True,'eligible_for_activation':False}))
        return process
    monkeypatch.setattr(w.subprocess,'Popen',start)
    terminate=Mock();monkeypatch.setattr(w,'terminate',terminate)
    assert w.run_once()
    assert calls[-1][0]=='finish' and calls[-1][1]['state']==terminal
    assert all(action!='activate' for action,_ in calls)
    if terminal!='completed':terminate.assert_called_once_with(process)


def test_lost_lease_terminates_child_without_registering_candidate(tmp_path,monkeypatch):
    monkeypatch.setenv('MODEL_QUALITY_DIR',str(tmp_path))
    store=Mock();store.transition.side_effect=[{'id':'00000000-0000-4000-8000-000000000002','params':{'max_seconds':1800}},RuntimeError('lease lost'),RuntimeError('lease lost')]
    monkeypatch.setattr(w,'QualityStore',lambda:store)
    child=Mock();monkeypatch.setattr(w.subprocess,'Popen',lambda *a,**k:child)
    terminate=Mock();monkeypatch.setattr(w,'terminate',terminate)
    assert w.run_once();terminate.assert_called_once_with(child)
    assert store.transition.call_args.kwargs['state']=='failed'
