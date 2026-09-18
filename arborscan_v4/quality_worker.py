"""One bounded CPU child, renewable database lease, durable jobs, no activation."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4
from .model_quality_api import QualityStore
from .quality_runtime import model_path


def run_once():
    store=QualityStore();lease=str(uuid4());job=store.transition('claim',lease)
    if not job:return False
    root=model_path(job['id']).parent;root.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();process=None
    try:
        with (root/'worker.log').open('ab') as log:
            process=subprocess.Popen([sys.executable,'-m','arborscan_v4.quality_train',job['id']],stdout=log,stderr=log,
               env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},start_new_session=True)
            while True:
                (root.parent.parent/'heartbeat').touch()
                result_file=root/'runs/candidate/results.csv'
                progress={'stage':'training_and_evaluation','elapsed_seconds':int(time.monotonic()-started)}
                stage_file=root/'stage.json'
                if stage_file.is_file():
                    try:progress.update(json.loads(stage_file.read_text()))
                    except (ValueError,OSError):pass
                if result_file.is_file():progress['completed_epochs']=max(0,len(result_file.read_text().splitlines())-1)
                current=store.transition('heartbeat',job['id'],lease_id=lease,progress=progress)
                if current['state']=='cancel_requested':
                    terminate(process);store.transition('finish',job['id'],lease_id=lease,state='cancelled',progress={'stage':'cancelled'});return True
                if time.monotonic()-started>job['params']['max_seconds']:raise RuntimeError('training_time_limit')
                code=process.poll()
                if code is not None:
                    if code!=0:raise RuntimeError('training_subprocess_failed')
                    model=json.loads((root/'result.json').read_text())
                    store.transition('finish',job['id'],lease_id=lease,state='completed',model=model,progress={'stage':'candidate_ready','automatic_activation':False})
                    return True
                time.sleep(5)
    except Exception as error:
        if process is not None:
            try:terminate(process)
            except (ProcessLookupError,subprocess.TimeoutExpired):pass
        safe_code=str(error) if str(error) in ('training_time_limit','training_subprocess_failed') else 'worker_service_or_result_failed'
        try:store.transition('finish',job['id'],lease_id=lease,state='failed',progress={'error':safe_code,'diagnostics':'private_worker_log'})
        except Exception:pass # Lease expiry leaves a durable failed status on next claim.
        return True


def terminate(process):
    import signal
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)


def main():
    import fcntl
    root=Path(os.getenv('MODEL_QUALITY_DIR','/app/model-quality'));root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print('Model quality worker ready: CPU, one job, no automatic activation',flush=True)
        while True:
            (root/'heartbeat').touch()
            try:run_once()
            except Exception:print('Worker service unavailable; will retry without exposing credentials',flush=True)
            time.sleep(5)


if __name__=='__main__':main()
