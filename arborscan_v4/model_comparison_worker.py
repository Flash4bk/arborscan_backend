"""Internal disposable comparison child, supervised by model_comparison_api."""
from pathlib import Path
import json
import socket
import sys


def _deny_network(*args, **kwargs):
    raise OSError('Network is disabled in model comparison')


def main(job_path, result_path):
    # CPU time includes model loading/imports. Windows uses the parent's hard
    # wall-clock process termination; it is not labelled a CPU hard limit.
    cpu_enforced = False
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
        cpu_enforced = True
    except (ImportError, OSError, ValueError):
        pass
    from .model_comparison_api import _run_models, ComparisonError, MAX_UPLOAD_BYTES
    try:
        job = json.loads(Path(job_path).read_text(encoding='utf-8'))
        folder = Path(job_path).parent
        with (folder / 'image.bin').open('rb') as source:
            raw = source.read(MAX_UPLOAD_BYTES + 1)
        if not raw or len(raw) > MAX_UPLOAD_BYTES:
            raise ComparisonError(413, 'Image exceeds upload limit')
        # Validate before importing the heavier tensor runtime.
        import cv2
        cv2.setNumThreads(1)
        from .model_comparison_api import _decode_frame
        _decode_frame(raw)
        socket.create_connection = _deny_network
        socket.getaddrinfo = _deny_network
        socket.socket.connect = _deny_network
        socket.socket.connect_ex = _deny_network
        socket.socket.sendto = _deny_network
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        value = _run_models(raw, job, folder / 'weights', cpu_enforced)
    except ComparisonError as exc:
        value = {'error': {'status': exc.status, 'detail': exc.detail}}
    except Exception:
        value = {'error': {'status': 503, 'detail': 'Model comparison could not finish'}}
    Path(result_path).write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    return 0 if 'error' not in value else 1


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
