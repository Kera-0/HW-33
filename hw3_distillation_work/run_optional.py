import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
work = root / 'hw3_distillation_work'
stages = [
    ('pruned_init', 'optional.py'),
    ('sharing', 'optional.py'),
    ('sharing_lora', 'optional.py'),
    ('intermediate', 'optional.py'),
    ('wanda', 'pruning.py'),
    ('sparsegpt', 'pruning.py'),
]
failed = []
for stage, script in stages:
    print(f'Starting {stage}', flush=True)
    with (work / f'{stage}.log').open('x') as log:
        result = subprocess.run(
            [sys.executable, '-u', str(work / script), stage],
            cwd=root, stdout=log, stderr=subprocess.STDOUT,
        )
    print(f'{stage} exit code: {result.returncode}', flush=True)
    if result.returncode:
        failed.append(stage)
print(f'Failed stages: {failed}', flush=True)
raise SystemExit(bool(failed))
