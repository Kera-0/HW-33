import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
work = root / 'hw3_distillation_work'
stages = [('teacher', []), ('factorized', []), ('student', ['--batch-size', '64'])]
for stage, args in stages:
    print(f'Starting {stage}', flush=True)
    with (work / f'{stage}.log').open('x') as log:
        result = subprocess.run(
            [sys.executable, '-u', str(work / 'experiment.py'), stage, *args],
            cwd=root, stdout=log, stderr=subprocess.STDOUT,
        )
    print(f'{stage} exit code: {result.returncode}', flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
print('All stages complete', flush=True)
