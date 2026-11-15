import subprocess
import sys
import os
import time

test_files = [
    'test_health.py',
    'test_auth.py',
    'test_models.py',
    'test_uploads.py',
    'test_chat.py',
    'test_reports.py',
    'test_history.py',
    'test_analyses.py',
    'test_explainability.py',
    'test_separate_workflows.py',
    'test_model_selection_persistence.py',
    'test_persistence_and_performance.py',
    'test_segmentation_routing_and_xai.py',
    'test_dynamic_workflows_and_xai.py',
    'test_brats_3d_service.py'
]

env = os.environ.copy()
env['PYTHONPATH'] = 'backend'
env['TF_ENABLE_ONEDNN_OPTS'] = '0'

results = {}
for tf in test_files:
    test_path = os.path.join('backend', 'tests', tf)
    print(f'Running {tf}...', end=' ', flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, '-m', 'pytest', '-c', 'pytest.ini', test_path, '-v', '--tb=short'],
            env=env,
            capture_output=True,
            text=True,
            timeout=40
        )
        dt = round(time.time() - t0, 2)
        passed = proc.returncode == 0
        status_str = f'PASSED ({dt}s)' if passed else f'FAILED (code {proc.returncode}, {dt}s)'
        print(status_str)
        results[tf] = {
            'status': status_str,
            'dt': dt,
            'passed': passed,
            'stdout': proc.stdout,
            'stderr': proc.stderr
        }
        if not passed:
            print("--- Output snippet ---")
            lines = proc.stdout.splitlines()[-15:]
            for line in lines:
                print("  ", line)
    except subprocess.TimeoutExpired as te:
        dt = round(time.time() - t0, 2)
        print(f'TIMEOUT/HANG after {dt}s!')
        results[tf] = {'status': f'TIMEOUT/HANG ({dt}s)', 'dt': dt, 'passed': False, 'stdout': '', 'stderr': ''}

print('\n' + '='*60)
print('SUMMARY OF TEST EXECUTION')
print('='*60)
for tf, res in results.items():
    print(f"{tf:38}: {res['status']}")

# Save diagnostic log
os.makedirs('docs', exist_ok=True)
with open('docs/test_diagnosis_baseline.txt', 'w', encoding='utf-8') as f:
    f.write('PHASE 0 TEST DIAGNOSIS BASELINE\n')
    f.write('='*60 + '\n')
    for tf, res in results.items():
        f.write(f"{tf:38}: {res['status']}\n")
    f.write('\n\nDETAILS:\n' + '='*60 + '\n')
    for tf, res in results.items():
        f.write(f"\n--- {tf} ---\n")
        f.write(res.get('stdout', '') + '\n')
        if res.get('stderr'):
            f.write("STDERR:\n" + res.get('stderr') + '\n')

