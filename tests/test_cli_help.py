import subprocess, sys

def test_cli_help_runs_fast():
    # --help 실행만 확인(아주 빠름)
    r = subprocess.run([sys.executable, "-m", "project.run_experiment", "--help"],
                       capture_output=True, text=True, timeout=20)
    assert r.returncode == 0
