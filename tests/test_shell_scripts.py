import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_all_shell_scripts_pass_bash_syntax_check():
    scripts = sorted(REPOSITORY_ROOT.rglob("*.sh"))
    assert scripts
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
