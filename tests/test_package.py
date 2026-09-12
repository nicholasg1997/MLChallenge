import os
import subprocess
import sys


def test_package_imports_from_a_neutral_cwd_without_path_hacks(tmp_path):
    # A fresh interpreter, cwd outside the repo, PYTHONPATH stripped: this can
    # only pass if the package is genuinely installed into the environment.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", "import meld_emotion.config as c; print(c.SPLITS)"],
        cwd=tmp_path, capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "train" in result.stdout
