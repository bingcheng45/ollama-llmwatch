#!/usr/bin/env python3
"""Run the whole test suite against the generated single file.

The package and the bundle are two builds of the same program, and only one of
them is what a `curl -O llmwatch.py` user actually runs. Testing the package
alone would leave that build unverified, which is the failure mode this exists
to prevent: the bundler is code, and code that only ever runs at release time
breaks quietly.

Works by assembling a tree that has llmwatch.py but no llmwatch/ directory, so
`import llmwatch` resolves to the single file, and running unittest there.
"""

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The package is deliberately absent; everything else the tests read (README.md
# and friends for test_docs, .github for the community-health checks) is linked
# through so the suite sees a normal checkout.
EXCLUDE = {"llmwatch", ".git", ".venv", "dist", "build", "__pycache__",
           ".pytest_cache", ".ruff_cache"}


def main():
    work = tempfile.mkdtemp(prefix="llmwatch-bundle-")
    try:
        for name in os.listdir(ROOT):
            if name in EXCLUDE or name.endswith(".egg-info"):
                continue
            src, dst = os.path.join(ROOT, name), os.path.join(work, name)
            if name == "llmwatch.py":
                shutil.copy2(src, dst)      # the thing under test
            else:
                os.symlink(src, dst)

        assert not os.path.exists(os.path.join(work, "llmwatch")), \
            "the package leaked into the bundle tree; import would resolve to it"

        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "tests"],
            cwd=work, env=env)
        if result.returncode == 0:
            print("\nbundle: the suite passes against the generated single file")
        return result.returncode
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
