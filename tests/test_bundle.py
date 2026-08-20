"""The generated single file has to match the package it is generated from.

llmwatch ships as two builds: the llmwatch/ package, and the llmwatch.py that
tools/bundle.py concatenates out of it. Only the second is what

    curl -O .../llmwatch.py

gets, so a package edit that never reached the bundle is a change that works
everywhere the maintainers look and nowhere a curl user runs it. That failure
is silent by construction, which is why it is worth a test rather than a note
in CONTRIBUTING.

These skip rather than fail when the package is absent, because the suite is
also run against a tree containing only the bundle (see tools/check_bundle.py),
and there is nothing to compare against there.
"""

import ast
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(ROOT, "llmwatch")
BUNDLE = os.path.join(ROOT, "llmwatch.py")
BUNDLER = os.path.join(ROOT, "tools", "bundle.py")

has_package = os.path.isdir(PACKAGE)


@unittest.skipUnless(has_package, "running against the bundle alone")
class TestBundleIsUpToDate(unittest.TestCase):

    def test_the_committed_file_matches_the_package(self):
        result = subprocess.run([sys.executable, BUNDLER, "--check"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(
            result.returncode, 0,
            "llmwatch.py is stale. Run: python tools/bundle.py\n%s"
            % result.stdout.decode())

    def test_every_module_reached_the_bundle(self):
        """A module that no other imports would be dropped silently by a
        bundler that walked the import graph instead of the directory."""
        bundled = open(BUNDLE).read()
        for name in sorted(os.listdir(PACKAGE)):
            if not name.endswith(".py") or name.startswith("__"):
                continue
            self.assertIn("# %s\n" % name[:-3], bundled,
                          "%s is missing from the bundle" % name)

    def test_the_bundle_defines_everything_the_package_exports(self):
        """The single file has to be the same program, not a subset of it."""
        sys.path.insert(0, ROOT)
        import llmwatch
        tree = ast.parse(open(BUNDLE).read())
        defined = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Assign):
                defined.update(t.id for t in node.targets
                               if isinstance(t, ast.Name))
        missing = sorted(set(llmwatch.__all__) - defined)
        self.assertEqual(missing, [], "not in the single file: %s" % missing)


@unittest.skipUnless(has_package, "running against the bundle alone")
class TestTheLayeringHolds(unittest.TestCase):
    """The package is a DAG, and the bundler depends on it staying one.

    Concatenating modules into one namespace only has a valid answer if the
    imports can be ordered, so a cycle does not degrade the build, it stops it.
    Catching that here names the two modules involved instead of failing at
    release time with a NameError.
    """

    def local_imports(self, name):
        tree = ast.parse(open(os.path.join(PACKAGE, name)).read())
        return {n.module for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module}

    def test_no_import_cycles_between_modules(self):
        modules = [n for n in os.listdir(PACKAGE)
                   if n.endswith(".py") and not n.startswith("__")]
        deps = {n[:-3]: self.local_imports(n) for n in modules}

        placed, remaining = set(), set(deps)
        while remaining:
            ready = {n for n in remaining if deps[n] <= placed}
            if not ready:
                self.fail("import cycle between: %s" % ", ".join(sorted(remaining)))
            placed |= ready
            remaining -= ready

    def test_nothing_imports_the_package_root(self):
        """`from llmwatch import x` inside the package would import __init__,
        which imports every module, which is a cycle through the front door."""
        for name in os.listdir(PACKAGE):
            if not name.endswith(".py") or name == "__init__.py":
                continue
            source = open(os.path.join(PACKAGE, name)).read()
            self.assertNotIn("from llmwatch import", source, name)
            self.assertNotIn("import llmwatch\n", source, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
