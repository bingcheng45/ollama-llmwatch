"""Guards on the text that users actually read."""

import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(name):
    with open(os.path.join(ROOT, name)) as fh:
        return fh.read()


class TestPublishedText(unittest.TestCase):

    FILES = ["README.md", "CHANGELOG.md",
             ".github/ISSUE_TEMPLATE/parser_drift.md",
             ".github/ISSUE_TEMPLATE/wrong_number.md",
             ".github/ISSUE_TEMPLATE/feedback.md"]

    def test_no_em_dashes(self):
        for name in self.FILES:
            self.assertNotIn("—", read(name), "em dash in %s" % name)

    def test_no_en_dashes(self):
        for name in self.FILES:
            self.assertNotIn("–", read(name), "en dash in %s" % name)

    def test_install_command_is_near_the_top(self):
        """A newcomer landing on PyPI should not have to scroll to find it."""
        lines = read("README.md").splitlines()
        index = next(i for i, l in enumerate(lines) if "uv tool install" in l)
        self.assertLess(index, 40)

    def test_no_mermaid_blocks(self):
        """GitHub renders mermaid; PyPI shows it as a raw code block, and PyPI is
        where most people see this first."""
        self.assertNotIn("```mermaid", read("README.md"))

    def test_faq_uses_headings_not_a_wall_of_bold(self):
        faq = read("README.md").split("## FAQ")[1].split("## How it works")[0]
        self.assertGreater(faq.count("\n### "), 8)


class TestVersionIsConsistent(unittest.TestCase):
    """One version, in three places that ship to different audiences.

    This drifted: the README's sample board still advertised 0.7.0 while the
    code and the package had moved on, so the first thing anyone saw on PyPI was
    a screenshot of the previous release.
    """

    def version(self):
        import re
        match = re.search(r'__version__ = "([^"]+)"', read("llmwatch.py"))
        return match.group(1)

    def test_pyproject_matches_the_module(self):
        self.assertIn('version = "%s"' % self.version(), read("pyproject.toml"))

    def test_the_readme_screenshot_is_not_from_a_previous_release(self):
        self.assertIn("ollama-llmwatch %s " % self.version(), read("README.md"))

    def test_the_changelog_leads_with_this_version(self):
        first = read("CHANGELOG.md").split("## ")[1].splitlines()[0].strip()
        self.assertEqual(first, self.version())


class TestNoShadowedDefinitions(unittest.TestCase):
    """A duplicate top-level definition silently shadows the earlier one.

    This actually happened: a scripted edit inserted a second `_key_reader`
    instead of replacing the first, so the old byte-at-a-time reader kept
    running and arrow keys stayed broken while the new code sat unreachable
    with passing unit tests.
    """

    def test_every_top_level_name_is_defined_once(self):
        import ast
        import collections
        tree = ast.parse(read("llmwatch.py"))
        names = [node.name for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        duplicates = [name for name, count in collections.Counter(names).items() if count > 1]
        self.assertEqual(duplicates, [], "shadowed definitions: %s" % duplicates)

    def test_no_unreachable_dead_helpers(self):
        """Anything defined but never referenced is either dead or a bug."""
        import ast
        source = read("llmwatch.py")
        tree = ast.parse(source)
        defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)
                   and not n.name.startswith("__")}
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        orphans = sorted(defined - used - {"main"})
        self.assertEqual(orphans, [], "defined but never used: %s" % orphans)
