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
