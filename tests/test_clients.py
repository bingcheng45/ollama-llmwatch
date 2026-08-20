"""Tests for pointing other people's apps at llmwatch.

The proxy only sees what is routed through it, so a client left pointing at the
model server bypasses it in complete silence: every request works, nothing is
measured, and the board looks broken. Fixing that should not require knowing
what a base URL is.

These are somebody's editor configs, so the tests here are mostly about not
damaging them.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    BACKUP_SUFFIX, find_clients, point_client_at, read_client_url, undo_client,
)

OPENCODE = """{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local": {
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      }
    }
  }
}
"""

CODEX = """# my notes, which must survive
model = "qwen"
openai_base_url = "http://127.0.0.1:8080/v1"

[plugins."github"]
enabled = true
"""


class TestReadingWhereAClientPoints(unittest.TestCase):

    def write(self, name, text):
        path = os.path.join(tempfile.mkdtemp(), name)
        with open(path, "w") as fh:
            fh.write(text)
        return path

    def test_json_style(self):
        path = self.write("opencode.json", OPENCODE)
        self.assertEqual(read_client_url(path, "baseURL"),
                         "http://127.0.0.1:8080/v1")

    def test_toml_style(self):
        path = self.write("config.toml", CODEX)
        self.assertEqual(read_client_url(path, "openai_base_url"),
                         "http://127.0.0.1:8080/v1")

    def test_a_missing_file_is_none_not_an_error(self):
        self.assertIsNone(read_client_url("/nonexistent/x.json", "baseURL"))

    def test_a_config_without_the_key_is_none(self):
        path = self.write("x.json", '{"something": "else"}')
        self.assertIsNone(read_client_url(path, "baseURL"))


class TestFindingInstalledClients(unittest.TestCase):

    def test_only_ones_that_exist_and_can_be_read(self):
        path = os.path.join(tempfile.mkdtemp(), "opencode.json")
        with open(path, "w") as fh:
            fh.write(OPENCODE)
        found = find_clients((
            ("opencode", path, "baseURL"),
            ("Ghost", "/nonexistent/nope.json", "baseURL"),
        ))
        self.assertEqual([row["name"] for row in found], ["opencode"])

    def test_a_config_it_cannot_understand_is_not_offered(self):
        """It must not offer to edit a file it could not read a URL out of:
        the edit would be a guess at someone else's config."""
        path = os.path.join(tempfile.mkdtemp(), "weird.json")
        with open(path, "w") as fh:
            fh.write('{"nothing": "familiar"}')
        self.assertEqual(find_clients((("Weird", path, "baseURL"),)), [])


class TestRepointing(unittest.TestCase):

    def client(self, name, text, key):
        path = os.path.join(tempfile.mkdtemp(), name)
        with open(path, "w") as fh:
            fh.write(text)
        return {"name": name, "path": path, "key": key,
                "url": read_client_url(path, key)}

    def test_the_url_changes(self):
        entry = self.client("opencode.json", OPENCODE, "baseURL")
        ok, _msg = point_client_at(entry, "http://127.0.0.1:8081/v1")
        self.assertTrue(ok)
        self.assertEqual(read_client_url(entry["path"], "baseURL"),
                         "http://127.0.0.1:8081/v1")

    def test_nothing_else_in_the_file_changes(self):
        """Their file, not ours to reformat. Reparsing and rewriting would drop
        the comment and reorder the keys."""
        entry = self.client("config.toml", CODEX, "openai_base_url")
        point_client_at(entry, "http://127.0.0.1:8081/v1")
        with open(entry["path"]) as fh:
            after = fh.read()
        self.assertIn("# my notes, which must survive", after)
        self.assertIn('[plugins."github"]', after)
        self.assertEqual(after,
                         CODEX.replace("8080", "8081"))

    def test_a_backup_is_written_before_the_change(self):
        entry = self.client("opencode.json", OPENCODE, "baseURL")
        point_client_at(entry, "http://127.0.0.1:8081/v1")
        with open(entry["path"] + BACKUP_SUFFIX) as fh:
            self.assertEqual(fh.read(), OPENCODE)

    def test_undo_puts_it_back_and_clears_the_backup(self):
        entry = self.client("opencode.json", OPENCODE, "baseURL")
        point_client_at(entry, "http://127.0.0.1:8081/v1")
        ok, _msg = undo_client(entry)
        self.assertTrue(ok)
        with open(entry["path"]) as fh:
            self.assertEqual(fh.read(), OPENCODE)
        self.assertFalse(os.path.exists(entry["path"] + BACKUP_SUFFIX))

    def test_undo_with_nothing_to_undo_says_so_rather_than_raising(self):
        entry = self.client("opencode.json", OPENCODE, "baseURL")
        ok, msg = undo_client(entry)
        self.assertFalse(ok)
        self.assertIn("nothing to undo", msg)

    def test_only_the_first_occurrence_is_touched(self):
        """A config may mention several providers. Rewriting all of them would
        repoint services llmwatch knows nothing about."""
        text = ('{"a": {"baseURL": "http://127.0.0.1:8080/v1"},\n'
                ' "b": {"baseURL": "https://api.openai.com/v1"}}')
        entry = self.client("two.json", text, "baseURL")
        point_client_at(entry, "http://127.0.0.1:8081/v1")
        with open(entry["path"]) as fh:
            after = fh.read()
        self.assertIn("http://127.0.0.1:8081/v1", after)
        self.assertIn("https://api.openai.com/v1", after)

    def test_a_file_without_the_key_is_refused_not_mangled(self):
        entry = {"name": "x", "key": "baseURL", "url": None,
                 "path": os.path.join(tempfile.mkdtemp(), "x.json")}
        with open(entry["path"], "w") as fh:
            fh.write('{"nothing": "familiar"}')
        ok, msg = point_client_at(entry, "http://127.0.0.1:8081/v1")
        self.assertFalse(ok)
        self.assertIn("no baseURL", msg)
        with open(entry["path"]) as fh:
            self.assertEqual(fh.read(), '{"nothing": "familiar"}')

    def test_an_unwritable_file_reports_rather_than_raising(self):
        entry = {"name": "x", "key": "baseURL", "path": "/proc/x/y.json",
                 "url": None}
        ok, _msg = point_client_at(entry, "http://127.0.0.1:8081/v1")
        self.assertFalse(ok)

    def test_repointing_to_where_it_already_points_is_harmless(self):
        entry = self.client("opencode.json", OPENCODE, "baseURL")
        ok, msg = point_client_at(entry, "http://127.0.0.1:8080/v1")
        self.assertTrue(ok)
        self.assertIn("already", msg)
        self.assertFalse(os.path.exists(entry["path"] + BACKUP_SUFFIX))


if __name__ == "__main__":
    unittest.main()
