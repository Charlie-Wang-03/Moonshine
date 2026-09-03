"""Regression test: store operations must close their sqlite connections.

Real Windows bug: ``with sqlite3.connect(...) as conn`` commits the transaction
on exit but NEVER closes the connection (the closing-context-manager gotcha).
``moonshine_state.SessionStateDB``, ``storage.knowledge_store.KnowledgeStore``
and ``storage.knowledge_vector_store.SQLiteVectorBackend`` all used that
pattern for every operation, so each call leaked an open handle on the database
file. On Windows those lingering handles lock the file — tempdir cleanup fails
with PermissionError WinError 32, which was the dominant failure mode of the
whole Windows test suite and was masked by
``tempfile.TemporaryDirectory(ignore_cleanup_errors=True)`` in existing tests.
"""

from __future__ import annotations

import tempfile
import unittest

from moonshine.app import MoonshineApp


class SqliteConnectionCleanupTest(unittest.TestCase):
    def test_app_usage_leaves_no_locked_files(self):
        """After ordinary app use the home dir must be fully deletable.

        TemporaryDirectory without ignore_cleanup_errors raises PermissionError
        at cleanup if any store leaked an open handle on a Windows-locked file.
        """
        temp_dir = tempfile.TemporaryDirectory()  # deliberately strict cleanup
        self.addCleanup(temp_dir.cleanup)

        app = MoonshineApp(home=temp_dir.name)
        app.start_shell_state(mode="research", project_slug="lock_repro")
        app.agent.memory_manager.knowledge_store.add_conclusion(
            title="cleanup probe",
            statement="store writes must not leak sqlite handles",
            project_slug="lock_repro",
        )
        # Leaving the method drops the app reference; addCleanup then deletes
        # the whole home tree, which fails on Windows while any handle is open.


if __name__ == "__main__":
    unittest.main()
