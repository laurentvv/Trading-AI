"""Tests du reset étendu : artefacts runtime dans logs_prod/ (CWD scheduler PROD).

Contexte (incident 2026-08-24) : le reset du 19/08 préservait logs_prod/
(KEEP_PATHS) alors que TOUS les artefacts runtime PROD y vivent (le scheduler
y est lancé) — model_performance.db pré-reset a survécu et pollué les poids
adaptatifs pendant 5 jours. Le script détecte désormais ces artefacts et
exige un choix explicite (--include-logs-prod / --keep-logs-prod).

Tout est sandboxé : REPO_ROOT est patché vers un tmpdir, aucun fichier réel
n'est touché.
"""

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import reset_for_fresh_test as rz  # noqa: E402


def make_sandbox(root: Path, with_logs_prod: bool = True) -> None:
    """Fabrique un faux repo : artefacts runtime racine + logs_prod/ + fichiers à préserver."""
    (root / "data_cache").mkdir(parents=True, exist_ok=True)
    (root / "data_cache" / "prices.parquet").write_text("x")
    (root / "trading_journal.csv").write_text("ts,ticker")
    (root / "t212_portfolio_state.json").write_text("{}")
    (root / "trading_history.db").write_text("sqlite")
    (root / "scheduler.lock").write_text("1234")
    (root / "uv.lock").write_text("# config — ne doit JAMAIS être effacé")
    (root / "main.py").write_text("# source")

    if with_logs_prod:
        lp = root / "logs_prod"
        lp.mkdir(parents=True, exist_ok=True)
        (lp / "model_performance.db").write_text("sqlite")
        (lp / "model_performance.db.bak-2026-08-24").write_text("backup pre-reset")
        (lp / "t212_portfolio_state.json").write_text("{}")
        (lp / "trading_journal.csv").write_text("ts,ticker")
        (lp / "scheduler.log").write_text("log")
        (lp / "scheduler.lock").write_text("4321")
        (lp / "audit_report.md").write_text("# rapport d'audit — ne doit pas être effacé")
        (lp / "data_cache").mkdir(exist_ok=True)
        (lp / "data_cache" / "foo.parquet").write_text("x")


class ResetLogsProdTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._orig_root = rz.REPO_ROOT
        rz.REPO_ROOT = self.root

    def tearDown(self):
        rz.REPO_ROOT = self._orig_root
        self._tmp.cleanup()

    def run_main(self, argv):
        orig_argv = sys.argv
        sys.argv = ["reset_for_fresh_test.py"] + argv
        try:
            return rz.main()
        finally:
            sys.argv = orig_argv


class TestDetection(ResetLogsProdTestCase):
    def test_detects_runtime_artifacts_excluding_md(self):
        make_sandbox(self.root)
        targets = rz._collect_logs_prod_runtime_artifacts()
        names = {t.relative_to(self.root).as_posix() for t in targets}
        self.assertEqual(
            names,
            {
                "logs_prod/data_cache",
                "logs_prod/model_performance.db",
                "logs_prod/model_performance.db.bak-2026-08-24",
                "logs_prod/scheduler.lock",
                "logs_prod/scheduler.log",
                "logs_prod/trading_journal.csv",
                "logs_prod/t212_portfolio_state.json",
            },
        )
        # Le rapport d'audit (.md) n'est PAS un artefact runtime.
        self.assertNotIn(str(self.root / "logs_prod" / "audit_report.md"), {str(t) for t in targets})

    def test_no_logs_prod_dir_returns_empty(self):
        make_sandbox(self.root, with_logs_prod=False)
        self.assertEqual(rz._collect_logs_prod_runtime_artifacts(), [])

    def test_is_runtime_file_name_preserves_source_config(self):
        self.assertFalse(rz._is_runtime_file_name("uv.lock"))
        self.assertFalse(rz._is_runtime_file_name("audit_report.md"))
        self.assertFalse(rz._is_runtime_file_name("pyproject.toml"))
        self.assertTrue(rz._is_runtime_file_name("scheduler.lock"))
        self.assertTrue(rz._is_runtime_file_name("model_performance.db.bak-2026-08-24"))
        self.assertTrue(rz._is_runtime_file_name("trading_history.db-wal"))


class TestExplicitChoiceGate(ResetLogsProdTestCase):
    def test_yes_without_flag_refuses(self):
        # Sécurité automation : jamais de wipe NI de conservation silencieuse
        # de l'état PROD vivant (racine de l'incident 2026-08-24).
        make_sandbox(self.root)
        rc = self.run_main(["--yes"])
        self.assertEqual(rc, 2)
        # Rien n'a bougé.
        self.assertTrue((self.root / "logs_prod" / "model_performance.db").exists())
        self.assertTrue((self.root / "trading_journal.csv").exists())
        self.assertFalse((self.root / "reset_backup").exists())

    def test_mutually_exclusive_flags_exit(self):
        make_sandbox(self.root)
        with self.assertRaises(SystemExit):
            self.run_main(["--yes", "--include-logs-prod", "--keep-logs-prod"])

    def test_dry_run_is_noop(self):
        make_sandbox(self.root)
        rc = self.run_main(["--dry-run"])
        self.assertEqual(rc, 0)
        self.assertTrue((self.root / "logs_prod" / "model_performance.db").exists())
        self.assertTrue((self.root / "data_cache" / "prices.parquet").exists())
        self.assertFalse((self.root / "reset_backup").exists())


class TestWipe(ResetLogsProdTestCase):
    def test_include_logs_prod_wipes_and_backs_up(self):
        make_sandbox(self.root)
        rc = self.run_main(["--yes", "--include-logs-prod"])
        self.assertEqual(rc, 0)

        # Artefacts runtime logs_prod/ effacés...
        lp = self.root / "logs_prod"
        self.assertFalse((lp / "model_performance.db").exists())
        self.assertFalse((lp / "model_performance.db.bak-2026-08-24").exists())
        self.assertFalse((lp / "t212_portfolio_state.json").exists())
        self.assertFalse((lp / "data_cache").exists())
        # ...mais le rapport d'audit (.md) et le répertoire logs_prod/ restent.
        self.assertTrue((lp / "audit_report.md").exists())
        self.assertTrue(lp.is_dir())

        # Racine : runtime effacé, source/config préservé.
        self.assertFalse((self.root / "trading_journal.csv").exists())
        self.assertFalse((self.root / "data_cache").exists())
        self.assertTrue((self.root / "uv.lock").exists())
        self.assertTrue((self.root / "main.py").exists())

        # Tout est réversible : présent sous reset_backup/<ts>/...
        backups = list((self.root / "reset_backup").iterdir())
        self.assertEqual(len(backups), 1)
        stamp_dir = backups[0]
        self.assertTrue((stamp_dir / "logs_prod" / "model_performance.db").exists())
        self.assertTrue((stamp_dir / "logs_prod" / "model_performance.db.bak-2026-08-24").exists())
        self.assertTrue((stamp_dir / "trading_journal.csv").exists())
        self.assertTrue((stamp_dir / "data_cache" / "prices.parquet").exists())

    def test_keep_logs_prod_preserves_artifacts(self):
        make_sandbox(self.root)
        rc = self.run_main(["--yes", "--keep-logs-prod"])
        self.assertEqual(rc, 0)
        # logs_prod/ intact (cas DEV : snapshot d'audit), racine nettoyée.
        self.assertTrue((self.root / "logs_prod" / "model_performance.db").exists())
        self.assertTrue((self.root / "logs_prod" / "data_cache" / "foo.parquet").exists())
        self.assertFalse((self.root / "trading_journal.csv").exists())
        self.assertFalse((self.root / "data_cache").exists())


class TestInteractiveChoice(ResetLogsProdTestCase):
    def test_interactive_yes_includes_logs_prod(self):
        make_sandbox(self.root)
        orig_confirm, orig_gate = rz.confirm, rz.dry_run_or_confirm
        try:
            rz.confirm = lambda *a, **k: True          # réponse "oui" à la question logs_prod
            rz.dry_run_or_confirm = lambda *a, **k: None  # passe la confirmation globale
            rc = self.run_main([])
        finally:
            rz.confirm, rz.dry_run_or_confirm = orig_confirm, orig_gate
        self.assertEqual(rc, 0)
        self.assertFalse((self.root / "logs_prod" / "model_performance.db").exists())

    def test_interactive_no_keeps_logs_prod(self):
        make_sandbox(self.root)
        orig_confirm, orig_gate = rz.confirm, rz.dry_run_or_confirm
        try:
            rz.confirm = lambda *a, **k: False         # réponse "non" à la question logs_prod
            rz.dry_run_or_confirm = lambda *a, **k: None
            rc = self.run_main([])
        finally:
            rz.confirm, rz.dry_run_or_confirm = orig_confirm, orig_gate
        self.assertEqual(rc, 0)
        self.assertTrue((self.root / "logs_prod" / "model_performance.db").exists())


if __name__ == "__main__":
    unittest.main()
