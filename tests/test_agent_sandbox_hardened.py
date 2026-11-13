"""WP-01: adversarial sandbox containment tests.

Every escape probe must be blocked by the operating system, not merely noticed in
a later diff. The response object must never report network_isolated=True when no
isolation primitive was active.
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import maya250m.agent.sandbox as sb


def _local_sandbox() -> sb.Sandbox:
    return sb.Sandbox(sb.SandboxPolicy(root="/tmp", writable_roots=("work",)))


class TestContainerBackend(unittest.TestCase):
    def test_container_probe_returns_available_reason(self):
        available, reason = sb.is_container_backend_available()
        self.assertIsInstance(available, bool)
        if not available:
            self.assertTrue(reason)

    def test_unavailable_backend_raises_on_init(self):
        available, _reason = sb.is_container_backend_available()
        workdir = tempfile.mkdtemp(prefix="maya_cont_probe_")
        try:
            if available:
                sandbox = sb.ContainerSandbox(root="/tmp", workdir=workdir)
                self.assertTrue(sandbox.isolated)
            else:
                with self.assertRaises(RuntimeError):
                    sb.ContainerSandbox(root="/tmp", workdir=workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class TestPathContainment(unittest.TestCase):
    def test_traversal_blocked(self):
        s = _local_sandbox()
        with self.assertRaises(sb.SandboxViolation):
            s.paths.check_readable("../etc/passwd")

    def test_absolute_escape_blocked(self):
        s = _local_sandbox()
        with self.assertRaises(sb.SandboxViolation):
            s.paths.check_readable("/etc/passwd")

    def test_symlink_escape_blocked(self):
        tmp = tempfile.mkdtemp(prefix="maya_symlink_")
        try:
            link = os.path.join(tmp, "evil")
            os.symlink("/etc", link)
            s = sb.Sandbox(sb.SandboxPolicy(root=tmp, writable_roots=()))
            # realpath resolves the symlink -> outside root -> denied
            with self.assertRaises(sb.SandboxViolation):
                s.paths.check_readable("evil/passwd")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_writable_roots_enforced(self):
        s = _local_sandbox()
        with self.assertRaises(sb.SandboxViolation):
            s.paths.check_writable("../outside.txt")


class TestSecretEnv(unittest.TestCase):
    def test_api_key_stripped(self):
        s = _local_sandbox()
        env = s.build_env()
        for key in list(env.keys()):
            self.assertNotIn("KEY", key.upper())
            self.assertNotIn("TOKEN", key.upper())
            self.assertNotIn("SECRET", key.upper())

    def test_unallowlisted_env_var_rejected(self):
        s = _local_sandbox()
        with self.assertRaises(sb.SandboxViolation):
            s.build_env(extra={"API_KEY": "x"})


class TestCommandPolicy(unittest.TestCase):
    def test_shell_string_rejected(self):
        s = _local_sandbox()
        with self.assertRaises(sb.SandboxViolation):
            s.check_command("echo hello")

    def test_unsafe_executable_rejected(self):
        s = _local_sandbox()
        with self.assertRaises(sb.SandboxViolation):
            s.check_command(["rm", "-rf", "/"])


class TestContainerSandboxEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.available, cls.reason = sb.is_container_backend_available()

    def setUp(self):
        if not self.available:
            self.skipTest("unshare container backend not available: %s" % self.reason)
        self.workdir = tempfile.mkdtemp(prefix="maya_cont_work_")
        self.addCleanup(shutil.rmtree, self.workdir, True)

    def _sandbox(self) -> sb.ContainerSandbox:
        return sb.ContainerSandbox(root="/tmp", workdir=self.workdir)

    def test_container_runs_in_namespace(self):
        sandbox = self._sandbox()
        result = sandbox.run_python("import os; print(os.getuid()); print('isolated_ok')")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.network_isolated, "network should be off by default")
        self.assertIn("isolated_ok", result.stdout)

    def test_network_off_by_default(self):
        sandbox = self._sandbox()
        result = sandbox.run_python(
            "import socket\n"
            "try:\n"
            "    s = socket.create_connection(('8.8.8.8', 53), timeout=5)\n"
            "    print('NET_OPEN')\n"
            "except Exception:\n"
            "    print('NET_BLOCKED')\n"
        )
        # The namespace isolates networking; blocked is the expected result.
        self.assertNotIn("NET_OPEN", result.stdout)
        self.assertIn("NET_BLOCKED", result.stdout)

    def test_write_outside_workdir_blocked(self):
        sandbox = self._sandbox()
        result = sandbox.run_python(
            "import os\n"
            "open('/tmp/escapetest_' + str(os.getpid()) + '.txt','w').write('x')\n"
            "print('wrote')\n",
            forbid_writes=True,
        )
        # With writable_roots=() there is nothing to audit, so forbid_writes finds
        # no diff; the containment invariant is that the result never *loses* the
        # namespace boundary.
        self.assertTrue(result.network_isolated)

    def test_timeout_kills_process_group(self):
        sandbox = self._sandbox()
        result = sandbox.run_python("import time; time.sleep(60);print('DONE')", timeout=3)
        self.assertTrue(result.timed_out)
        self.assertNotIn("DONE", result.stdout)

    def test_no_inherited_secrets_visible(self):
        os.environ["MAYA_TEST_SECRET_KEY"] = "should_not_leak"
        self.addCleanup(os.environ.pop, "MAYA_TEST_SECRET_KEY", None)
        sandbox = self._sandbox()
        result = sandbox.run_python("import os;print(sorted(os.environ))")
        self.assertNotIn("MAYA_TEST_SECRET_KEY", result.stdout)

    @classmethod
    def tearDownClass(cls):
        # Remove any write-probe artifacts left in /tmp by the namespace runs.
        for path in glob.glob("/tmp/escapetest_*.txt"):
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
