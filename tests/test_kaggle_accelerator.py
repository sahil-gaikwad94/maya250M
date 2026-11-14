"""WP-05 tests: the Kaggle trainer must not detect an accelerator and ignore it."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

_PATH = os.path.join(PROJECT_ROOT, "kaggle", "train_maya250m_kaggle.py")
_spec = importlib.util.spec_from_file_location("train_maya250m_kaggle", _PATH)
kaggle_trainer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kaggle_trainer)


def _torch_xla_available() -> bool:
    return importlib.util.find_spec("torch_xla") is not None


def _torch_available() -> bool:
    """True only when torch actually imports (``find_spec`` can find a stub)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


class TestAcceleratorDetection(unittest.TestCase):
    def test_tpu_env_is_detected(self):
        saved = {key: os.environ.get(key) for key in ("PJRT_DEVICE", "TPU_NAME")}
        try:
            os.environ["PJRT_DEVICE"] = "TPU"
            os.environ.pop("TPU_NAME", None)
            hw = kaggle_trainer.detect_kaggle_accelerator()
            self.assertEqual(hw["backend"], "tpu")
            self.assertEqual(hw["dtype"], "bfloat16")
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_cpu_fallback_shape(self):
        saved = {key: os.environ.get(key) for key in ("PJRT_DEVICE", "TPU_NAME")}
        try:
            os.environ.pop("PJRT_DEVICE", None)
            os.environ.pop("TPU_NAME", None)
            hw = kaggle_trainer.detect_kaggle_accelerator()
            if hw["backend"] == "cpu":
                self.assertEqual(hw["batch_size"], 2)
                self.assertEqual(hw["grad_accum"], 2)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class TestDeviceResolution(unittest.TestCase):
    def test_cpu_backend_resolves_to_cpu(self):
        device, strategy = kaggle_trainer.resolve_training_device({"backend": "cpu"})
        self.assertEqual(str(device), "cpu")
        self.assertEqual(strategy["parallel"], "single_cpu")
        self.assertEqual(strategy["world_size"], 1)

    def test_tpu_without_torch_xla_fails_loudly(self):
        if _torch_xla_available():
            self.skipTest("torch_xla is installed; the failure path cannot be exercised")
        saved = os.environ.get("WORLD_SIZE")
        os.environ.pop("WORLD_SIZE", None)
        try:
            with self.assertRaises(SystemExit) as ctx:
                kaggle_trainer.resolve_training_device({"backend": "tpu"})
        finally:
            if saved is not None:
                os.environ["WORLD_SIZE"] = saved
        message = str(ctx.exception)
        self.assertIn("torch_xla", message)
        self.assertIn("refusing", message.lower())

    def test_cuda_backend_without_cuda_fails_loudly(self):
        if not _torch_available():
            self.skipTest("PyTorch not installed")
        import torch

        if torch.cuda.is_available():
            self.skipTest("CUDA is available on this host")
        with self.assertRaises(SystemExit) as ctx:
            kaggle_trainer.resolve_training_device({"backend": "cuda_x2"})
        self.assertIn("cuda", str(ctx.exception).lower())

    def test_multi_process_is_reported_not_faked(self):
        if not _torch_available():
            self.skipTest("PyTorch not installed")
        import torch

        saved = os.environ.get("WORLD_SIZE")
        os.environ["WORLD_SIZE"] = "2"
        try:
            with self.assertRaises(SystemExit) as ctx:
                kaggle_trainer.resolve_training_device({"backend": "cuda_x2"})
        finally:
            if saved is None:
                os.environ.pop("WORLD_SIZE", None)
            else:
                os.environ["WORLD_SIZE"] = saved
        message = str(ctx.exception)
        if torch.cuda.is_available():
            # On a CUDA host the missing DDP implementation is the blocker.
            self.assertIn("NOT_RUN", message)
        else:
            # Without CUDA the accelerator guard fires first; either way the run
            # must fail loudly instead of training on CPU.
            self.assertIn("cuda", message.lower())


class TestFlagshipDataGate(unittest.TestCase):
    """P0.6: the flagship trainer must refuse synthetic fallbacks silently."""

    def test_flagship_invocation_without_data_exits_cleanly(self):
        import subprocess

        completed = subprocess.run(
            [sys.executable, _PATH],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_ROOT,
        )
        self.assertEqual(completed.returncode, 2, completed.stderr[-2000:])
        self.assertIn(
            "flagship training requires explicit data inputs", completed.stderr
        )
        self.assertIn("--smoke-test-synthetic", completed.stderr)

    def test_smoke_fixture_mode_is_an_explicit_opt_in(self):
        source = open(_PATH, encoding="utf-8").read()
        self.assertIn('"--smoke-test-synthetic"', source)
        self.assertIn('checkpoint_role = "smoke_test_only"', source)
        self.assertIn("expected_vocab = 32000", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
