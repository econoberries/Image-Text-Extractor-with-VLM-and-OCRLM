"""Regression checks for the packages needed by the SLURM GPU workers."""
from pathlib import Path
import ast
import os
import subprocess
import tempfile
import tomllib
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

class HelixEnvironmentContract(unittest.TestCase):
    def test_project_extras_cover_the_previous_failures(self):
        cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        extras = cfg["project"]["optional-dependencies"]
        self.assertTrue(any(x.startswith("torchvision") for x in extras["vlm"]))
        self.assertTrue(any(x.startswith("paddleocr") for x in extras["ocr"]))
        self.assertTrue(cfg["project"]["requires-python"].startswith(">=3.12"))

    def test_cuda_130_profile_is_explicit(self):
        pytorch = (ROOT / "requirements" / "helix-pytorch-cu130.txt").read_text()
        paddle = (ROOT / "requirements" / "helix-paddle-cu130.txt").read_text()
        self.assertIn("torch==2.14.1+cu130", pytorch)
        self.assertIn("torchvision==0.29.1+cu130", pytorch)
        self.assertIn("paddlepaddle-gpu==3.3.0", paddle)
        self.assertIn("/cu130/", paddle)
        self.assertIn("paddlepaddle-gpu", paddle)
        self.assertNotIn("paddlepaddle==", paddle)
        self.assertIn("paddleocr==3.7.0", tomllib.loads(
            (ROOT / "pyproject.toml").read_text()
        )["project"]["optional-dependencies"]["ocr"])

    def test_repair_requirements_match_pytorch(self):
        repair = (ROOT / "requirements" / "helix-torch-repair-cu130.txt").read_text()
        self.assertIn("nvidia-cudnn-cu13==9.24.0.43", repair)
        self.assertIn("nvidia-nccl-cu13==2.30.7", repair)

    def test_isolated_ocr_venv_is_gitignored(self):
        ignore = (ROOT / ".gitignore").read_text()
        self.assertIn("/.venv-ocr/", ignore)

    def test_kie_only_evaluator_requirements(self):
        path = ROOT / "requirements" / "ocrbench-v2-kie-eval.txt"
        packages = [line.strip().lower() for line in path.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
        self.assertIn("apted", packages)
        self.assertIn("distance", packages)
        self.assertIn("ipdb", packages)
        self.assertIn("lxml", packages)
        self.assertNotIn("polygon3", packages)

    def test_environment_checker_parses(self):
        path = ROOT / "scripts" / "check_dependencies.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('find_spec("paddle")', source)
        self.assertIn("paddle.is_compiled_with_cuda()", source)
        self.assertIn("nvidia-cudnn-cu13", source)
        self.assertIn("nvidia-nccl-cu13", source)
        self.assertIn(".venv-ocr", source)

    def test_gpu_workers_use_disjoint_environments(self):
        worker = (ROOT / "slurm" / "worker.sbatch").read_text(encoding="utf-8")
        smoke = (ROOT / "slurm" / "check_environment.sbatch").read_text(encoding="utf-8")
        self.assertIn('ocr) python_bin=".venv-ocr/bin/python"', worker)
        self.assertIn('run) python_bin=".venv/bin/python"', worker)
        self.assertIn('paddle|ocr) python_bin=".venv-ocr/bin/python"', smoke)
        self.assertIn('torch|model) python_bin=".venv/bin/python"', smoke)
        self.assertIn(".venv-ocr", (ROOT / "src" / "ulm_extractor" / "resume.py").read_text())
        self.assertIn(".venv-ocr", (ROOT / "requirements" / "helix-paddle-cu130.txt").read_text())

    def test_remapped_slurm_gpu_id_in_shell_scripts(self):
        # Helix reported SLURM_JOB_GPUS=4 while the sole visible nvidia-smi
        # device was indexed 0. Both smoke tests and workers must accept it.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text(
                '#!/bin/sh\n'
                'if [ "$1" = "--version" ]; then echo "Python 3.12.13"; fi\n'
                'exit 0\n'
            )
            python.chmod(0o755)

            smi = root / "bin" / "nvidia-smi"
            smi.parent.mkdir()
            smi.write_text(
                '#!/bin/sh\n'
                'case "$1" in\n'
                '  --query-gpu=*) printf "%s\\n" "$FAKE_GPU_REPORT" ;;\n'
                '  *) echo "mock nvidia-smi" ;;\n'
                'esac\n'
            )
            smi.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{smi.parent}:{os.environ.get('PATH', '')}",
                "SLURM_SUBMIT_DIR": str(root),
                "SLURM_JOB_ID": "123",
                "SLURM_JOB_GPUS": "4",
                "CUDA_VISIBLE_DEVICES": "0",
                "EXTRACTOR_EXPECT_GPU": "A100",
                "EXTRACTOR_MIN_GPU_MEMORY_MIB": "71680",
            }
            for script, action in (
                ("slurm/check_environment.sbatch", "torch"),
                ("slurm/worker.sbatch", "run"),
            ):
                for gpu_report, expected_code in (
                    ("NVIDIA A100-SXM4-80GB, 81920", 0),
                    ("NVIDIA A100-SXM4-80GB, 81920\\nNVIDIA A100-SXM4-80GB, 81920", 2),
                    ("NVIDIA A100-SXM4-40GB, 40960", 2),
                ):
                    with self.subTest(script=script, gpu_report=gpu_report):
                        result = subprocess.run(
                            ["bash", str(ROOT / script), action],
                            cwd=root, env={**env, "FAKE_GPU_REPORT": gpu_report},
                            capture_output=True, text=True,
                        )
                        self.assertEqual(
                            result.returncode, expected_code,
                            result.stdout + result.stderr,
                        )

    def test_gpu_provenance_uses_one_visible_device(self):
        from ulm_extractor.gpu_hardware import gpu_hardware

        with patch.dict(os.environ, {"SLURM_JOB_GPUS": "4", "CUDA_VISIBLE_DEVICES": "0"}):
            with patch("ulm_extractor.gpu_hardware.subprocess.run") as smi:
                smi.return_value.stdout = "NVIDIA A100-SXM4-80GB, 81920\\n"
                self.assertEqual(
                    gpu_hardware(),
                    {"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920},
                )
                self.assertNotIn("--id", smi.call_args.args[0])
                smi.return_value.stdout = (
                    "NVIDIA A100-SXM4-80GB, 81920\\n"
                    "NVIDIA A100-SXM4-80GB, 81920\\n"
                )
                with self.assertRaisesRegex(RuntimeError, "single allocated GPU"):
                    gpu_hardware()


if __name__ == "__main__":
    unittest.main()
