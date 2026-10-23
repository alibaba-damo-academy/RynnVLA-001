import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch


class InferenceDeviceTest(unittest.TestCase):
    def test_cli_device_index_is_passed_to_inferencer_without_loading_model(self):
        captured = {}

        torch = types.ModuleType("torch")
        torch.device = lambda value: value

        yaml = types.ModuleType("yaml")
        yaml.FullLoader = object()
        yaml.load = lambda stream, Loader=None: {}

        easydict = types.ModuleType("easydict")
        easydict.EasyDict = dict

        inferencer_package = types.ModuleType("inferencer")
        inferencer_package.__path__ = []
        inferencer_module = types.ModuleType("inferencer.lerobot_inferencer")

        class FakeInferencer:
            def __init__(self, checkpoint, precision, configs, device):
                captured["device"] = device

            def step(self, obs, goal):
                return None, None

        inferencer_module.LeRobotInferencer = FakeInferencer

        modules = {
            "torch": torch,
            "yaml": yaml,
            "easydict": easydict,
            "inferencer": inferencer_package,
            "inferencer.lerobot_inferencer": inferencer_module,
        }
        script = Path(__file__).resolve().parents[1] / "inference_lerobot.py"
        argv = [
            str(script),
            "--vla_ckpt_path", "synthetic-checkpoint",
            "--action_vae_path", "synthetic-action-vae",
            "--configs_path", "synthetic-config.yaml",
            "--device", "4",
        ]

        with patch.dict(sys.modules, modules):
            with patch.object(sys, "argv", argv):
                with patch("builtins.open", mock_open(read_data="{}")):
                    namespace = runpy.run_path(str(script), run_name="rynnvla_entry_test")
                    namespace["main"]()

        self.assertEqual(captured["device"], "cuda:4")


if __name__ == "__main__":
    unittest.main()
