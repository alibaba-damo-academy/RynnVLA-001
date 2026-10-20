"""Device selection tests for the inference path. The mocked tests need no accelerator."""

import contextlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import device as device_utils  # noqa: E402

XPU_PRESENT = device_utils.is_available("xpu")


def fake_host(xpu=False, cuda=False):
    available = {"xpu": xpu, "cuda": cuda, "cpu": True}
    return mock.patch.object(device_utils, "is_available", side_effect=lambda name: available[name])


class DeviceModuleTest(unittest.TestCase):
    def test_default_prefers_xpu_then_cuda_then_cpu(self):
        with fake_host(xpu=True, cuda=True):
            self.assertEqual(device_utils.default_device_type(), "xpu")
        with fake_host(cuda=True):
            self.assertEqual(device_utils.default_device_type(), "cuda")
        with fake_host():
            self.assertEqual(device_utils.default_device_type(), "cpu")

    def test_resolve_device(self):
        with fake_host(xpu=True, cuda=True):
            self.assertEqual(device_utils.resolve_device(None), torch.device("xpu", 0))
            self.assertEqual(device_utils.resolve_device("cuda"), torch.device("cuda", 0))
            self.assertEqual(device_utils.resolve_device("xpu:1"), torch.device("xpu", 1))
            self.assertEqual(device_utils.resolve_device("cpu"), torch.device("cpu"))
        with fake_host(cuda=True):
            self.assertEqual(device_utils.resolve_device(None), torch.device("cuda", 0))
        with fake_host():
            self.assertEqual(device_utils.resolve_device(None), torch.device("cpu"))

    def test_device_type_of_tensor_and_module(self):
        self.assertEqual(device_utils.device_type(torch.zeros(1)), "cpu")
        self.assertEqual(device_utils.device_type(torch.nn.Linear(1, 1)), "cpu")
        self.assertEqual(device_utils.device_type("xpu:0"), "xpu")

    def test_autocast_enabled(self):
        self.assertTrue(device_utils.autocast_enabled("cuda:0", torch.bfloat16))
        self.assertTrue(device_utils.autocast_enabled("cuda:0", torch.float32))
        self.assertTrue(device_utils.autocast_enabled("xpu:0", torch.bfloat16))
        self.assertFalse(device_utils.autocast_enabled("xpu:0", torch.float32))
        self.assertFalse(device_utils.autocast_enabled("cpu", torch.bfloat16))

    def test_empty_cache_dispatch(self):
        with fake_host(xpu=True, cuda=True), mock.patch.object(
            torch.cuda, "empty_cache"
        ) as cuda_empty, mock.patch.object(torch.xpu, "empty_cache") as xpu_empty:
            device_utils.empty_cache("cuda:0")
            cuda_empty.assert_called_once()
            xpu_empty.assert_not_called()
            device_utils.empty_cache("xpu:0")
            xpu_empty.assert_called_once()
            device_utils.empty_cache("cpu")
            self.assertEqual(cuda_empty.call_count + xpu_empty.call_count, 2)


@contextlib.contextmanager
def inferencer_workdir():
    """Working directory with the fixed relative files the inferencer opens."""
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        tok_dir = os.path.join(tmp, "pretrained_models", "Chameleon", "original_tokenizers")
        os.makedirs(tok_dir)
        with open(os.path.join(tok_dir, "text_tokenizer.json"), "w", encoding="utf8") as f:
            json.dump({"model": {"vocab": {}}}, f)
        stats = {k: [0.0] * 6 for k in (
            "rel_min_action", "rel_max_action", "rel_mean_action", "rel_std_action", "mean_state", "std_state")}
        stats_path = os.path.join(tmp, "stats.json")
        with open(stats_path, "w", encoding="utf8") as f:
            json.dump(stats, f)
        os.chdir(tmp)
        try:
            yield stats_path
        finally:
            os.chdir(old)


class InferencerDispatchTest(unittest.TestCase):
    def build(self, device, host):
        from inferencer import lerobot_inferencer as li

        with inferencer_workdir() as stats_path:
            configs = {
                "actionvae_config": {},
                "actionvae_pretrained_path": "vae.pth",
                "train_dataset": {
                    "img_size": 384, "use_rel_action": True, "min_max_norm": False,
                    "mean_std_norm": True, "data_path": stats_path,
                },
            }
            model = mock.MagicMock()
            with host, mock.patch.object(li.RynnVLAForActionPrediction, "from_pretrained", return_value=model) as load, \
                    mock.patch.object(li, "chameleon_vae_ori"), mock.patch.object(li, "Tokenizer"), \
                    mock.patch.object(li, "ActionVAE"), mock.patch.object(li.torch, "load", return_value={}) as torch_load:
                li.LeRobotInferencer("ckpt", "bf16", configs, device)
        return load, torch_load

    def test_loader_gets_resolved_device(self):
        load, _ = self.build(None, fake_host(xpu=True))
        self.assertEqual(load.call_args.kwargs["device_map"], torch.device("xpu", 0))
        load, _ = self.build(None, fake_host(cuda=True))
        self.assertEqual(load.call_args.kwargs["device_map"], torch.device("cuda", 0))
        load, _ = self.build(None, fake_host())
        self.assertEqual(load.call_args.kwargs["device_map"], torch.device("cpu"))

    def test_explicit_device_wins(self):
        load, _ = self.build("cpu", fake_host(xpu=True, cuda=True))
        self.assertEqual(load.call_args.kwargs["device_map"], torch.device("cpu"))
        load, _ = self.build(torch.device("cuda", 1), fake_host(xpu=True))
        self.assertEqual(load.call_args.kwargs["device_map"], torch.device("cuda", 1))

    def test_action_vae_checkpoint_loaded_on_cpu(self):
        _, torch_load = self.build("cpu", fake_host())
        self.assertEqual(torch_load.call_args.kwargs["map_location"], "cpu")


def tiny_model(dtype):
    from models.chameleon_model.chameleon import ChameleonConfig, RynnVLAForActionPrediction

    config = ChameleonConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
        state_dim=12, action_head_type="one_layer", replace_inputs_with_action=False,
        mask_image_logits=False, vocabulary_map={}, attn_implementation="sdpa",
    )
    torch.manual_seed(0)
    model = RynnVLAForActionPrediction(config).to(dtype).eval()
    with torch.no_grad():
        for p in model.parameters():  # no zero-initialised layer may hide a mismatch
            p.copy_(torch.randn_like(p) * 0.05)
    return model


@unittest.skipUnless(XPU_PRESENT, "needs an Intel XPU")
class TinyModelParityTest(unittest.TestCase):
    def test_forward_matches_cpu(self):
        model = tiny_model(torch.float32)
        ids = torch.randint(0, 100, (1, 24))
        ids[0, -1] = model.state_idx
        state = [torch.randn(12)]

        def run(m, device):
            with torch.no_grad():
                out = m(input_ids=ids.clone().to(device), state_embeds=[s.to(device) for s in state], return_dict=True)
            return out.logits.float().cpu(), out.action_pred.float().cpu()

        cpu_logits, cpu_action = run(model, "cpu")
        xpu_logits, xpu_action = run(model.to("xpu"), "xpu")
        self.assertGreater(cpu_logits.std().item(), 1e-3)
        torch.testing.assert_close(xpu_logits, cpu_logits, atol=2e-3, rtol=1e-3)
        torch.testing.assert_close(xpu_action, cpu_action, atol=2e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
