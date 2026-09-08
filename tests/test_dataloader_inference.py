import tempfile
import unittest
from pathlib import Path

from PIL import Image

from aicomp_grounding.models.base import ModelInput, Prediction
from offline.infer import _run_dataloader_inference_loop



class _FakePreparedAdapter:
    model_name = "fake-prepared"
    model_revision = "rev"
    supports_prepared_inputs = True

    def __init__(self):
        self.loaded = False
        self.prepared_batches: list[list[str]] = []

    def load(self, *, device="cuda", lora_path=None, model_path=None):
        self.loaded = True

    def prepare_inputs(self, samples: list[ModelInput]) -> dict:
        self.prepared_batches.append([s.key for s in samples])
        return {"queries": [s.query for s in samples]}

    def predict_from_inputs(self, inputs: dict) -> list[Prediction]:
        return [
            Prediction(bbox=[0.1, 0.2, 0.3, 0.4], score=None)
            for _ in inputs["queries"]
        ]


class _FakeLegacyAdapter:
    model_name = "fake-legacy"
    model_revision = "rev"

    def __init__(self):
        self.loaded = False

    def load(self, *, device="cuda", lora_path=None, model_path=None):
        self.loaded = True

    def predict(self, samples: list[ModelInput]) -> list[Prediction]:
        return [Prediction(bbox=[0.5, 0.5, 0.6, 0.6], score=None) for _ in samples]


def _make_items(root: Path, count: int) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / "frame.png"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_path)
    return [
        {
            "key": f"seq_{i:03d}_0001",
            "visible": "frame.png",
            "infrared": "frame.png",
            "depth": "frame.png",
            "query": f"the object number {i}",
        }
        for i in range(count)
    ]


class DataLoaderInferenceLoopTests(unittest.TestCase):
    def _run(self, adapter, items, data_dir, **kwargs):
        return _run_dataloader_inference_loop(
            items,
            adapter,
            adapter_dir=None,
            model_path=None,
            data_dir=data_dir,
            batch_size=2,
            batch_save=10,
            num_workers=0,
            device="cpu",
            **kwargs,
        )

    def test_prepared_path_predicts_all_keys_in_order(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            items = _make_items(root, 5)
            adapter = _FakePreparedAdapter()
            predictions = self._run(adapter, items, root)
            self.assertEqual(
                sorted(predictions),
                sorted(item["key"] for item in items),
            )
            self.assertTrue(all(p == [0.1, 0.2, 0.3, 0.4] for p in predictions.values()))
            self.assertTrue(adapter.loaded)
            # Batching preserved: 5 items at batch_size 2 -> 3 batches.
            self.assertEqual([len(b) for b in adapter.prepared_batches], [2, 2, 1])
            self.assertEqual(adapter.prepared_batches[0][0], "seq_000_0001")

    def test_prepared_path_with_worker_processes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            items = _make_items(root, 4)
            adapter = _FakePreparedAdapter()
            predictions = _run_dataloader_inference_loop(
                items,
                adapter,
                adapter_dir=None,
                model_path=None,
                data_dir=root,
                batch_size=2,
                batch_save=10,
                num_workers=2,
                device="cpu",
            )
            self.assertEqual(len(predictions), 4)
            self.assertTrue(all(p == [0.1, 0.2, 0.3, 0.4] for p in predictions.values()))

    def test_legacy_adapter_falls_back_to_predict(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            items = _make_items(root, 3)
            adapter = _FakeLegacyAdapter()
            predictions = self._run(adapter, items, root)
            self.assertEqual(len(predictions), 3)
            self.assertTrue(all(p == [0.5, 0.5, 0.6, 0.6] for p in predictions.values()))

    def test_existing_predictions_are_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            items = _make_items(root, 4)
            adapter = _FakePreparedAdapter()
            predictions = self._run(
                adapter, items, root, existing_predictions={"seq_000_0001": [0, 0, 1, 1]}
            )
            self.assertEqual(predictions["seq_000_0001"], [0, 0, 1, 1])
            self.assertEqual([len(b) for b in adapter.prepared_batches], [2, 1])


if __name__ == "__main__":
    unittest.main()
