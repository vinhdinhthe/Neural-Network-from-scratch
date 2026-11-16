"""Checks for epoch evaluation and saved run artifacts."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.Losses import MSELoss
from src.train import evaluate
from src.Visualisation import TrainingHistory


class TrainingResultsTests(unittest.TestCase):
    def test_evaluation_weights_partial_batch(self):
        class FixedPredictions:
            def forward(self, images):
                return images

        predictions = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])
        labels = np.array([0, 1, 1])
        loss, accuracy = evaluate(FixedPredictions(), predictions, labels, MSELoss(), batch_size=2)
        self.assertAlmostEqual(loss, np.mean((predictions - np.eye(2)[labels]) ** 2))
        self.assertAlmostEqual(accuracy, 2 / 3)

    def test_report_history_and_plots(self):
        config = dict(epochs=2, batch_size=4, learning_rate=0.01, seed=6,
                      widths=[2, 2], activations=["Softmax"], criterion="CrossEntropyLoss")
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            history = TrainingHistory(run, config)
            self.assertIn("epochs = 2", (run / "session.txt").read_text())
            history.record(1, 0.8, 0.6, 0.9, 0.5)
            history.record(2, 0.5, 0.8, 0.7, 0.7)
            history.finish("completed", 1.0)
            with (run / "history.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[-1]["test_accuracy"]), 0.7)
            self.assertIn("Status: completed", (run / "session.txt").read_text())
            for name in ["accuracy.png", "loss.png"]:
                self.assertTrue((run / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            with self.assertRaises(FileExistsError):
                TrainingHistory(run, config)


if __name__ == "__main__":
    unittest.main()
