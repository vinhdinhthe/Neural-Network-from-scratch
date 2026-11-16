"""Save a training report, numeric history, and train/test learning curves."""

import csv
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


class TrainingHistory:
    """Write settings before training; append metrics after each completed epoch."""

    columns = ["epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy"]

    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.rows = []
        self.loss_name = config["criterion"]
        self.header = (
            f"epochs = {config['epochs']}\n"
            f"batch_size = {config['batch_size']}\n"
            f"learning_rate = {config['learning_rate']}\n\n"
            f"model = NeuralNetwork(seed={config['seed']}, widths={config['widths']}, "
            f"activations={config['activations']})\n\n"
            f"criterion = {config['criterion']}()\n"
        )
        details = "\n".join(f"{key}: {value}" for key, value in config.items()
                            if key not in {"epochs", "batch_size", "learning_rate", "widths", "activations", "criterion"})
        (self.directory / "session.txt").write_text(
            self.header + "\n" + details + "\n\nTraining history\n", encoding="utf-8"
        )
        with (self.directory / "history.csv").open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(self.columns)

    def record(self, epoch, train_loss, train_accuracy, test_loss, test_accuracy):
        row = [epoch, train_loss, train_accuracy, test_loss, test_accuracy]
        self.rows.append(row)
        with (self.directory / "history.csv").open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(row)
        line = (
            f"Epoch {epoch} | Train loss: {train_loss:.6f} | Train accuracy: {train_accuracy:.2%} | "
            f"Test loss: {test_loss:.6f} | Test accuracy: {test_accuracy:.2%}"
        )
        with (self.directory / "session.txt").open("a", encoding="utf-8") as file:
            file.write(line + "\n")
        print(line, flush=True)

    def finish(self, status, elapsed_seconds):
        with (self.directory / "session.txt").open("a", encoding="utf-8") as file:
            file.write(f"\nStatus: {status}\nCompleted epochs: {len(self.rows)}\nElapsed seconds: {elapsed_seconds:.2f}\n")
        if not self.rows:
            return
        epochs = [row[0] for row in self.rows]
        for filename, title, ylabel, train_column, test_column, scale in [
            ("accuracy.png", "Training and test accuracy", "Accuracy (%)", 2, 4, 100),
            ("loss.png", "Training and test loss", self.loss_name, 1, 3, 1),
        ]:
            # Agg renders without opening windows and works on headless machines.
            figure = Figure(figsize=(9, 5), layout="constrained")
            FigureCanvasAgg(figure)
            axis = figure.subplots()
            axis.plot(epochs, [row[train_column] * scale for row in self.rows], label="Train", color="#2563eb", marker="." )
            axis.plot(epochs, [row[test_column] * scale for row in self.rows], label="Test", color="#e87924", marker=".")
            axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
            if filename == "accuracy.png":
                axis.set_ylim(0, 100)
            axis.grid(alpha=0.25)
            axis.legend()
            figure.savefig(self.directory / filename, dpi=160)
