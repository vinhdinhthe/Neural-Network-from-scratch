"""Train the NumPy MNIST network and save session metrics and plots."""

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import time

import numpy as np

if __package__:
    from .Network import NeuralNetwork
    from .Losses import CrossEntropyLoss
    from .Data import load_mnist, one_hot, get_mini_batches
    from .Visualisation import TrainingHistory
else:
    from Network import NeuralNetwork
    from Losses import CrossEntropyLoss
    from Data import load_mnist, one_hot, get_mini_batches
    from Visualisation import TrainingHistory

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate(model, images, labels, criterion, batch_size=512):
    """Evaluate integer labels; average loss by sample count, including last batch."""
    correct = 0
    total_loss = 0.0
    for start in range(0, len(images), batch_size):
        x = images[start:start + batch_size]
        y = labels[start:start + batch_size]
        predictions = model.forward(x)
        total_loss += float(criterion(predictions, one_hot(y, predictions.shape[1]))) * len(y)
        correct += np.count_nonzero(predictions.argmax(axis=1) == y)
    return total_loss / len(images), correct / len(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "MNIST" / "raw")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0015)
    parser.add_argument("--seed", type=int, default=6)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("epochs, batch size, and learning rate must be positive and finite")
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("seed must be between 0 and 2**32 - 1")

    # Edit architecture and loss here. CLI arguments control training settings.
    widths = [784, 256, 128, 64, 32, 10]
    activations = ["ReLU", "ReLU", "ReLU", "ReLU", "Softmax"]
    model = NeuralNetwork(seed=args.seed, widths=widths, activations=activations)
    criterion = CrossEntropyLoss()
    np.random.seed(args.seed)

    try:
        x_train, y_train, x_test, y_test = load_mnist(args.data_dir)
    except FileNotFoundError as error:
        parser.error(f"{error}. Supply --data-dir with the four uncompressed MNIST IDX files.")
    x_train = x_train.reshape(len(x_train), -1).astype(np.float32) / 255.0
    x_test = x_test.reshape(len(x_test), -1).astype(np.float32) / 255.0
    targets = one_hot(y_train)

    started = datetime.now(timezone(timedelta(hours=7)))
    config = {
        "started_at": started.isoformat(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "widths": widths,
        "activations": activations,
        "criterion": type(criterion).__name__,
        "seed": args.seed,
        "optimizer": "SGD (no momentum)",
        "initialization": "Normal(0, 1) * sqrt(2 / (fan_in + fan_out)); zero biases",
        "preprocessing": "Flatten 28x28 images; float32 pixels divided by 255",
        "parameter_dtype": str(model.weights[0].dtype),
        "target_dtype": str(targets.dtype),
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "numpy_version": np.__version__,
        "metrics": "Full train/test sets evaluated after each epoch; accuracy in [0, 1]",
    }
    run_dir = args.results_dir / started.strftime("run_%Y%m%d_%H%M%S_%f")
    history = TrainingHistory(run_dir, config)
    print(history.header, flush=True)
    print(f"Saving results to: {run_dir}", flush=True)
    start_time = time.perf_counter()
    try:
        for epoch in range(1, args.epochs + 1):
            for x_batch, y_batch in get_mini_batches(x_train, targets, args.batch_size):
                predictions = model.forward(x_batch)
                model.backward(criterion.grad(predictions, y_batch))
                model.step(args.learning_rate)

            # Evaluation overwrites the forward cache: only do it AFTER updates.
            train_loss, train_acc = evaluate(model, x_train, y_train, criterion)
            test_loss, test_acc = evaluate(model, x_test, y_test, criterion)
            history.record(epoch, train_loss, train_acc, test_loss, test_acc)
    except KeyboardInterrupt:
        history.finish("interrupted", time.perf_counter() - start_time)
        print("Interrupted; completed epochs have been saved.")
        return
    except Exception:
        history.finish("failed", time.perf_counter() - start_time)
        raise
    history.finish("completed", time.perf_counter() - start_time)
    print(f"Report, history, and plots saved to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
