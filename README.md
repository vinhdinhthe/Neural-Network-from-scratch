# Neural Network from Scratch: MNIST

A fully connected neural network implemented with NumPy: forward propagation,
manual backpropagation, and mini-batch stochastic gradient descent. Matplotlib
is used only for plotting. Training does not use PyTorch or automatic differentiation.

## Setup and training

Use Python 3.10 or newer:

```sh
python -m pip install -r requirements.txt
python train.py --data-dir /path/to/MNIST/raw
```

Supply these four **uncompressed** MNIST IDX files in the data directory:

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

The default data directory is `data/MNIST/raw` inside this project. The dataset
is not bundled. On Windows, quote paths containing spaces. `python src/train.py`
and `python -m src.train` are also supported entry points.

Defaults: 100 epochs, batch size 128, learning rate 0.0015, seed 6, widths
`784 -> 256 -> 128 -> 64 -> 32 -> 10`, ReLU hidden layers, softmax output,
and cross-entropy loss. Edit the architecture and criterion in `src/train.py`.

```sh
python train.py --data-dir /path/to/MNIST/raw --epochs 5 --batch-size 128 --learning-rate 0.0015 --seed 6
```

Use `--results-dir` to choose an output directory. Importing the training module
does not load data or start training.

## Saved results

Each run creates a separate `results/run_<timestamp>/` directory containing:

- `session.txt`: settings first, per-epoch metrics, completion status, and runtime.
- `history.csv`: epoch, train loss/accuracy, and test loss/accuracy (accuracy in 0-1 units).
- `accuracy.png`: training and test accuracy curves in percent.
- `loss.png`: training and test loss curves.

Both sets are evaluated in full **after** each epoch, using the same model state.
Loss is weighted by sample count, including the last partial batch. This extra
training-set evaluation takes additional time. Metrics are written each epoch;
plots are saved when training finishes (or on Ctrl+C for completed epochs).
No model weights are saved. Existing results are not overwritten.

Test curves are for observing these experiments. Use a separate validation split
for hyperparameter tuning and reserve the test set for final assessment.
`results/Train1.txt` is an existing manually recorded experiment; it does not
contain the new training-set metrics.

## Project layout

- `src/Network.py`: dense layers, forward cache, backpropagation, and SGD updates.
- `src/Activation.py`: sigmoid, ReLU, and softmax with backward operations.
- `src/Losses.py`: mean squared error and cross-entropy.
- `src/Data.py`: MNIST loading, one-hot labels, and shuffled batches.
- `src/train.py`: configuration, training, and evaluation.
- `src/Visualisation.py`: session reports, CSV history, and plots.
- `docs/`: space for handwritten derivations and project images (to be added).
