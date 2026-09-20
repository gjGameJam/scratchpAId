# scratchpAId

  A small PyTorch CNN that reads handwritten digits from MNIST, plus a visual audit tool for
  spotting the ones it gets wrong.

  Everything lives in `number_classifier.py`.

  ## The model

  `CNNModel` is two convolutional blocks feeding two fully connected layers:

  | Layer | Shape |
  | --- | --- |
  | `conv1` + `pool1` | 1 → 16 channels, 3×3, then 2×2 max pool |
  | `conv2` + `pool2` | 16 → 32 channels, 3×3, then 2×2 max pool |
  | `fc1` | 32×7×7 = 1568 → 128 |
  | `fc2` | 128 → 10 |

  Trained with cross-entropy and Adam (`lr=1e-3`) for 5 epochs at batch size 64.

  ## Results

  **99.07% on the full 10,000-image held-out test set**, reported by `evaluate_model` after the
  final epoch.

  Note that the accuracy in the audit figure's title is a different number. `audit_predictions`
  counts only the digits it displays, so with `max_images=100` its title scores those 100 cells
  alone — one miss reads as 99.00%, none as 100.00%. The model's accuracy is the `Test Acc` line
  that `train_model` prints each epoch.

  ## Running it

  ```bash
  pip install torch torchvision matplotlib tqdm
  python number_classifier.py
  ```
  The first run downloads MNIST into ./data. Training prints per-epoch train and test accuracy,
  then the audit window opens.

   ## The visual audit

  audit_predictions lays out a 10×10 grid of test digits, each titled P:<predicted> / A:<actual> with a green ✓ or a red
  ✗. Scanning for the red marks is the fastest way to see
  what the network actually confuses.
