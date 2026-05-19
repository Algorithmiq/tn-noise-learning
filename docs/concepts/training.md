# Training

Training reconstructs the noise LPDO by minimising a combined loss over the
trainable Kraus tensors:

$$
\mathcal{L}(\theta) = \mathrm{KL}\bigl(P_{\text{true}} \,\|\, P_{\Lambda(\theta)}\bigr) +
                       \lambda \,\bigl\|\mathrm{Tr}_\text{aux}[\Lambda(\theta)] - \mathbb{1}\bigr\|_F \,/\, 2^{n/2}
$$

with $\lambda = 1.2$ by default. The KL term enforces fidelity with the
empirical transition probabilities; the trace-preservation (TP) penalty
keeps the LPDO close to a physically valid channel.

## End-to-end driver

The recommended entry point is
[`train`][tn_noise_learning.training.pipeline.train], which mirrors the
sabrina ``noise_learning.py`` / ``noise_learning_spam.py`` scripts:

```python
from tn_noise_learning import train

metrics = train(
    data_file="…experiment.json",
    spam=False,                  # set True for SPAM datasets
    virtual_bd=4,
    kraus_bd=4,
    batch_size=250,
    epochs=150,
    lr="exponential_decay",
)
```

## Lower-level pieces

* [`losses.loss_fn`][tn_noise_learning.training.losses.loss_fn] —
  the combined KL + TP loss, batched over the `exp` hyperindex.
* [`losses.kl_divergence`][tn_noise_learning.training.losses.kl_divergence]
  and [`losses.tp_penalty`][tn_noise_learning.training.losses.tp_penalty] —
  individual loss components.
* [`optimizer.optimize_tn`][tn_noise_learning.training.optimizer.optimize_tn]
  — main SGD loop on the Kraus tensors, with an Adam-with-exponential-decay
  default schedule. Captures full trajectories of train/test loss, channel
  fidelity, and TP distance.
* [`optimizer.optimize_trace_preservation`][tn_noise_learning.training.optimizer.optimize_trace_preservation]
  — warm-start that pushes the initial random LPDO onto the TP manifold via
  L-BFGS-B.
* [`dataset.create_training_dataset`][tn_noise_learning.training.dataset.create_training_dataset]
  — convert raw QPT JSON counts into batched hyper-indexed tensor networks.
* [`dataset.evolve_hyper_tn`][tn_noise_learning.training.dataset.evolve_hyper_tn]
  — apply the ideal unitary to the input-state hyper-TN, so the loss only
  has to learn the noise residual.

## Persistence

[`io.save_results`][tn_noise_learning.io.results.save_results] serialises
the full training trajectory + final Kraus parameters to JSON. Complex
arrays are stringified on write and converted back to `complex` on load.
