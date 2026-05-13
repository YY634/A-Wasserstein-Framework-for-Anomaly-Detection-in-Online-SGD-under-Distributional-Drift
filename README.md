title: A Wasserstein Framework for Abnormal Batch Detection in Online SGD under Distributional Drift

**Submission to STAI-X 2026.**

This package contains a single Jupyter notebook that gather together all four experiments in the order they appear in the paper.

---

## Contents

| File | Description |
| --- | --- |
| `STAIX_2026_combined_experiments.ipynb` | Merged main notebook, containing all four experiments in order |
| `README.md` | This file |

---

## Notebook Structure

The merged notebook has **105 cells** in total (49 markdown + 56 code), organized into four sections:

| Section | Paper reference | Content |
| --- | --- | --- |
| **Experiment 1** | §4.1 Case 1 | A family of Uniform distribution vs. discrete Gaussian distribution. |
| **Experiment 2** | §4.1 Case 2 | 2D annulus family vs. Beta-radial omega-uniform distribution. |
| **Experiment 3** | §4.2 | Online linear regression with $d=5$, $m=60$, $T=120$ and $\theta$-jitter anomaly injections. Compares the WSD-based p-value against gradient norm, Mean-grad $L_2$, Hotelling $T^2$, and MMD (RBF). Reports ROC curve, AUC table. |
| **Experiment 4** | §5 | Streaming MNIST with drifting proportions interpolating two Dirichlet(0.4) end points; abnormal batch contains 10% mislabeled images. Detection signal is the logits gradient at an anchor $\theta_0$. AUC table, KS / chi-square calibration tests, and four baselines for comparison |

Each section starts with a clearly marked divider markdown cell (title + paper reference + short description).

---

## Environment

Mainly tested on Python 3.10+. Required third-party packages:

```text
numpy
scipy
scikit-learn
pandas
matplotlib
joblib
pot                 # this is the package; the import name is `ot` (Python Optimal Transport)
torch
torchvision         # only needed by Experiment 4, to load MNIST
```

One-line install:

```bash
pip install numpy scipy scikit-learn pandas matplotlib joblib pot torch torchvision
```

> Note: the import name `ot` is provided by the `pot` package. **Do not run `pip install ot`** — that installs an unrelated package.

---

## How to Run

1. Launch Jupyter:
   ```bash
   jupyter notebook STAIX_2026_combined_experiments.ipynb
   # or
   jupyter lab STAIX_2026_combined_experiments.ipynb
   ```
2. Recommended: run **section by section** from top to bottom (Run All also works). Each experiment re-imports its own libraries, redefines its own config, and defines its own helpers, so the four sections are independent.
3. Experiment 4 will auto-download MNIST on first run via `torchvision.datasets.MNIST(..., download=True)`. If the machine has no internet access, please pre-download MNIST to the default cache location.

### Approximate running time (CPU reference)

| Experiment | Single-seed time | Multi-seed time |
| --- | --- | --- |
| Experiment 1 | seconds to tens of seconds (longer when the 50-replication MC block runs) | — |
| Experiment 2 | tens of seconds | ~ a few minutes for 20 reps |
| Experiment 3 | tens of seconds to a few minutes | ~ 10 minutes for 20 seeds |
| Experiment 4 | a few minutes (online CNN training + WSD scoring) | ~ 15–30 minutes for 5 seeds |

Times are machine-dependent. For a fast smoke test, reduce `N_REPS` / `N_SEEDS`.

---

## Reproducibility Notes

* **Seeds**. Experiments 1 / 2 default to `SEED=42`. Experiments 3 / 4 use seeds `0..N_SEEDS-1` to reproduce the mean ± std numbers reported in the paper.
* **OT solver choice**:
  - Experiments 1 / 2 use Sinkhorn regularization (`reg = 0.05`, `max_iter = 5000`).
  - Experiment 3 uses **exact EMD** (POT's network simplex solver), because at $m=60$, $d=5$ the exact solver is still fast and avoids the extra approximation error of Sinkhorn.
  - Experiment 4 uses **adaptive Sinkhorn**: $\varepsilon = 0.01 \cdot \text{median}(C)$, which is more robust across gradient streams of different dimensions (see paper Appendix D.4).
* **Anchor $\theta_0$ (Experiment 4 only)**. After burn-in, a snapshot of the classifier is frozen as an anchor. All WSD detection signals are then taken as logits gradients at this anchor (see Appendix D.2).
* **Reference window**. Experiments 3 / 4 both use a sliding window as the reference collection. Accepted batches in the window are used to compute leave-one-out depths, which then yield the empirical p-values.
* **Calibration caveat**. Under the configuration reported in Appendix D.6, the p-values on regular batches are approximately Uniform[0,1], but the KS calibration test rejects at the 5% level in 3 out of 5 seeds. This caused by the strong drift concentration $c=0.4$, which introduces non-negligible time-to-time shifts even in the regular batches. This does not hurt discriminative power (AUC ≥ 0.913 in all 5 seeds).

---

## Citation

This is an anonymous double-blind submission, so a complete citation is not provided. A placeholder is given below; the camera-ready version will be updated upon acceptance.

```bibtex
@inproceedings{anonymous2026wsd,
  title     = {A Wasserstein Framework for Abnormal Batch Detection in Online SGD under Distributional Drift},
  author    = {Anonymous Author(s)},
  booktitle = {STAI-X 2026 (under review)},
  year      = {2026},
  note      = {Double-blind submission}
}
```

The underlying Wasserstein Spatial Depth method, cited as prior work in the paper, is:

```bibtex
@article{bachoc2024wsd,
  title   = {Wasserstein spatial depth},
  author  = {Bachoc, Fran\c{c}ois and Gonz\'alez-Sanz, Alberto and Loubes, Jean-Michel and Yao, Yisha},
  journal = {arXiv preprint arXiv:2411.10646},
  year    = {2024}
}
```
