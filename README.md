# transformer-digit-addition

Reverse-engineering the carry circuit in a small transformer trained on decimal addition, extending Nanda et al.'s grokking work beyond modular arithmetic.

## Motivation

Nanda et al.'s *Progress measures for grokking via mechanistic interpretability* analysed **modular** arithmetic and found the model had learned a Fourier algorithm built out of trigonometric identities. Ordinary multi-digit addition is a different problem: it is positional, and it requires carrying. Whatever circuit appears should therefore be structurally different, and that difference is the point of the project.

## Method

A 1-layer transformer on 2-digit decimal addition, with the answer digits reversed (`49+97=641`) so carries propagate in the same direction the causal mask allows information to flow. Architecture matches Nanda et al. exactly — `d_model=128`, 4 heads, `d_mlp=512`, no layer norm, full batch, AdamW at a constant `lr=1e-3` — so the task is the only variable between his setup and this one. The remaining differences are a 12-symbol vocabulary and a 9-token context, both forced by the task.

The fraction of the 10,000 possible pairs used for training is swept from 0.005 to 0.03, three seeds each, 100,000 steps per run. The sweep ran in parallel on Imperial DoC's Condor pool; `train.py` reproduces any single run from the command line.

Interpretability uses TransformerLens: attention patterns, per-head ablation at `hook_z`, and the same ablation repeated across 50 checkpoints saved through training.

## Findings

**Decimal addition does not grok under Nanda's protocol.** At his training fraction the model solves the task by step 150, with train and test loss indistinguishable throughout. The reason is that addition decomposes: the entire function is 200 `(digit, digit, carry_in)` table entries, and 3,000 training examples cover each of them about thirty times. Generalising is cheaper than memorising, so the model never memorises first and there is no delayed transition to observe.

**Below a threshold, the model generalises beyond what it could have memorised.** Sweeping the training fraction down until sub-problems become sparse produces a sharp threshold, with a critical point at `frac_train=0.01` where three seeds land at 0.43, 0.53 and 0.98. Eleven of fifteen runs exceed the accuracy achievable by learning only the table entries their training set exposed. Two reached **exactly 1.000 on all 9,700 held-out sums**, and at `frac_train=0.015` a model reached 0.980 having seen only 139 of the 200 entries, so these models are computing rather than looking up. The generalisation is delayed — flat for ~21,000 steps after memorisation — but the subsequent climb takes 25,000 to 67,000 steps, so it is not the abrupt transition grokking describes.

**Weight decay is what causes generalisation in that regime.** Two runs, same seed, same data, differing only in `weight_decay`:

| | memorised | test accuracy at 10k / 30k / 60k / 99k | peak |
|---|---|---|---|
| `wd = 0.0` | step 200 | 0.081, 0.082, 0.079, 0.069 | 0.089 |
| `wd = 1.0` | step 200 | 0.474, 0.930, 0.968, 0.957 | **0.975** |

Both memorise immediately. Without weight decay the model then does nothing for 100,000 steps. This inverts the conclusion from the high-data regime, where weight decay looked like pure downside because it destabilised training with no benefit.

**Attention routes, the MLP computes, and no head owns the carry.** Attention patterns are input-independent: the heads that fetch the operand digits do so identically on carry and non-carry inputs, because the tens output always needs both digit pairs. Ablating the MLP destroys every answer position; ablating all attention leaves the leading digit largely intact. Ablating either routing head takes both the units and tens digits to chance, so the carry is not separable from the rest of the addition.

**The circuit is not stable, within a run or across seeds.** Test accuracy sits between 0.92 and 0.98 from step 24,000 onwards while the set of load-bearing heads keeps changing. All seven reorganisations after the accuracy plateau occur within 2,500 steps of a training loss spike. Across seeds, every model uses three of its four heads and leaves one redundant, but which head is redundant varies. Any claim of the form "head *n* does *x*" is a claim about one checkpoint of one run.

The contrast with Kruthoff's [Carrying over algorithm in transformers](https://arxiv.org/abs/2401.07993) is that his two-layer encoder-only models split the algorithm cleanly, with layer 1 adding same-position digits and layer 2 deciding and applying the carry. A one-layer decoder-only model compresses both into a single attention-plus-MLP block, and the compressed version has no stable per-head assignment.

### Figures

| | |
|---|---|
| `results/sweep_frac_train.png` | generalisation against the memorise-only ceiling, by training fraction |
| `results/circuit_formation.png` | per-head ablation damage across 50 checkpoints |
| `results/spike_alignment.png` | head swaps against training loss spikes |
| `results/weight_decay_comparison.png` | the weight decay instability |

## Repo structure

```
train.py                        CLI entry point; reproduces any single run
notebooks/digit_addition.ipynb  task, training setup, frac_train sweep
notebooks/carry_circuit.ipynb   interpretability, from a saved checkpoint
results/checkpoints/            trained models (*_best.pt)
results/sweep/                  per-run training histories
results/                        analysis outputs and figures
```

## Setup

Notebooks are stripped of outputs on commit via [nbstripout](https://github.com/kynan/nbstripout). After cloning:

```
nbstripout --install --attributes .gitattributes
```
