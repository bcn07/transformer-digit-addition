# transformer-digit-addition

Ablation and attention analysis of a 1-layer transformer trained on 2-digit decimal addition, testing Nanda et al.'s grokking protocol outside modular arithmetic.

## Motivation

Nanda et al.'s *Progress measures for grokking via mechanistic interpretability* analysed **modular** arithmetic and found the model had learned a Fourier algorithm built out of trigonometric identities. Ordinary multi-digit addition is a different problem: it is positional, and it requires carrying, so the Fourier construction has nothing to latch onto. The question this repo starts from is whether the *protocol* transfers — whether a task that decomposes into a small table groks at all — and it turns out not to, which is what makes the low-data regime the interesting place to look.

## Method

A 1-layer transformer on 2-digit decimal addition, with the answer digits reversed (`49+97=641`) so carries propagate in the same direction the causal mask allows information to flow. Architecture follows Nanda et al. — `d_model=128`, 4 heads, `d_mlp=512`, no layer norm, full batch, AdamW at a constant `lr=1e-3` — so the task is as close to the only variable as the setup allows. The differences are a 12-symbol vocabulary and a 9-token context, both forced by the task, and AdamW's default `betas=(0.9, 0.999)` against his `(0.9, 0.98)`.

The fraction of the 10,000 possible pairs used for training is swept from 0.005 to 0.03, three seeds each, 100,000 steps per run. The sweep ran in parallel on Imperial DoC's Condor pool; `train.py` reproduces any single run from the command line.

Interpretability uses TransformerLens: attention patterns, per-head **zero**-ablation at `hook_z`, and the same ablation repeated across 50 checkpoints saved through training.

## Findings

**Decimal addition does not grok under Nanda's protocol.** At his training fraction the model solves the task by step 150, with train and test loss indistinguishable throughout. The reason is that addition decomposes: the entire function is 200 `(digit, digit, carry_in)` table entries, and 3,000 training examples cover each of them about thirty times. Generalising is cheaper than memorising, so the model never memorises first and there is no delayed transition to observe.

**Below a threshold, the model generalises beyond what it could have memorised.** Sweeping the training fraction down until sub-problems become sparse produces a sharp threshold, with a critical point at `frac_train=0.01` where three seeds land at 0.43, 0.53 and 0.98. Eleven of fifteen runs exceed the accuracy achievable by learning only the table entries their training set exposed. Two reached **exactly 1.000 on all 9,700 held-out sums**, and at `frac_train=0.015` a model reached 0.980 having seen only 136 of the 200 entries, so these models are computing rather than looking up. The generalisation is delayed but not abrupt, which is the opposite of the grokking signature: most runs jump within 500 steps to the accuracy their memorised table allows, then grind upward from there, taking 23,000 to 94,000 steps to reach 90% of their peak. Only one run (`frac_train=0.01`, seed 1) sits genuinely flat near chance first, for 20,000 steps. At `frac_train=0.03` the climb is over inside 1,500 steps.

**Weight decay is what drives generalisation in that regime.** One pair of runs, same seed, same data, differing only in `weight_decay` — a single pair, so read this as the mechanism in this run rather than an established general claim:

| | memorised | test accuracy at 10k / 30k / 60k / 99k | peak |
|---|---|---|---|
| `wd = 0.0` | step 200 | 0.081, 0.082, 0.079, 0.069 | 0.089 |
| `wd = 1.0` | step 200 | 0.474, 0.930, 0.968, 0.957 | **0.975** |

Both memorise immediately. Without weight decay the model then does nothing for 100,000 steps. This inverts the conclusion from the high-data regime, where weight decay looked like pure downside because it destabilised training with no benefit.

**Attention routes, the MLP computes, and no head owns the carry.** Attention patterns are input-independent: the heads that fetch the operand digits do so identically on carry and non-carry inputs, because the tens output always needs both digit pairs. Ablating the MLP destroys every answer position; ablating all attention leaves the leading digit largely intact. Ablating either routing head takes both the units and tens digits to chance, so the carry is not separable from the rest of the addition.

**The circuit is not stable, within a run or across seeds.** Test accuracy reaches 0.92 by step 24,000 and ends at 0.948, but it does not hold in between: it collapses to 0.385 at step 58,200 and recovers. Across those 75,000 steps the set of load-bearing heads keeps changing. The seven head flips after step 30,000 happen at four distinct steps, and all four fall within 2,500 steps of a training loss spike — against a 26% chance rate, so p = 0.004. Across seeds the four heads are never equally loaded, but the least load-bearing one is genuinely free in only two of the four models; in the other two it still costs 0.16 and 0.19 accuracy. Which head it is varies. Any claim of the form "head *n* does *x*" is a claim about one checkpoint of one run.

### Against prior work

Kruthoff's [Carrying over algorithm in transformers](https://arxiv.org/abs/2401.07993) finds a clean split in two-layer encoder-only models: layer 1 adds same-position digits, layer 2 decides and applies the carry. A one-layer decoder-only model has to compress both into a single attention-plus-MLP block.

That is not the novelty, though. Quirke and Barez's [Understanding addition in transformers](https://arxiv.org/abs/2310.13121) already reverse-engineers a *one-layer* model on n-digit addition into per-digit sub-tasks with identified per-head roles, and Lee et al.'s [Teaching arithmetic to small transformers](https://arxiv.org/abs/2307.03381) already establishes that reversing the output digits turns addition into a simple function of two digits and a carry bit, with sharp phase transitions against data volume. Both work in the high-data regime.

What is left over, and what this repo is actually about, is the low-data end: where the transition to computing rather than looking up sits, and the fact that a converged circuit keeps reorganising afterwards. The loss spikes driving those reorganisations are the Slingshot Mechanism of [Thilak et al.](https://arxiv.org/abs/2206.04817), which reports the same Adam-driven spikes at near-zero loss without weight decay — matching the `wd=0` control here. What that paper does not report is that the circuit is different on the other side of each spike.

### Figures

| | |
|---|---|
| `results/sweep_frac_train.png` | generalisation against the memorise-only ceiling, by training fraction |
| `results/circuit_formation.png` | per-head ablation damage across 50 checkpoints |
| `results/spike_alignment.png` | head swaps against training loss spikes |
| `results/weight_decay_comparison.png` | the weight decay instability, on the earlier **2-layer** runs |

`results/README.md` maps every file to the run or command that produced it.

## Repo structure

```
train.py                        CLI entry point; reproduces any single run
notebooks/digit_addition.ipynb  task, training setup, frac_train sweep
notebooks/carry_circuit.ipynb   interpretability, from a saved checkpoint
results/checkpoints/            trained models (*_best.pt)
results/sweep/                  per-run training histories
results/                        analysis outputs and figures, indexed in results/README.md
```

## Setup

Python 3.13. `transformer-lens` is the pin that matters: 1.x and 2.x disagree on the shape of the
derived attention buffers, and the checkpoint loader works around that.

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install --attributes .gitattributes   # outputs are stripped on commit
```

## Reproducing

One sweep run, about 25 minutes on a CPU core:

```
python train.py --frac-train 0.015 --weight-decay 1.0 --seed 2 --steps 100000 \
    --eval-every 100 --out-dir results/sweep
```

The analysed checkpoint, adding the 50 per-step snapshots that `carry_circuit.ipynb` sweeps
(gitignored, ~40 MB; the notebook falls back to the committed `results/circuit_formation.json`
when they are absent):

```
python train.py --frac-train 0.015 --weight-decay 1.0 --seed 2 --steps 100000 \
    --eval-every 100 --out-dir results/sweep \
    --ckpt-dir results/checkpoints --ckpt-every 2000
```

Train loss reaches ~1e-7 in these runs, which is where the Adam instability lives, so the spike
schedule is sensitive to floating-point detail and will not reproduce step-for-step on different
hardware.
