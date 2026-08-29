# transformer-digit-addition

Ablation and attention analysis of a 1-layer transformer trained on 2-digit decimal addition, testing Nanda et al.'s grokking protocol outside modular arithmetic.

## Motivation

Nanda et al.'s *Progress measures for grokking via mechanistic interpretability* analysed **modular** arithmetic and found the model had learned a Fourier algorithm built out of trigonometric identities. Ordinary multi-digit addition is a different problem: it is positional, and it requires carrying, so the Fourier construction has nothing to latch onto. The question this repo starts from is whether the *protocol* transfers — whether a task that decomposes into a small table groks at all — and it turns out not to, which is what makes the low-data regime the interesting place to look.

## Method

A 1-layer transformer on 2-digit decimal addition, with the answer digits reversed (`49+97=641`) so carries propagate in the same direction the causal mask allows information to flow. Architecture follows Nanda et al. — `d_model=128`, 4 heads, `d_mlp=512`, no layer norm, full batch, AdamW at a constant `lr=1e-3` — so the task is as close to the only variable as the setup allows. The differences are a 12-symbol vocabulary and a 9-token context, both forced by the task, and AdamW's default `betas=(0.9, 0.999)` against his `(0.9, 0.98)`.

The fraction of the 10,000 possible pairs used for training is swept from 0.005 to 0.03, three seeds each, 100,000 steps per run. The sweep ran in parallel on Imperial DoC's Condor pool; `train.py` reproduces any single run from the command line.

Interpretability uses TransformerLens: attention patterns averaged over the full test set and split by carry, per-head ablation at `hook_z` under both zero- and mean-ablation, and the same ablation repeated across 50 checkpoints saved through training.

## Findings

**Decimal addition does not grok under Nanda's protocol.** Under his exact configuration — `frac_train=0.3`, `weight_decay=1.0` — the model reaches 0.99 per-digit test accuracy at step 150 and stays at 1.000 for the rest of the run. Train and test whole-sum accuracy cross 0.99 ten steps apart, at 170 and 180, and the largest train–test loss gap over 5,000 steps is 0.089. There is no window in which the model has memorised and not yet generalised. The reason is that addition decomposes: the entire function is 200 `(digit, digit, carry_in)` table entries, and 3,000 training examples cover each of them about thirty times. Generalising is cheaper than memorising, so the model never memorises first and there is no delayed transition to observe.

**Below a threshold, the model generalises beyond what it could have memorised.** Sweeping the training fraction down until sub-problems become sparse produces a sharp threshold, with a critical point at `frac_train=0.01` where three seeds land at 0.43, 0.53 and 0.98. Eleven of fifteen runs exceed the accuracy achievable by learning only the table entries their training set exposed. Two reached **exactly 1.000 on all 9,700 held-out sums**, and at `frac_train=0.015` a model reached 0.980 having seen only 136 of the 200 entries, so these models are computing rather than looking up. The generalisation is delayed but not abrupt, which is the opposite of the grokking signature: most runs jump within 500 steps to the accuracy their memorised table allows, then grind upward from there. Eight of the nine runs at `frac_train` ≤ 0.02 take 22,900 to 93,800 steps to reach 90% of their peak; the ninth takes 2,300. Only one run (`frac_train=0.01`, seed 1) sits genuinely flat near chance first, for 20,000 steps. At `frac_train=0.03` the climb is over inside 1,100 steps.

**Weight decay is what drives generalisation in that regime.** Three pairs of runs at `frac_train=0.015`, each pair sharing a seed and therefore a data split, differing only in `weight_decay`:

| seed | memorised | `wd=0` peak | `wd=1` peak | `wd=0` at 99k | `wd=1` at 99k | that split's ceiling |
|---|---|---|---|---|---|---|
| 0 | step 200 | 0.320 | **0.745** | 0.194 | 0.727 | 0.640 |
| 1 | step 200 | 0.218 | **0.980** | 0.216 | 0.980 | 0.650 |
| 2 | step 200 | 0.089 | **0.975** | 0.069 | 0.957 | 0.716 |

All six memorise by step 200, and the two ranges do not overlap: the worst `wd=1` run beats the best `wd=0` run by more than a factor of two. Every `wd=1` run also clears its memorise-only ceiling, and no `wd=0` run comes close to its own — so without decay the model does not merely fail to generalise, it fails even to consolidate the table entries its training set showed it. This inverts the conclusion from the high-data regime, where weight decay looked like pure downside because it destabilised training with no benefit.

**Attention routes the carry positionally, and the MLP computes it.** Two heads do the routing. For each answer digit, one fetches the column being added and the other fetches the column whose overflow becomes that digit's carry — at the tens digit they point at the tens and units operands respectively, and at the units digit, which has no carry in, they coincide. Neither is input-dependent: across 9,850 test sums the largest carry versus non-carry difference in either head's attention is 0.007 on a budget of 1.0. They fetch the units column unconditionally, because you cannot know whether a carry happened until you have looked, so the conditional part must happen downstream. Ablating the MLP destroys every answer position.

Mean-ablating the second routing head leaves the units digit at 0.988 and takes the tens and hundreds to 0.732 and 0.636 — damage confined to exactly the positions that consume a carry. Zero-ablation hides this completely — 0.130 / 0.127 / 0.635, i.e. everything breaks — because zeroing a head hands the MLP an input it never saw in training, so it fails for reasons unrelated to what the head encodes. I ran only zero-ablation first and wrote the carry-head prediction up as falsified. The prediction was fine; the method was wrong.

**The circuit is not stable, within a run or across seeds.** Test accuracy reaches 0.92 by step 24,000 and ends at 0.948, but it does not hold in between: it collapses to 0.385 at step 58,200 and recovers. Across those 75,000 steps the set of load-bearing heads keeps changing. The seven head flips after step 30,000 happen at four distinct steps, and all four fall within 2,500 steps of a training loss spike. Against a 26% base rate that is p = 0.004; scoring the same analysis on mean-ablation damage gives five events, four at a spike, p = 0.017. Across seeds, the division of labour is constant and the assignment is not: every converged model has one head fetching the digits being added and another fetching the digits that generate the carry, but which head takes which job changes with the seed. A claim of the form "head *n* does *x*" is about one checkpoint of one run; "some head does *x*" survives.

### Against prior work

Kruthoff's [Carrying over algorithm in transformers](https://arxiv.org/abs/2401.07993) finds a clean split in two-layer encoder-only models: layer 1 adds same-position digits, layer 2 decides and applies the carry. A one-layer decoder-only model has to compress both into a single attention-plus-MLP block.

The one-layer model here does compress that split, into two attention heads plus the MLP rather than two layers. But that is a replication, not a novelty: Quirke and Barez's [Understanding addition in transformers](https://arxiv.org/abs/2310.13121) already reverse-engineers a *one-layer* model on n-digit addition into per-digit sub-tasks with identified per-head roles, and Lee et al.'s [Teaching arithmetic to small transformers](https://arxiv.org/abs/2307.03381) already establishes that reversing the output digits turns addition into a simple function of two digits and a carry bit, with sharp phase transitions against data volume. Both work in the high-data regime. Finding the same structure here at 150 training examples, in a different architecture and output order, is corroboration worth having but not new.

What is left over, and what this repo is actually about, is the low-data end: where the transition to computing rather than looking up sits, and the fact that a converged circuit keeps reorganising afterwards while its behaviour holds. The loss spikes driving those reorganisations are the Slingshot Mechanism of [Thilak et al.](https://arxiv.org/abs/2206.04817), which reports the same Adam-driven spikes at near-zero loss without weight decay — matching the `wd=0` control here. What that paper does not report is that the circuit is different on the other side of each spike.

### Figures

| | |
|---|---|
| `results/sweep_frac_train.png` | generalisation against each run's own memorise-only ceiling |
| `results/circuit_formation.png` | per-head ablation damage across 50 checkpoints, zero and mean |
| `results/spike_alignment.png` | head swaps against training loss spikes |
| `results/runs_L1.png` | the single runs, including the `frac_train=0.3` control |
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
