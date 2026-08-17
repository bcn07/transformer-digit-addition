# transformer-digit-addition

Reverse-engineering the carry circuit in a small transformer trained on decimal addition, extending Nanda et al.'s grokking work beyond modular arithmetic.

## Motivation

Nanda et al.'s *Progress measures for grokking via mechanistic interpretability* analysed **modular** arithmetic and found the model had learned a Fourier algorithm built out of trigonometric identities. Ordinary multi-digit addition is a different problem: it is positional, and it requires carrying. Whatever circuit appears should therefore be structurally different, and that difference is the point of the project.

## Method

1. Train a small transformer on 2-digit decimal addition, with the answer digits reversed so carries propagate in the same direction as generation.
2. Hold out 70% of the 10,000 possible pairs, which is the setup under which grokking appears.
3. Interpret the trained model with TransformerLens: attention patterns, per-head ablation, and direct logit attribution.

Architecture and training follow Nanda et al. where the task allows — `d_model=128`, 4 heads, `d_mlp=512`, no layer norm, full batch, AdamW at a constant `lr=1e-3` with `weight_decay=1.0`. The deliberate differences are 2 layers rather than 1, a 12-symbol vocabulary, and a 9-token context.

## Findings

_TODO — to be written up once training and interpretability analysis are complete._

The minimum result is identifying which attention heads implement the carry, via ablation and attention patterns.

## Repo structure

```
notebooks/    digit-addition training and interpretability
results/      loss curves, attention visualisations, other interpretability outputs
```

## Setup

Notebooks are stripped of outputs on commit via [nbstripout](https://github.com/kynan/nbstripout). After cloning:

```
nbstripout --install --attributes .gitattributes
```
