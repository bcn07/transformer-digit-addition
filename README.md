# transformer-digit-addition
From-scratch transformer trained on digit addition, interpreted with TransformerLens — extending Nanda et al.'s grokking research beyond modular arithmetic.

## Motivation

Nanda et al.'s grokking work analyzed modular arithmetic. This project retrains the same from-scratch transformer architecture on digit addition instead — a fresher task than modular arithmetic, chosen to produce an original interpretability angle rather than reproduce existing results.

## Method

1. Build a transformer from scratch (`notebooks/1_transformer_from_scratch.ipynb`), following ARENA's Ch1.1 exercises.
2. Retrain it on a digit-addition task.
3. Interpret the trained model with TransformerLens.

## Findings

_TODO — to be written up once training and interpretability analysis are complete._

## Repo structure

```
notebooks/    ARENA transformer-from-scratch exercises, digit-addition training + interp
results/      loss curves, attention visualizations, other interpretability outputs
```
