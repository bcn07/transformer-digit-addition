# results/

Everything here is committed output. Nothing in this directory is generated at import time.

```
sweep/                  the frac_train sweep: 5 fractions x 3 seeds, 100k steps each.
                        This is the real dataset behind the generalisation results.
history_L1_*_s42.json   single runs driven from notebooks/digit_addition.ipynb
checkpoints/*_best.pt   best-test-accuracy weights for the four analysed models
circuit_formation.json  per-head ablation damage at 50 checkpoints through one run
universality.json       per-head ablation damage for the four best checkpoints
legacy/                 2-layer runs from before the architecture was matched to
                        Nanda et al. Kept only because weight_decay_comparison.png
                        was plotted from them; not used by any current result.
```

Naming is `history_L{n_layers}_wd{weight_decay}_ft{frac_train}_s{seed}.json`, matching
`TrainArgs.run_name` in `train.py`, so any file maps back to the command that made it.

Per-step checkpoints (`*_step*.pt`, 50 files, ~40 MB) are gitignored. `circuit_formation.json`
is the analysis output computed from them, and `notebooks/carry_circuit.ipynb` falls back to it
when they are absent. Regenerate them with the command in that notebook.

## Figures

| file | from |
|---|---|
| `sweep_frac_train.png` | `sweep/` — generalisation against the memorise-only ceiling |
| `circuit_formation.png` | `circuit_formation.json` — ablation damage across training |
| `spike_alignment.png` | `circuit_formation.json` + `sweep/history_L1_wd1.0_ft0.015_s2.json` |
| `runs_L1.png` | the `history_L1_*_s42.json` runs |
| `weight_decay_comparison.png` | `legacy/` — **2-layer** runs, `wd=0.1` against `wd=1.0` |
