# Erasable but Not Dormant

Code and per-run results for *Erasable but Not Dormant: Locating Textual Backdoors in Fine-Tuned Transformer Encoders, and Why Trigger-Free Removal Fails*.

## Contents

```
notebooks/
  Study_01_Run_Experiments.ipynb        # trains models and runs all analyses (Colab)
  Study_02_Make_Tables_and_Figures.ipynb # builds tables and figures from results/
src/
  bdstudy.py      # attacks, oracles, finders, erasure, baselines, job runner
  aggregate.py    # tables (LaTeX/CSV) and figures from result files
paper/
  make_tables.py  # converts aggregate.py CSVs into the LaTeX table bodies used in the paper
  figs/
results/          # one JSON file per job (dataset, model, attack, poison rate, seed)
```

## Reproducing

1. Open `notebooks/Study_01_Run_Experiments.ipynb` in Google Colab (A100 recommended).
2. Set `STAGE` and run all cells; repeat for stages `1_core`, `2_more_datasets`, `3_poison_rates`, `4_hsol_matched`.
3. Run `notebooks/Study_02_Make_Tables_and_Figures.ipynb`.

The notebooks write `src/` modules into the runtime themselves, so no installation is needed in Colab. SynBkd and StyleBkd data are cloned from [thunlp/HiddenKiller](https://github.com/thunlp/HiddenKiller) and [thunlp/StyleAttack](https://github.com/thunlp/StyleAttack).

Results in the paper were produced with PyTorch 2.14.1 (CUDA 13.0) and Python 3.13 on an NVIDIA A100 40GB; the full grid took about 19 GPU-hours.

## Using the result files without rerunning

```python
import sys; sys.path.insert(0, "src")
import aggregate as A
R = A.load("results")
A.table_oracle(R, "out")
```

## Citation

```bibtex
@article{auyon2026erasable,
  author  = {Auyon, Md Omum Siddique and others},
  title   = {Erasable but Not Dormant: Locating Textual Backdoors in Fine-Tuned Transformer Encoders, and Why Trigger-Free Removal Fails},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT. Attack data from third-party repositories keep their original licenses.
