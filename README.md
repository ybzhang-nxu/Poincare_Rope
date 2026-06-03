# PJ-RoPE Minimal Reproducibility Code

This repository contains the minimal code needed to rerun the experiments behind
the final PJ-RoPE manuscript conclusions. It intentionally excludes manuscript
source, planning notes, processed datasets, generated figures/tables, run logs,
and checkpoints.

The code regenerates outputs under ignored local directories such as `data/` and
`runs/`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the implementation checks:

```bash
python -m unittest discover -s tests -v
```

## Experiment Families

- `pjrope/`: Fourier-jet-affine kernels, diagnostics, attention modules, exact
  PJ-rotary utilities, and GRAPE special-case bases.
- `experiments/fixed_kernel_recovery.py`: fixed scalar-kernel containment probes.
- `experiments/adaptive_spectrum_*.py`: adaptive sector/order diagnostics.
- `experiments/synthetic_query_lm.py`: controlled trainable attention bridge.
- `experiments/byte_lm_smoke.py`: byte-level language and symbolic-MIDI
  long-context experiments.
- `experiments/prepare_phase_d_corpus.py`: public text-corpus preparation.
- `experiments/prepare_phase_e_*.py`: symbolic, MAESTRO MIDI, and MusicNet
  reference-MIDI corpus preparation.
- `experiments/phase_f_*.py`: LC stability, cache, retrieval, and resolution
  probes.
- `experiments/grape_appendix_*.py`: GRAPE special-case controls.
- `experiments/phase_*_export_tables.py` and `experiments/phase_*_plot_tables.py`:
  consolidation and plotting from freshly generated local runs.

Most long-context runs are computational stress tests. Use each script's
`--help` output to inspect the available settings before launching full reruns.
