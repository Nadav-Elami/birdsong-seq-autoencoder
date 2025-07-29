## FEATURE:
Build a *research-grade* Python package named **birdsongs-seq-autoencoder** that
wraps my existing LFADS-style birdsong code and exposes:

1. **Clean package layout** (`src/birdsong/...`) that keeps every line of the
   original scripts but makes them importable.
2. **CLI utilities**
   * `birdsong-train --config path/to.yaml`
   * `birdsong-eval  --checkpoint ckpt.pt`
3. **Minimal, reproducible training run** inside `examples/quickstart.ipynb`.
4. **Pytest regression tests** that confirm the refactor preserves numerical
   results (loss within 1e-5 of the legacy script on a 10-step synthetic run).
5. **Context-engineering rules for Cursor**
   * Global rule: `.cursor/rules/project_overview.mdc` (already added)
   * Scoped rules: `data_pipeline.mdc`, `model_conventions.mdc`, `testing.mdc`
6. **Docs stub** in `docs/` powered by Markdown; auto-generated API reference
   later is okay, but at minimum `motivation.md` and `architecture.md`.

The project is **research-only**—production hardening, cloud orchestration,
and hyper-scale performance are *out of scope*.

---

## EXAMPLES:
* `examples/legacy/birdsong_data_generation.py`  
  → reference implementation of data simulation we must *not* alter.
* `examples/legacy/birdsong_training_2.py`  
  → shows CLI pattern to mimic (`argparse`, config YAML).
* `examples/quickstart.ipynb` (to be created)  
  → should import the package, generate synthetic data, run one training epoch,
    and plot reconstructions.

---

## DOCUMENTATION:
* **PDF**: `docs/Tracking_Time_Varying_Syntax_in_Birdsong_with_a_Sequential_Autoencoder_CCN.pdf`  
  – abstract & methods drive the doc pages.
* PyTorch LFADS reference  
  https://github.com/google-research/lfads-torch (architecture inspiration)
* Cursor rule format  
  https://www.cursor.sh/docs/context-engineering (YAML front‑matter spec)

---

## OTHER CONSIDERATIONS:
* **No numerical drift**: Unit tests must compare tensors from the new package
  against outputs of the original scripts.
* **≤ 500 LoC per file**; split long legacy scripts if necessary.
* Strict dependency set: `torch`, `numpy`, `h5py`, `matplotlib`, `tqdm`.
* Everything must install with `pip install -e .` on Python 3.10+.
* CLI entry‑points must be registered in `pyproject.toml` under
  `[project.scripts]`.
* Common gotcha: absolute imports inside the old scripts—convert to relative
  imports to avoid circular dependencies.
