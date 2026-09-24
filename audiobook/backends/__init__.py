"""Optional, heavy ML backends.

Everything here imports large dependencies (torch, transformers, model toolkits) lazily
and is only loaded when the corresponding backend is explicitly selected via `Config`.
Install with `pip install -r requirements-ml.txt`. These are working adapters with real,
documented call sites — not empty stubs — but they are kept out of the core import path so
the light pipeline runs with no ML deps. See RESEARCH.md for model choices and ids.
"""
