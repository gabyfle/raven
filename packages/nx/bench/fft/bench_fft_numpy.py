"""NumPy (pocketfft) cross-reference for the nx_fft thumper suite.

Sanity anchor only, never a gate: run it on the same quiet host as the
thumper suite and compare the per-case medians by eye. Sizes mirror
bench_fft.ml. The NxN scaffolding in ../bench_numpy.py does not fit 1-D
FFT shapes, so this sibling reuses the same ubench harness directly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
while not (_SCRIPTS_DIR / "dune-project").exists():
    _SCRIPTS_DIR = _SCRIPTS_DIR.parent
_SCRIPTS_DIR = _SCRIPTS_DIR / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import ubench  # type: ignore

_RNG = np.random.default_rng(seed=42)


def build_benchmarks() -> List[Any]:
    benchmarks: List[Any] = []
    for n in (65536, 100000, 65521, 4099):
        a = (_RNG.random(n) + 1j * _RNG.random(n)).astype(np.complex128)
        benchmarks.append(
            ubench.bench(f"fft {n} c128 (NumPy)", lambda a=a: np.fft.fft(a))
        )
    for n in (65536, 44100, 65535, 131042):
        x = _RNG.random(n)
        benchmarks.append(
            ubench.bench(f"rfft {n} f64 (NumPy)", lambda x=x: np.fft.rfft(x))
        )
    for shape in ((65536,), (256, 8192)):
        spec = np.fft.rfft(_RNG.random(shape))
        n = shape[-1]
        benchmarks.append(
            ubench.bench(
                f"irfft {'x'.join(map(str, shape))} c128 (NumPy)",
                lambda spec=spec, n=n: np.fft.irfft(spec, n),
            )
        )
    return benchmarks


def main() -> None:
    config = (
        ubench.Config.default()
        .time_limit(1.0)
        .warmup(1)
        .min_measurements(5)
        .min_cpu(0.01)
        .geometric_scale(1.3)
        .gc_stabilization(False)
        .build()
    )
    ubench.run(build_benchmarks(), config=config, output_format="pretty", verbose=False)


if __name__ == "__main__":
    main()
