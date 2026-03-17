"""
synth.py  —  Synthetic dataset generation framework.

Typical usage
-------------
    from dgp import IIDProcess, ARProcess, ARGARCHProcess, NormalInnov, StudentTInnov
    from synth import TrajectorySpec, SyntheticGenerator

    specs = [
        TrajectorySpec(
            dgp  = IIDProcess(NormalInnov()),
            name = "iid_normal",
            n    = 500,
            length = 1000,
        ),
        TrajectorySpec(
            dgp  = ARProcess(phi=[0.8], innov=StudentTInnov(df=5)),
            name = "ar1_t5",
            n    = 500,
            length = (750, 1250),        # uniform draw in [750, 1250]
        ),
        TrajectorySpec(
            dgp  = ARGARCHProcess(ar_lags=1),
            name = "ar1_garch11",
            n    = 300,
            length = lambda rng: int(rng.normal(1000, 200)),
        ),
    ]

    gen  = SyntheticGenerator(seed=42)
    data = gen.generate(specs)

    # data is a DataFrame with MultiIndex (dgp_name, traj_id) and columns:
    #   time, value
    # plus a flat column `dgp_name` for easy groupby/filter.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from core.dgp import DGP


# ─────────────────────────────────────────────────────────────────────────────
# TrajectorySpec
# ─────────────────────────────────────────────────────────────────────────────

Length = int | tuple[int, int] | Callable[[np.random.Generator], int]


@dataclass
class TrajectorySpec:
    """
    Full specification for one group of trajectories.

    Parameters
    ----------
    dgp : DGP
        Data generating process instance.
    name : str
        Unique label stored in the output (used as dgp_name).
    n : int
        Number of trajectories to generate.
    length : int | tuple[int, int] | callable
        Trajectory length.
        - int:   every trajectory has this exact length.
        - tuple: (min, max) — length drawn uniformly from [min, max] per trajectory.
        - callable(rng) -> int: arbitrary length distribution.
    """
    dgp:    DGP
    name:   str
    n:      int
    length: Length = 1000

    def sample_length(self, rng: np.random.Generator) -> int:
        if isinstance(self.length, int):
            return self.length
        if isinstance(self.length, tuple):
            lo, hi = self.length
            return int(rng.integers(lo, hi + 1))
        if callable(self.length):
            return max(2, int(self.length(rng)))
        raise TypeError(f"Unsupported length type: {type(self.length)}")


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticGenerator:
    """
    Generates synthetic datasets from a list of TrajectorySpec objects.

    Parameters
    ----------
    seed : int
        Master seed.  All randomness is derived from this.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    # ── public ────────────────────────────────────────────────────────────

    def generate(self, specs: list[TrajectorySpec]) -> pd.DataFrame:
        """
        Simulate all trajectories and return a tidy DataFrame.

        Output schema
        -------------
        Index : (dgp_name, traj_id)
        Columns:
            time      int    — time index within trajectory, starting at 0
            value     float  — simulated observation

        Parameters
        ----------
        specs : list[TrajectorySpec]
            May contain duplicate names; trajectories are numbered globally
            within each name group.

        Returns
        -------
        pd.DataFrame
        """
        self._validate(specs)
        #rng    = np.random.default_rng(self.seed)
        chunks = []

        for spec in specs:
            #group_rng = np.random.default_rng(
            #    rng.integers(0, 2**31)
            #)
            group_rng = np.random.default_rng(self.seed)
            chunks.append(self._simulate_spec(spec, group_rng))

        df = pd.concat(chunks)
        df.index = pd.MultiIndex.from_arrays(
            [df.index.get_level_values("dgp_name"),
             df.index.get_level_values("traj_id")],
            names=["dgp_name", "traj_id"],
        )
        return df

    def generate_from_dict(self, spec_dict: dict) -> pd.DataFrame:
        """
        Convenience wrapper — accepts a dict whose values are spec dicts.

        Each value must have keys: ``dgp``, ``n``, ``length`` (optional).
        The dict key becomes the trajectory name.

        Example
        -------
        ::

            gen.generate_from_dict({
                "iid_normal":   {"dgp": IIDProcess(NormalInnov()), "n": 500},
                "ar1_t5":       {"dgp": ARProcess(phi=[0.8], innov=StudentTInnov(5)),
                                 "n": 300, "length": (500, 1500)},
            })
        """
        specs = [
            TrajectorySpec(
                dgp    = v["dgp"],
                name   = name,
                n      = v["n"],
                length = v.get("length", 1000),
            )
            for name, v in spec_dict.items()
        ]
        return self.generate(specs)

    # ── I/O ───────────────────────────────────────────────────────────────

    def save(self, df: pd.DataFrame, path: str | Path) -> None:
        """Save dataset to a parquet file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    @staticmethod
    def load(path: str | Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    # ── internals ─────────────────────────────────────────────────────────

    def _simulate_spec(
        self,
        spec: TrajectorySpec,
        rng:  np.random.Generator,
    ) -> pd.DataFrame:
        rows = []
        for i in range(spec.n):
            T    = spec.sample_length(rng)
            data = spec.dgp.simulate(T, rng)
            rows.append(
                pd.DataFrame(
                    {"dgp_name": spec.name, "traj_id": i,
                     "time": np.arange(T), "value": data}
                )
            )
        df = pd.concat(rows, ignore_index=True)
        return df.set_index(["dgp_name", "traj_id"])

    @staticmethod
    def _validate(specs: list[TrajectorySpec]) -> None:
        if not specs:
            raise ValueError("specs list is empty.")
        names = [s.name for s in specs]
        if len(names) != len(set(names)):
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate spec names: {dupes}. Each name must be unique.")
        for s in specs:
            if s.n < 1:
                raise ValueError(f"Spec '{s.name}': n must be >= 1.")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset inspection utilities
# ─────────────────────────────────────────────────────────────────────────────

def summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-DGP summary statistics.

    Columns: n_trajectories, mean_length, min_length, max_length,
             value_mean, value_std, value_skew, value_kurt

    TODO add more columns, lungbox, JB, hetero tests
    """
    groups = df.groupby("dgp_name")

    lengths = (
        df.reset_index()
        .groupby(["dgp_name", "traj_id"])["time"]
        .max()
        .add(1)               # length = max_time + 1
        .groupby("dgp_name")
        .agg(n_trajectories="count", mean_length="mean",
             min_length="min", max_length="max")
    )

    stats = groups["value"].agg(
        value_mean="mean",
        value_std="std",
        value_skew=lambda x: x.skew(),
        value_kurt=lambda x: x.kurt(),
        value_sr=lambda x: x.mean() / x.std()
    )

    return pd.concat([lengths, stats], axis=1).round(4)


def iter_trajectories(df: pd.DataFrame):
    """
    Yield (dgp_name, traj_id, series) for every trajectory.

    ``series`` is a 1-D numpy array of the ``value`` column.
    """
    for (name, tid), grp in df.groupby(level=["dgp_name", "traj_id"]):
        yield name, tid, grp["value"].values



if __name__ == "__main__":
    from dgp import ARProcess, ARGARCHProcess, IIDProcess, NormalInnov, StudentTInnov
    #from synth import TrajectorySpec, SyntheticGenerator, summary

    specs = [
        TrajectorySpec(IIDProcess(NormalInnov()),    "iid_normal",  n=100, length=1000),
        TrajectorySpec(IIDProcess(StudentTInnov()),    "iid_student",  n=100, length=1000),
        TrajectorySpec(ARProcess(phi=0.8),          "ar1_normal",  n=100, length=1000),
        TrajectorySpec(ARGARCHProcess(ar_lags=1),     "ar1_garch11", n=100, length=(750, 1250)),
    ]

    data = SyntheticGenerator(seed=42).generate(specs)
    print(data)
    print(summary(data))
    # MultiIndex (dgp_name, traj_id), columns: time, value