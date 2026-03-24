"""
run_coverage.py  —  Entry point for the Sharpe ratio CI coverage study.

Builds the canonical DGP grid and calls run_coverage_study() from coverage.py.
All nuisance parameters (rho, skew, exc_kurt, nu) are estimated from the
sample for each trajectory — the formulas are tested as they would be used
in practice.

Usage
-----
    python run_coverage.py                          # full default run
    python run_coverage.py --T 500 --n_sim 5000
    python run_coverage.py --theta 1.0
    python run_coverage.py --dgps iid_normal ar1_phi02_normal garch11_normal
    python run_coverage.py --models iid_normal ar1_normal
    python run_coverage.py --out results/my_run.csv
    python run_coverage.py --list_dgps
    python run_coverage.py --list_models
"""

import argparse
import sys
import os

from core.dgp import (
    IIDProcess, ARProcess,
    NormalInnov, StudentTInnov, GaussianMixtureInnov, GARCHProcess
)
from core.models import REGISTRY
from core.coverage import DGPSpec, run_coverage_study, coverage_report
from config import RESULTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Canonical DGP catalogue
# ─────────────────────────────────────────────────────────────────────────────

def make_dgp_specs() -> list[DGPSpec]:
    """
    Return the canonical list of DGPSpec objects.

    Each DGP is passed raw (uncalibrated). CalibratedDGP inside
    run_coverage_study shifts the mean so the unconditional SR equals
    target_sr automatically, preserving all dynamics.
    """
    a_g, b_g = 0.08, 0.87          # GARCH(1,1) canonical parameters

    return [
        # ── IID families ─────────────────────────────────────────────────
        DGPSpec(IIDProcess(NormalInnov()), "iid_normal"),
        DGPSpec(IIDProcess(StudentTInnov(df=10)), "iid_t10"),
        DGPSpec(IIDProcess(StudentTInnov(df=5)), "iid_t5"),
        #DGPSpec(
        #    IIDProcess(GaussianMixtureInnov([-2, 2], [1, 1])),
        #    "iid_gmix",
        #),

        # ── AR(1) Normal ──────────────────────────────────────────────────
        DGPSpec(ARProcess(phi=0.6, innov=NormalInnov()),    "ar1_phi06_normal"),
        DGPSpec(ARProcess(phi=-0.6, innov=NormalInnov()),   "ar1_phi-06_normal"),

        # ── AR(1) Student-t ───────────────────────────────────────────────
        DGPSpec(ARProcess(phi=0.6, innov=StudentTInnov(df=6)),  "ar1_phi06_t6"),
        DGPSpec(ARProcess(phi=-0.6, innov=StudentTInnov(df=6)),  "ar1_phi-06_t6"),

        # ── GARCH(1,1) ────────────────────────────────────────────────────
        DGPSpec(
             GARCHProcess(
                 mu=0.0, omega=1-a_g-b_g, alpha=a_g, beta=b_g, dist="normal"),
             "garch11_normal",
         ),
        # DGPSpec(
        #     ConstMeanGARCHProcess(
        #         mu=0.0, omega=1-a_g-b_g, alpha=a_g, beta=b_g,
        #         dist="t", dist_params={"nu": 8}),
        #     "garch11_t8",
        # ),

        # # ── AR(1)-GARCH(1,1) ─────────────────────────────────────────────
        # DGPSpec(
        #     ARGARCHProcess(
        #         ar_lags=1, p_vol=1, q_vol=1, dist="t",
        #         params=[0.0, 0.2, 1-a_g-b_g, a_g, b_g, 8.0]),
        #     "ar1_garch11_t8",
        # ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(args=None):
    p = argparse.ArgumentParser(
        description="Sharpe Ratio CI Coverage Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--theta",       type=float, default=0.5,
                   help="True Sharpe ratio θ₀")
    p.add_argument("--T",           type=int,   default=1000,
                   help="Trajectory length")
    p.add_argument("--n_sim",       type=int,   default=2000,
                   help="Monte Carlo replications per cell")
    p.add_argument("--alpha",       type=float, default=0.05,
                   help="Nominal CI error rate  (target coverage = 1 - alpha)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--calibration_n", type=int, default=100_000,
                   help="Pilot sample size for DGP mean calibration")
    p.add_argument("--tol",         type=float, default=0.03,
                   help="Pass/fail half-band around nominal coverage")
    p.add_argument("--dgps",        nargs="*",  default=None, metavar="NAME",
                   help="DGP subset by name.  See --list_dgps.")
    p.add_argument("--models",      nargs="*",  default=None, metavar="SHORT_NAME",
                   help="AvarModel subset by short_name.  See --list_models.")
    p.add_argument("--out",         type=str,   default=None,
                   help="Output CSV path (auto-named if omitted)")
    p.add_argument("--list_dgps",   action="store_true",
                   help="Print available DGP names and exit")
    p.add_argument("--list_models", action="store_true",
                   help="Print available AvarModel short_names and exit")
    p.add_argument("--quiet",       action="store_true",
                   help="Suppress per-cell progress output")
    p.add_argument("--th_moments",  action="store_true",
                   help="Use theoretical moments")
    return p.parse_args(args)


def _filter_specs(all_specs, names):
    if names is None:
        return all_specs
    by_name = {s.name: s for s in all_specs}
    missing = [n for n in names if n not in by_name]
    if missing:
        print(f"[ERROR] Unknown DGP name(s): {missing}")
        print(f"        Available: {sorted(by_name)}")
        sys.exit(1)
    return [by_name[n] for n in names]


def _filter_models(short_names):
    if short_names is None:
        return list(REGISTRY.values())
    missing = [n for n in short_names if n not in REGISTRY]
    if missing:
        print(f"[ERROR] Unknown model short_name(s): {missing}")
        print(f"        Available: {sorted(REGISTRY)}")
        sys.exit(1)
    return [REGISTRY[n] for n in short_names]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(cli_args=None):
    args = parse_args(cli_args)

    all_specs = make_dgp_specs()

    if args.list_dgps:
        print("Available DGPs:")
        for s in all_specs:
            print(f"  {s.name}")
        return

    if args.list_models:
        print("Available AvarModels:")
        for key, m in REGISTRY.items():
            print(f"  {key:<22}  ({m.name})")
        return

    specs  = _filter_specs(all_specs, args.dgps)
    models = _filter_models(args.models)

    out_path = args.out or RESULTS_DIR / f"coverage_T{args.T}_n{args.n_sim}.csv"

    
    results = run_coverage_study(
        dgp_specs     = specs,
        avar_models   = models,
        target_sr     = args.theta,
        T             = args.T,
        n_sim         = args.n_sim,
        alpha         = args.alpha,
        th_moments    = args.th_moments,
        seed          = args.seed,
        verbose       = not args.quiet,
    )

    if not args.quiet:
        print("\n" + coverage_report(results, alpha=args.alpha, tol=args.tol))
    results.to_csv(out_path, index=False)
    print(f"\nRaw results saved → {out_path}")



if __name__ == "__main__":
    #main()
        # configuración rápida de debug
    test_args = [
        "--T", "500",
        "--n_sim", "10000",
        "--theta", "0.5",
        "--dgps", "iid_normal", "iid_t5", "ar1_phi06_normal", "ar1_phi06_t6",
        "--models", "iid_normal", "iid_student_t",
        #"--th_moments",
        "--seed", "42"
    ]

    main(test_args)