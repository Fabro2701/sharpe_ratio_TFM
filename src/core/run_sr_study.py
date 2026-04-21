"""
run_sr_study.py  —  CLI entry point for the Sharpe ratio inference study.

Supports all 6 study types via --study_type.

Usage
-----
    # two-sided coverage (original behaviour)
    python run_sr_study.py --study_type TWO_SIDED_COVERAGE

    # one-sided size
    python run_sr_study.py --study_type ONE_SIDED_COVERAGE

    # power of the two-sided test (null θ₀ = 0.3, DGP at θ = 0.5)
    python run_sr_study.py --study_type TWO_SIDED_POWER --null_sr 0.3

    # power of the one-sided test
    python run_sr_study.py --study_type ONE_SIDED_POWER --null_sr 0.3

    # two-sample size  (both DGPs at θ = 0.5)
    python run_sr_study.py --study_type TWO_SAMPLE_COVERAGE

    # two-sample power (DGP1 at θ = 0.5, DGP2 at θ = 0.3)
    python run_sr_study.py --study_type TWO_SAMPLE_POWER --target_sr2 0.3

    # other common flags
    python run_sr_study.py --T 500 --n_sim 5000 --theta 0.5
    python run_sr_study.py --dgps iid_normal ar1_06_normal
    python run_sr_study.py --models iid_normal ar1_normal
    python run_sr_study.py --out results/my_run.csv
    python run_sr_study.py --list_dgps
    python run_sr_study.py --list_models
    python run_sr_study.py --list_study_types
"""

from __future__ import annotations

import argparse
import sys

from core.dgp import DGP_EXAMPLES
from core.models import REGISTRY
from core.sr_sim import DGPSpec, StudyType, run_study, study_report
from config import RESULTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# DGP catalogue
# ─────────────────────────────────────────────────────────────────────────────

def make_dgp_specs() -> list[DGPSpec]:
    return [DGPSpec(dgp(), name) for name, dgp in DGP_EXAMPLES.items()]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(args=None):
    p = argparse.ArgumentParser(
        description="Sharpe Ratio Inference Study (coverage & power)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── study type ────────────────────────────────────────────────────────────
    p.add_argument(
        "--study_type",
        type=str,
        default="TWO_SIDED_COVERAGE",
        choices=[st.name for st in StudyType],
        help="Which inference scenario to simulate.  See --list_study_types.",
    )

    # ── primary DGP / test parameters ─────────────────────────────────────────
    p.add_argument("--theta",     type=float, default=0.5,
                   help="True SR for the primary DGP (calibration target θ).")
    p.add_argument("--null_sr",   type=float, default=None,
                   help=(
                       "Null hypothesis SR θ₀ (one-sample studies).  "
                       "Defaults to --theta for coverage, must be set for power."
                   ))

    # ── two-sample parameters ─────────────────────────────────────────────────
    p.add_argument("--theta2",    type=float, default=0.3,
                   help="True SR for the secondary DGP (two-sample studies).")
    p.add_argument("--null_diff", type=float, default=0.0,
                   help="Null difference SR₁ − SR₂ ≤ Δ₀ (two-sample studies).")
    p.add_argument("--dgps2",     nargs="*",  default=None, metavar="NAME",
                   help=(
                       "Secondary DGP subset for two-sample studies.  "
                       "Defaults to the same set as --dgps."
                   ))

    # ── simulation settings ───────────────────────────────────────────────────
    p.add_argument("--T",         type=int,   default=1000)
    p.add_argument("--n_sim",     type=int,   default=2000)
    p.add_argument("--alpha",     type=float, default=0.05)
    p.add_argument("--calib_mu",  type=float, default=None)
    p.add_argument("--calib_sigma", type=float, default=None)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--tol",       type=float, default=0.03,
                   help="Pass/fail half-band around nominal coverage.")

    # ── subset filters ────────────────────────────────────────────────────────
    p.add_argument("--dgps",      nargs="*",  default=None, metavar="NAME")
    p.add_argument("--models",    nargs="*",  default=None, metavar="SHORT_NAME")

    # ── misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--out",          type=str,  default=None)
    p.add_argument("--quiet",        action="store_true")
    p.add_argument("--th_moments",   action="store_true",
                   help="Use theoretical moments instead of estimated ones.")
    p.add_argument("--n_jobs",       type=int,  default=1)

    # ── listing helpers ───────────────────────────────────────────────────────
    p.add_argument("--list_dgps",         action="store_true")
    p.add_argument("--list_models",       action="store_true")
    p.add_argument("--list_study_types",  action="store_true")

    return p.parse_args(args)


# ─────────────────────────────────────────────────────────────────────────────
# Filter helpers
# ─────────────────────────────────────────────────────────────────────────────

def _filter_specs(all_specs: list[DGPSpec], names: list[str] | None) -> list[DGPSpec]:
    if names is None:
        return all_specs
    by_name = {s.name: s for s in all_specs}
    missing = [n for n in names if n not in by_name]
    if missing:
        print(f"[ERROR] Unknown DGP name(s): {missing}")
        print(f"        Available: {sorted(by_name)}")
        sys.exit(1)
    return [by_name[n] for n in names]


def _filter_models(short_names: list[str] | None) -> list:
    if short_names is None:
        return list(REGISTRY.values())
    missing = [n for n in short_names if n not in REGISTRY]
    if missing:
        print(f"[ERROR] Unknown model short_name(s): {missing}")
        print(f"        Available: {sorted(REGISTRY)}")
        sys.exit(1)
    return [REGISTRY[n] for n in short_names]


# ─────────────────────────────────────────────────────────────────────────────
# Null SR defaulting logic
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_null_sr(args, study_type: StudyType) -> float | None:
    """
    • Coverage variants: null_sr = target_sr (null is true by definition).
    • Power variants:    user must supply --null_sr explicitly.
    • Two-sample:        null is controlled via --null_diff; return None.
    """
    if study_type.is_two_sample:
        return None   # not used; null_diff handles it

    if not study_type.is_power:
        return args.theta   # coverage: null = truth

    # Power study
    if args.null_sr is None:
        print(
            f"[ERROR] --null_sr is required for power study type '{study_type.name}'.\n"
            f"        Supply a value different from --theta ({args.theta})."
        )
        sys.exit(1)

    if args.null_sr == args.theta:
        print(
            f"[WARN]  --null_sr equals --theta ({args.theta}). "
            "Power will equal the nominal size."
        )
    return args.null_sr


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(cli_args=None):
    args       = parse_args(cli_args)
    study_type = StudyType[args.study_type]
    all_specs  = make_dgp_specs()

    # ── listing helpers ───────────────────────────────────────────────────────
    if args.list_study_types:
        print("Available study types:")
        for st in StudyType:
            tag = "power" if st.is_power else "coverage"
            print(f"  {st.name:<28}  ({tag})")
        return

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

    # ── build specs / models ──────────────────────────────────────────────────
    specs   = _filter_specs(all_specs, args.dgps)
    models  = _filter_models(args.models)
    null_sr = _resolve_null_sr(args, study_type)

    calib_sigma = args.calib_sigma
    calib_mu = args.calib_mu
    if (calib_sigma is None) and (calib_mu is None):
        print("You must provide either calib_sigma or calib_mu")
        sys.exit(1)
    elif (calib_sigma is not None) and (calib_mu is not None):
        print("Provide strictly either calib_sigma or calib_mu, not both.")
        sys.exit(1)

    # Two-sample: secondary specs default to primary specs
    specs2  = None
    models2 = None
    if study_type.is_two_sample:
        raw2    = args.dgps2 if args.dgps2 is not None else args.dgps
        specs2  = _filter_specs(all_specs, raw2)
        models2 = models   # same model set by default

    # ── output path ───────────────────────────────────────────────────────────
    out_path = (
        args.out
        or RESULTS_DIR / f"{study_type.name.lower()}_T{args.T}_n{args.n_sim}.csv"
    )

    # ── run ───────────────────────────────────────────────────────────────────
    results = run_study(
        study_type   = study_type,
        dgp_specs    = specs,
        avar_models  = models,
        target_sr    = args.theta,
        calib_mu     = calib_mu,
        calib_sigma  = calib_sigma,
        null_sr      = null_sr,
        T            = args.T,
        n_sim        = args.n_sim,
        alpha        = args.alpha,
        th_moments   = args.th_moments,
        seed         = args.seed,
        verbose      = not args.quiet,
        n_jobs       = args.n_jobs,
        dgp_specs2   = specs2,
        avar_models2 = models2,
        target_sr2   = args.theta2,
        null_diff    = args.null_diff,
    )

    if not args.quiet:
        print("\n" + study_report(results, alpha=args.alpha, tol=args.tol))

    results.to_csv(out_path, index=False)
    print(f"\nRaw results saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Debug entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_args = [
        "--study_type", "TWO_SIDED_COVERAGE",
        "--T",          "1000",
        "--n_sim",      "10000",
        "--theta",      "0.5",
        "--calib_mu",   "0.15",
        "--dgps",       "iid_t6",
        "--models",     "iid_normal", "iid_student_t", "iid_nonnormal",
        "--seed",       "43",
        "--n_jobs",     "8",
    ]
    main(test_args)