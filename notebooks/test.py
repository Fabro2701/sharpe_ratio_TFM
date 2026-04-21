from pathlib import Path

from config import RESULTS_DIR
from core.sr_study_analysis import run_analysis, run_selected_configs, ExperimentSpec
from core.sr_sim import StudyType

df= 80
eta = -0.3

from core.dgp import DGP_EXAMPLES, GARCHProcess, ARGARCHProcess



DGP_EXAMPLES["argarch_dgp"]=lambda: ARGARCHProcess(phi=0.3)
DGP_EXAMPLES["garch_dgp"]=lambda: GARCHProcess(dist='skewt', dist_params=[8.0,-0.3])

# name : (dgp_names, model_short_names)
scenarios = {
    "aux": (
        #["argarch_dgp"],
        #["garch_dgp"],
        #["ar1_06_normal", "ar1_m06_normal"],
        #["ar1_06_normal", "ar1_06_t6", "garch_normal", "garch_t"],
        ["ar1_06_normal","ar1_m06_normal","ar1_06_t6", "ar1_m06_t6"],

        #["ar1_garch11normal"],
        ["ar1_garch11symm", "ar1_nonnormal"],
        #["garch11"],

        
        #["iid_t6"],
        #["iid_nonnormal"],
    ),
}

parameters = {
    "n_sim": [1_000, 10_000, 30_000],
    "T":     [100, 500, 2000],
    #"T":     [3000, 5000],
}

N_SIM  = 30_000
N_JOBS = 1
param_name = "T"
param_values = parameters[param_name]

experiments = {
    "aux": ExperimentSpec(
        scenario   = scenarios["aux"],
        param_name = param_name,
        param_values = param_values,
        study_type = StudyType.TWO_SIDED_COVERAGE,
        calib_sigma   = 1.0,
        n_default  = N_SIM,
        n_jobs = N_JOBS,
        th_moments = True,
    ),
}
run_selected_configs(
    experiments,
    selected_experiments=[
        "aux",
    ],
)

line_plot_kargs = dict(reverse=False, 
                       linewidth=1,markers=['D', 's', 'o', 'X', 'v'])

run_analysis(experiments, "aux", alpha=0.05, plot_mask=[0,0,1],
             line_plot_kargs=line_plot_kargs |
             dict(log=True, xticks=parameters['T']))