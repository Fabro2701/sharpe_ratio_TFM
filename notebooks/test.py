from core.dgp import ARProcess, IIDProcess, NormalInnov, StudentTInnov, SkewTInnov
from core.synth import TrajectorySpec, SyntheticGenerator, summary
from core.models import AR1NormalModel, IIDNormalModel, IIDStudentTModel, IIDNonNormalModel, GARCH11Model
from core.run_coverage import main
from core.coverage import run_coverage_study, DGPSpec
import matplotlib.pyplot as plt

from core.dgp import GARCHProcess
import numpy as np
from scipy import stats


res = run_coverage_study(
        dgp_specs=[DGPSpec(GARCHProcess(mu=0.15, omega=0.05, alpha=0.10, beta=0.85,
             dist='normal'), "garch"),
             DGPSpec(IIDProcess(NormalInnov(0.15)), "iid_normal")], 
        avar_models=[IIDNormalModel(), GARCH11Model()],
        target_sr=0.5, T=1000, n_sim=200, alpha=0.05,
        th_moments = True,
        seed=42, verbose=False
    )

print(res)

""" 

dgp = GARCHProcess(mu=0.05, omega=0.05, alpha=0.10, beta=0.85,
             dist='t', dist_params=[8.0, 0.9]).calibrate_params(0.15, 0.3)

#dgp = IIDProcess(SkewTInnov(df=6, eta=0.8)).calibrate_params(0.15, 0.3)

rng = np.random.default_rng(42)
res = dgp.simulate(10000, rng)

print(res.mean(), res.std())
print(stats.skew(res), stats.kurtosis(res, fisher=True),)

#plt.hist(res, bins=100)
#plt.show()
print(dgp.get_theo_moments()) """