import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from core.dgp import ARProcess, IIDProcess, NormalInnov, StudentTInnov, SkewTInnov, GARCHProcess
from core.synth import TrajectorySpec, SyntheticGenerator, summary
from core.models import AR1NormalModel, IIDNormalModel, IIDStudentTModel, IIDNonNormalModel, GARCH11Model
from core.run_coverage import main
from core.coverage import run_coverage_study, DGPSpec

from core.dgp import IIDProcess, NormalInnov, StudentTInnov, ARProcess, GARCHProcess
from core.dgp import DGP_EXAMPLES
from core.synth import TrajectorySpec, SyntheticGenerator



# start_time = time.perf_counter()
# res = run_coverage_study(
#         dgp_specs=[
#             #DGPSpec(GARCHProcess(mu=0.15, omega=0.05, alpha=0.10, beta=0.85,
#             # dist='normal'), "garch"),
#             DGPSpec(IIDProcess(StudentTInnov(3)), "iid_3"),
#             DGPSpec(IIDProcess(StudentTInnov(7)), "iid_7")], 
#         avar_models=[IIDStudentTModel()],
#         target_sr=0.5, T=1000, n_sim=10000, alpha=0.05,
#         th_moments = False,
#         seed=42, verbose=False,
#         n_jobs=8
#     )

# print(res)
# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(elapsed_time)



# #dgp = GARCHProcess(mu=0.05, omega=0.05, alpha=0.10, beta=0.85,
# #             dist='t', dist_params=[8.0, 0.9]).calibrate_params(0.15, 0.3)

# dgp = IIDProcess(SkewTInnov(df=6, eta=0.8)).calibrate_params(0.15, 0.3)

# rng = np.random.default_rng(42)
# res = dgp.simulate(10000, rng)

# print(res.mean(), res.std())
# print(stats.skew(res), stats.kurtosis(res, fisher=True),)

# #plt.hist(res, bins=100)
# #plt.show()
# print(dgp.get_theo_moments()) 