from core.dgp import ARProcess, ARGARCHProcess, IIDProcess, NormalInnov, StudentTInnov
from core.synth import TrajectorySpec, SyntheticGenerator, summary

specs = [TrajectorySpec(ARProcess(phi=[0.8], drift=2, innov=StudentTInnov()).calibrate_params(mu=1.5, sigma=0.4),    
                        "ar1_student",  n=100, length=10000),
        TrajectorySpec(ARProcess(phi=[0.8], drift=2, innov=NormalInnov()).calibrate_params(mu=1.5, sigma=0.4),    
                        "ar2_normal",  n=100, length=10000)]

data = SyntheticGenerator(seed=42).generate(specs)
print(data)
print(summary(data))