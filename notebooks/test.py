from core.dgp import ARProcess, ARGARCHProcess, IIDProcess, NormalInnov, StudentTInnov
from core.synth import TrajectorySpec, SyntheticGenerator, summary
from core.models import AR1NormalModel, IIDStudentTModel

specs = [TrajectorySpec(IIDProcess(StudentTInnov(df=5.5).calibrate_params(mu=1.5, sigma=0.4)),    
                        "iid_student",  n=10, length=10000)]

data = SyntheticGenerator(seed=42).generate(specs)
print(summary(data))

model = IIDStudentTModel()
print(model.fit(data.loc[('iid_student', 0)].value.values))

print(specs[0].dgp.get_theo_moments())
