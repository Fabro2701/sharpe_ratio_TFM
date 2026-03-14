from core.dgp import ARProcess, ARGARCHProcess, IIDProcess, NormalInnov, StudentTInnov
from core.synth import TrajectorySpec, SyntheticGenerator, summary
from core.models import AR1NormalModel, IIDStudentTModel

specs = [TrajectorySpec(IIDProcess(StudentTInnov(df=5.5).calibrate_params(mu=1.5, sigma=1)),    
                        "iid_student",  n=1, length=10000)]

data = SyntheticGenerator(seed=42).generate(specs)
print(data)
print(summary(data))

model = IIDStudentTModel()
print(model.fit(data.loc[('iid_student', 0)].value.values))
