from core.dgp import ARProcess, ARGARCHProcess, IIDProcess, NormalInnov, StudentTInnov
from core.synth import TrajectorySpec, SyntheticGenerator, summary
from core.models import AR1NormalModel, IIDStudentTModel
from core.run_coverage import main

test_args = [
        "--T", "500",
        "--n_sim", "1000",
        "--theta", "0.5",
        "--dgps", "iid_t5",
        "--models", "iid_student_t",
        #"--th_moments",
        "--seed", "42"
    ]

main(test_args)