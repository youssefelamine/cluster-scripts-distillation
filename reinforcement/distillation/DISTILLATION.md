# Configuration-Driven Training And Distillation

`reinforcement/Main.py` remains the single training entrypoint. With no new
options it builds the original teacher model and uses the original DDQN replay
targets. Experiment JSON values are loaded first and explicit command-line
options override them.

## Models And Strategies

Available model types are `teacher`, `student_a`, and `student_b`. Student A is
the symmetric compressed model; Student B omits the sigmoid compression layer.
Available distillation methods are `none` and `q_blend`.

`q_blend` requires a teacher `.weights.h5` checkpoint with state and action
dimensions compatible with the student run. Its `distillation.alpha` value must
be in `[0.0, 1.0]`.

## Experiment Files

Experiment files live under `experiments/`. String values expand environment
variables such as `${TEACHER_WEIGHTS}`. Required runtime settings include the
experiment name, model type, distillation method, seed, attackers, episode and
step counts, topology, controlled-switch count, epsilon decay, and checkpoint
retention settings.

Explicit CLI options override JSON values:

```bash
python reinforcement/Main.py \
  --experiment-config experiments/student_a_q_blend.json \
  --seed 43
```

Existing commands continue to select the teacher baseline:

```bash
python reinforcement/Main.py -a '[h1]' -e 50 -s 100
```

## Offline Verification

The focused tests use synthetic states and replay experiences; they do not
start Mininet, Open vSwitch, packet capture, or CICFlowMeter:

```bash
pytest -q tests
bash -n kadeploy_runtime/*.sh
```

## Cluster Commands

Student smoke tests run one episode with nine steps so the replay memory exceeds
the current batch size:

```bash
TEACHER_WEIGHTS=/root/project/artifacts/teacher/best.weights.h5 \
  ./kadeploy_runtime/run_distillation_smoke_test.sh \
  experiments/smoke/student_a_q_blend.json

TEACHER_WEIGHTS=/root/project/artifacts/teacher/best.weights.h5 \
  ./kadeploy_runtime/run_distillation_smoke_test.sh \
  experiments/smoke/student_b_q_blend.json
```

Full experiments use the same launcher:

```bash
TEACHER_WEIGHTS=/root/project/artifacts/teacher/best.weights.h5 \
  ./kadeploy_runtime/run_experiment.sh experiments/student_a_q_blend.json

TEACHER_WEIGHTS=/root/project/artifacts/teacher/best.weights.h5 \
  ./kadeploy_runtime/run_experiment.sh experiments/student_b_q_blend.json
```

Each run writes its effective experiment configuration under `configs/`, model
weights and `model_manifest.json` under `models/`, and student Q-value metrics
under `rl_stats/distillation_metrics.csv`.

Evaluation accepts an explicit architecture or a manifest:

```bash
python reinforcement/Play.py --model-type student_a --model PATH.weights.h5
python reinforcement/Play.py --manifest results/train_TIMESTAMP/models/model_manifest.json
```
