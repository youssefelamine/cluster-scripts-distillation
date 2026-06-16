# Configuration-Driven Training And Distillation

This repository now includes a configuration-driven framework for running the
original teacher DDQN training, compressed student training, and Q-value
knowledge distillation experiments without duplicating the training loop.

The original entrypoint remains:

```text
reinforcement/Main.py
```

The original agent remains:

```text
reinforcement/DdqnAgent.py
```

The old command style still works:

```bash
python reinforcement/Main.py -a '[h1]' -e 50 -s 100
```

When no experiment configuration or distillation options are supplied, the
framework defaults to:

```text
model_type = teacher
distillation_method = none
```

That means default training still builds the original teacher architecture and
uses the original DDQN replay target behavior.

## High-Level Workflow

The cluster-side workflow is:

```text
kadeploy_runtime/run_experiment.sh
  -> kadeploy_runtime/start_training.sh
    -> reinforcement/Main.py --experiment-config ...
      -> ExperimentConfig validates the selected experiment
      -> Configuration prepares topology paths and output folders
      -> Environment exposes the state and action dimensions
      -> ModelFactory builds the selected neural network architecture
      -> DdqnAgent builds online and target models
      -> DdqnAgent optionally loads a separate frozen teacher model
      -> DistillationStrategy constructs replay targets
      -> DdqnAgent fits the selected model
      -> Main.py saves checkpoints, metrics, config, manifest, and results
```

The important architectural decision is that model architecture and
distillation method are independent concepts:

```text
model_type selects the network architecture.
distillation_method selects how replay targets are constructed.
```

This gives the framework a clean experiment matrix:

```text
teacher   + none    -> original teacher baseline
student_a + q_blend -> Student A distillation experiment
student_b + q_blend -> Student B distillation experiment
```

The design deliberately avoids separate training-loop copies such as
`StudentAMain.py` or `StudentBMain.py`. All experiments use the same environment
loop, reward calculation, checkpoint behavior, and result structure.

## ExperimentConfig

The experiment configuration logic lives in:

```text
reinforcement/ExperimentConfig.py
```

It exists to move experiment choices out of hardcoded Python and into JSON files
that can be launched independently on cluster nodes.

An experiment file contains fields like:

```json
{
  "experiment_name": "student_a_q_blend_seed42",
  "model_type": "student_a",
  "distillation_method": "q_blend",
  "teacher_weights": "${TEACHER_WEIGHTS}",
  "distillation": {
    "alpha": 0.5
  },
  "seed": 42,
  "attackers": "[h1]",
  "episodes": 50,
  "steps": 100,
  "hosts_topo_file": "hosts-toplogy-6hosts",
  "nbr_controlled_switches": 4,
  "epsilon_decay": 0.999,
  "checkpoint_every": 5,
  "keep_last_checkpoints": 10
}
```

What it does:

- Loads experiment JSON files.
- Expands environment variables inside string values, such as
  `${TEACHER_WEIGHTS}`.
- Validates the selected `model_type`.
- Validates the selected `distillation_method`.
- Validates `q_blend` has a teacher checkpoint path.
- Validates `distillation.alpha` is in `[0.0, 1.0]`.
- Validates positive episode, step, checkpoint, and retention counts.
- Validates controlled-switch limits.
- Writes the effective experiment configuration into each run output folder.

The validation is intentionally strict. It is better for a cluster job to fail
before starting Mininet than to spend time running an invalid or unreproducible
experiment.

The precedence order is:

```text
built-in defaults
  -> experiment JSON
    -> explicit command-line overrides
```

For example:

```bash
./kadeploy_runtime/run_experiment.sh \
  experiments/student_a_q_blend.json \
  --seed 43
```

uses the JSON file but overrides its seed with `43`. The effective
configuration is saved under:

```text
results/train_*/configs/experiment.json
```

Alternatives that were possible:

- Use only command-line arguments.
- Create one Python script per experiment.
- Use YAML instead of JSON.
- Use executable Python config files.

Those alternatives were not chosen because CLI-only runs are easy to mistype,
separate scripts duplicate the training loop, YAML adds another parser
dependency, and executable config files are unnecessary for these experiments.

Trade-offs:

- JSON is verbose.
- JSON does not support comments.
- Every new field must be added to the schema/defaults.
- Bad configs fail early instead of being silently accepted.

If this component were removed, cluster nodes could no longer select
experiments reproducibly by JSON file, and `Main.py` would need to grow ad hoc
argument parsing for every new experiment.

## ModelFactory

Model construction lives in:

```text
reinforcement/ModelFactory.py
```

The public interface is:

```python
build_model(model_type, state_size, action_size, learning_rate)
```

The factory currently supports:

```text
teacher
student_a
student_b
```

The teacher architecture is preserved exactly:

```text
Input: nS
Dense: nS, ReLU
Dense: 2*nS, ReLU
Dense: 4*(nS+nA)+2, ReLU
Dense: 2*nS+2, Sigmoid
Output: nA, Linear
```

Student A is the symmetric compressed student:

```text
Input: nS
Dense: nS, ReLU
Dense: nS, ReLU
Dense: 2*(nS+nA)+1, ReLU
Dense: nS+1, Sigmoid
Output: nA, Linear
```

Student B is the structural variant without the sigmoid compression layer:

```text
Input: nS
Dense: nS, ReLU
Dense: nS, ReLU
Dense: 2*(nS+nA)+1, ReLU
Output: nA, Linear
```

Stable layer names are used:

```text
hidden_1
hidden_2
expansion
compression
q_values
```

Student B intentionally has no `compression` layer because dropping that layer
is the controlled structural change being tested.

Why this exists:

The original architecture was hardcoded directly inside `DdqnAgent.build_model`.
That made the agent responsible for both training mechanics and architecture
definition. `ModelFactory` separates those concerns. The agent asks for a model
by name; the factory owns the layer details.

How it connects:

`DdqnAgent.build_model()` delegates to `ModelFactory.build_model()`. This means
the online model, target model, and frozen teacher model all use the same
registry mechanism.

Alternatives that were possible:

- Keep `if model_type == ...` logic inside `DdqnAgent`.
- Create separate files for each student architecture.
- Use subclassed Keras models.
- Define layer lists directly in JSON.

Those alternatives were not chosen because they either couple architecture to
training logic, add unnecessary boilerplate, complicate Keras weight loading, or
make it too easy to define invalid architectures in config files.

Trade-offs:

- Adding a future model still requires Python code.
- The registry is explicit rather than dynamically discovered.
- Layer shapes must stay compatible with the saved checkpoints.

Hidden constraints:

- Teacher weights are saved and loaded as weights-only `.weights.h5` files.
- The teacher architecture must remain shape-compatible with older unnamed
  Keras teacher models.
- Teacher, Student A, and Student B must all use the same state size and action
  size for a given topology/action-space setup.

If this component were removed, student selection would collapse back into
hardcoded agent logic, and evaluation would not know how to rebuild student
architectures before loading weights.

## Distillation Strategy Package

Replay-target construction lives in:

```text
reinforcement/distillation/
```

The package contains:

```text
BaseStrategy.py
NoDistillation.py
QBlendStrategy.py
__init__.py
```

The strategy layer exists because the main difference between normal teacher
training and distilled student training is not the environment loop. The main
difference is how the replay targets are constructed.

During replay, the agent already has:

```text
sampled experiences
current-state Q-values from the online model
next-state Q-values from the online model
next-state Q-values from the target model
gamma
optional frozen teacher model
```

The selected strategy receives those values and returns the target matrix used
for `model.fit`.

### BaseStrategy

`BaseStrategy` defines the common interface:

```python
construct_targets(
    minibatch,
    student_q_values,
    next_student_q_values,
    next_target_q_values,
    gamma,
    teacher_model=None,
    states=None,
)
```

It exists so every current and future strategy has the same contract.

Alternatives that were possible:

- Use plain standalone functions.
- Let strategies mutate the agent directly.
- Keep all target logic inside `experience_replay`.

Those alternatives were not chosen because class-based strategies can hold
parameters such as `alpha`, mutating the agent would make strategies too
tightly coupled, and keeping all target logic inside replay would make the
agent rigid again.

If this interface were removed, future distillation methods would likely drift
into inconsistent function signatures or direct agent modifications.

### NoDistillation

`NoDistillation` reproduces the original DDQN target behavior.

For each replay item:

```text
If done:
  target = reward
Else:
  next_action = argmax(online_model(next_state))
  target = reward + gamma * target_model(next_state)[next_action]
```

Then:

```text
target_vector = online_model(state)
target_vector[action] = target
```

This preserves the Double DQN behavior: the online model selects the next action
and the target model evaluates that action.

Why it exists:

Backward compatibility. The old teacher training path must remain the default.
Making normal replay a strategy also makes the non-distilled path testable.

If this component were removed, `teacher + none` would have no explicit replay
target implementation, and the original logic would need to be hardcoded back
into the agent.

### QBlendStrategy

`QBlendStrategy` implements the soft-target distillation mechanism from the
initial observation plan.

For each replay batch:

1. Predict teacher Q-values for the current batch states.
2. Initialize the entire student target matrix from the teacher Q-values.
3. Compute the normal Bellman target for each taken action.
4. Replace only the taken action target with a blend:

```text
target[action] = alpha * bellman_target + (1 - alpha) * teacher_Q[action]
```

For all non-taken actions:

```text
target[other_action] = teacher_Q[other_action]
```

The meaning of `alpha` is:

```text
alpha = 0.0 -> taken action fully follows teacher Q-value
alpha = 0.5 -> taken action is half Bellman target and half teacher Q-value
alpha = 1.0 -> taken action fully follows Bellman target
```

Even when `alpha = 1.0`, non-taken actions still come from the teacher. That is
intentional: the strategy preserves teacher relational information across the
action space while allowing the taken action to be shaped by environment
rewards.

Why it exists:

This is the core training-time distillation mechanism for Student A and Student
B. It lets the student learn from both environment feedback and the teacher's
Q-value structure.

Alternatives that were possible:

- Pure imitation of teacher Q-values.
- Pure DDQN Bellman targets without teacher Q-values.
- KL divergence over softened Q-value distributions.
- Hidden-layer feature matching.
- Relational distillation.

Those alternatives were not chosen for the starter framework because the
initial plan specifically called for blended Bellman/teacher Q targets. KL,
feature matching, and relational distillation introduce additional design
choices such as temperature, layer alignment, and relation definitions.

Trade-offs:

- The student can inherit teacher mistakes.
- Teacher checkpoint quality matters.
- Teacher and student output dimensions must match.
- Training does an additional teacher prediction during replay.
- The scale of teacher Q-values influences the student targets.

Hidden constraints:

- The teacher model must be compatible with the same state and action
  dimensions.
- The teacher is used only for prediction and is never fit.
- Distillation metrics are batch-level metrics over replay samples, not over
  every possible environment state.

If this component were removed, Student A and Student B could still train as
small DDQN models, but the initial knowledge-distillation plan would not be
implemented.

## DdqnAgent Integration

The agent file is:

```text
reinforcement/DdqnAgent.py
```

The agent still owns:

- replay memory
- epsilon and epsilon decay
- batch size
- online model
- target model
- target-network synchronization
- Keras fitting
- model save/load

The new framework adds:

- `model_type`
- `distillation_method`
- selected `distillation_strategy`
- optional `teacher_weights`
- optional frozen `teacher_model`
- replay update counter
- distillation metrics CSV output

The agent now builds models through `ModelFactory`:

```text
self.model        -> online teacher or student model
self.model_target -> target teacher or student model
self.teacher_model -> frozen teacher model, only when required
```

For normal teacher training:

```text
self.model        -> teacher online model
self.model_target -> teacher target model
self.teacher_model -> None
```

For Student A or Student B distillation:

```text
self.model        -> student online model
self.model_target -> student target model
self.teacher_model -> frozen teacher model
```

The frozen teacher is separate on purpose. It is built as `model_type =
teacher`, loaded from the provided `.weights.h5` file, marked non-trainable, and
never passed to `fit`.

Why this design exists:

The agent is the correct place for replay mechanics, but it should not hardcode
every possible architecture or distillation method. Delegating model building
and target construction keeps the existing training loop intact while allowing
experiments to vary.

Replay flow:

```text
1. Sample a minibatch from replay memory.
2. Build current-state and next-state arrays.
3. Predict current Q-values with the online model.
4. Predict next-state Q-values with the online model.
5. Predict next-state Q-values with the target model.
6. Ask the selected strategy to construct targets.
7. Fit the online model on states and targets.
8. Record loss.
9. Decay epsilon.
10. Write distillation metrics if the strategy returns them.
```

Alternatives that were possible:

- Create separate student agent classes.
- Put distillation target logic in `Main.py`.
- Make the environment aware of student/teacher choices.

Those alternatives were not chosen because they duplicate DDQN mechanics,
pollute the training script with replay math, or mix learning concerns into the
network environment.

Trade-offs:

- The agent constructor does more setup than before.
- Teacher checkpoint errors happen at agent initialization.
- Strategies depend on the agent to provide the correct prediction arrays.

Hidden constraints:

- Replay batch size remains `8`.
- Replay starts only when memory contains more than 8 experiences.
- The smoke tests use 9 steps for that reason.
- The teacher checkpoint must be present on the node before student training
  starts.

If this integration were removed, the framework would have config files and
model definitions but no way to actually use them during training.

## Distillation Metrics

Student distillation writes:

```text
results/train_*/rl_stats/distillation_metrics.csv
```

Columns:

```text
episode
step
replay_update_number
student_teacher_q_mse
mean_abs_student_q
max_abs_student_q
```

Why these metrics exist:

The initial plan asks for Q-value MSE against the teacher. Student B also needs
Q-value stability observation, so mean and maximum absolute student Q-values
are recorded.

What the metrics mean:

- `student_teacher_q_mse`: mean squared error between student Q-values and
  teacher Q-values on the replay batch states.
- `mean_abs_student_q`: average absolute magnitude of student Q-values.
- `max_abs_student_q`: maximum absolute student Q-value in the replay batch.

Teacher training does not require this file and does not produce it.

Trade-offs:

- Metrics are sampled from replay batches, not the entire state space.
- They are useful for relative comparison between runs, especially Student A
  versus Student B.
- They do not replace final reward or policy evaluation.

If this output were removed, the framework would lose the main internal
distillation signal needed to compare student fidelity.

## Main.py Runtime Integration

`reinforcement/Main.py` remains the only training entrypoint.

It now additionally handles:

- `--experiment-config`
- `--model-type`
- `--distillation-method`
- `--teacher-weights`
- `--distillation-alpha`
- `--seed`
- experiment config loading and validation
- command-line overrides
- seed setup
- effective config output
- model manifest output

It deliberately does not duplicate the environment loop. The Mininet, traffic,
reward, CICFlowMeter, state transformation, action application, plotting, and
checkpoint behavior remain in the existing flow.

Why this design exists:

The safest way to add distillation without changing networking behavior is to
modify the training setup and replay-target construction while leaving the
environment interaction path alone.

Trade-offs:

- `Main.py` remains a large procedural script.
- The framework is integrated around the existing shape instead of rewriting it
  into a cleaner object-oriented runner.
- This is less elegant, but it preserves behavior and reduces regression risk.

If this integration were removed, JSON configs and launch scripts would no
longer affect actual training.

## Model Manifest

Each run writes:

```text
results/train_*/models/model_manifest.json
```

The manifest includes:

- experiment name
- model type
- distillation method
- distillation parameters
- teacher checkpoint path
- teacher checkpoint SHA-256 when available
- seed
- state size
- action size
- trainable parameter count
- topology
- controlled-switch count
- episodes
- steps
- TensorFlow version
- Keras version
- final model path

Why it exists:

A `.weights.h5` file alone is not enough to know how to rebuild the model. A
Student A checkpoint and Student B checkpoint have different shapes. The
manifest records the architecture and run context needed for later evaluation
or comparison.

Alternatives that were possible:

- Encode metadata in filenames.
- Save full Keras model artifacts.
- Keep metadata outside the run folder.

Those alternatives were not chosen because filenames become brittle, full model
saving would alter the existing weights-only convention, and metadata belongs
next to the artifacts it describes.

If the manifest were removed, evaluation would require the user to manually
remember and pass the correct model type.

## Evaluation With Play.py

Evaluation lives in:

```text
reinforcement/Play.py
```

It now supports:

```bash
python reinforcement/Play.py --model-type student_a --model PATH.weights.h5
```

or:

```bash
python reinforcement/Play.py \
  --manifest results/train_TIMESTAMP/models/model_manifest.json
```

Why this exists:

Evaluation must build the correct architecture before loading weights. Teacher,
Student A, and Student B checkpoints are not interchangeable.

Trade-offs:

- If no manifest is used, the user must pass the correct `--model-type`.
- Passing the wrong model type should fail with an incompatible checkpoint
  error rather than silently evaluating the wrong architecture.

If this support were removed, only teacher checkpoints would be straightforward
to evaluate.

## Runtime Scripts

The runtime scripts live in:

```text
kadeploy_runtime/
```

### start_training.sh

`start_training.sh` still owns the operational cluster behavior:

- root validation
- virtualenv activation
- headless display setup
- cleanup
- status files
- logging
- invocation of `reinforcement/Main.py`

It now accepts positional arguments in addition to the older `TRAIN_ARGS`
environment variable.

This preserves backward compatibility:

```bash
TRAIN_ARGS="-a [h1] -e 50 -s 100" \
  ./kadeploy_runtime/start_training.sh
```

and also supports safer direct argument passing:

```bash
./kadeploy_runtime/start_training.sh \
  --experiment-config experiments/student_a_q_blend.json
```

### run_experiment.sh

Usage:

```bash
./kadeploy_runtime/run_experiment.sh experiments/student_a_q_blend.json
```

It validates that the experiment file exists and delegates everything else to
`start_training.sh`.

It contains no training logic. That is intentional. Environment setup, cleanup,
logging, and status handling stay in one place.

Trailing arguments are passed through as CLI overrides:

```bash
TEACHER_WEIGHTS=/root/project/artifacts/teacher/best.weights.h5 \
./kadeploy_runtime/run_experiment.sh \
  experiments/student_a_q_blend.json \
  --seed 43
```

### run_distillation_smoke_test.sh

Usage:

```bash
TEACHER_WEIGHTS=/root/project/artifacts/teacher/best.weights.h5 \
./kadeploy_runtime/run_distillation_smoke_test.sh \
  experiments/smoke/student_a_q_blend.json
```

It delegates to `run_experiment.sh` and forces:

```text
episodes = 1
steps = 9
checkpoint_every = 1
keep_last_checkpoints = 1
```

The 9-step choice is not arbitrary. Replay starts only after the memory has
more than 8 experiences, so a one-episode smoke run needs at least 9 steps to
trigger one synthetic/student replay update.

If these scripts were removed, cluster-side experiment launching would become
manual again and would be more likely to diverge between nodes.

## Experiment Files

Ready-to-run experiment files live under:

```text
experiments/
```

Available files:

```text
experiments/teacher_baseline.json
experiments/student_a_q_blend.json
experiments/student_b_q_blend.json
experiments/smoke/student_a_q_blend.json
experiments/smoke/student_b_q_blend.json
```

The initial observation plan maps to these files as follows:

```text
Approach 1:
  Student A + q_blend + alpha 0.5
  experiments/student_a_q_blend.json

Approach 2:
  Student B + q_blend + alpha 0.5
  experiments/student_b_q_blend.json

Teacher baseline:
  teacher + none
  experiments/teacher_baseline.json
```

The TFLite quantization approach from the initial plan is not implemented in
this framework. It is orthogonal to training-time distillation and should be a
separate post-training tool.

## Initial Observation Plan Applicability

The framework supports the first two approaches from the initial distillation
plan.

Approach 1, Student A:

- Symmetric x0.5 compression.
- Sigmoid compression layer preserved.
- Same expand-then-compress pattern as the teacher.
- Q-blend distillation.
- Configurable alpha.

Approach 2, Student B:

- Similar reduced capacity.
- Sigmoid compression layer removed.
- Same Q-blend distillation as Student A.
- Same episodes, steps, epsilon decay, and replay behavior.

This makes the Student A versus Student B comparison meaningful because the
training loop, environment, replay batch size, reward logic, checkpointing, and
distillation strategy stay the same. The main controlled difference is the
presence or absence of the sigmoid compression layer.

Approach 3, post-training TFLite quantization:

- Not implemented.
- No training loop changes are needed for it.
- It should be added later as a separate conversion and evaluation utility.

## Example Cluster Commands

Teacher baseline:

```bash
./kadeploy_runtime/run_experiment.sh experiments/teacher_baseline.json
```

Student A smoke test:

```bash
TEACHER_WEIGHTS=/root/project/artifacts/teacher/best.weights.h5 \
./kadeploy_runtime/run_distillation_smoke_test.sh \
  experiments/smoke/student_a_q_blend.json
```

Student B smoke test:

```bash
TEACHER_WEIGHTS=/root/project/artifacts/teacher/best.weights.h5 \
./kadeploy_runtime/run_distillation_smoke_test.sh \
  experiments/smoke/student_b_q_blend.json
```

Student A full run:

```bash
TEACHER_WEIGHTS=/root/project/artifacts/teacher/best.weights.h5 \
./kadeploy_runtime/run_experiment.sh experiments/student_a_q_blend.json
```

Student B full run:

```bash
TEACHER_WEIGHTS=/root/project/artifacts/teacher/best.weights.h5 \
./kadeploy_runtime/run_experiment.sh experiments/student_b_q_blend.json
```

Student A with a different seed:

```bash
TEACHER_WEIGHTS=/root/project/artifacts/teacher/best.weights.h5 \
./kadeploy_runtime/run_experiment.sh \
  experiments/student_a_q_blend.json \
  --seed 43
```

Example four-node mapping:

```text
Node 1 -> teacher baseline
Node 2 -> Student A + q_blend + seed 42
Node 3 -> Student B + q_blend + seed 42
Node 4 -> Student A + q_blend + seed 43
```

## Artifacts And Result Locations

Each training run creates:

```text
results/train_TIMESTAMP/
```

Important outputs:

```text
results/train_*/configs/experiment.json
results/train_*/models/rl_model.weights.h5
results/train_*/models/model_manifest.json
results/train_*/models/checkpoints/latest.weights.h5
results/train_*/models/checkpoints/best.weights.h5
results/train_*/models/checkpoints/checkpoint_info.json
results/train_*/rl_stats/distillation_metrics.csv
results/train_*/training.log
results/train_*/status.txt
results/train_*/exit_code.txt
```

Teacher runs do not need `distillation_metrics.csv`. Student `q_blend` runs
write it during replay updates.

## Offline Tests

Focused offline tests live under:

```text
tests/
```

They do not require root, Mininet, Open vSwitch, packet capture, or
CICFlowMeter.

Run:

```bash
pytest -q tests
bash -n kadeploy_runtime/*.sh
```

The tests cover:

- teacher architecture compatibility
- Student A architecture and activations
- Student B architecture and activations
- student parameter counts lower than the teacher
- unchanged normal DDQN replay targets
- Q-blend targets for alpha `0.0`, `0.5`, and `1.0`
- non-taken action targets matching teacher predictions
- frozen teacher weights during student replay
- teacher weight save/reload
- loading old unnamed teacher weights
- clear failure for incompatible checkpoints
- invalid config validation
- environment-variable expansion
- finite distillation metrics
- all experiment JSON files
- shell script syntax

These tests do not prove a full Mininet experiment will succeed on the cluster,
but they do verify the model, config, replay-target, and script mechanics that
can be checked offline.

## Important Assumptions And Constraints

The framework assumes:

- Teacher and student runs use compatible state dimensions.
- Teacher and student runs use compatible action dimensions.
- Teacher checkpoints are weights-only `.weights.h5` files.
- Student distillation receives a valid teacher checkpoint path.
- The teacher checkpoint was trained for the same topology/action-space setup.
- Replay batch size remains 8.
- A cluster smoke run needs at least 9 steps to trigger replay.
- TensorFlow/Keras versions on the runtime node can load the saved weights.

The framework intentionally does not change:

- Mininet behavior
- Open vSwitch behavior
- CICFlowMeter behavior
- reward calculation
- environment state construction
- action definitions
- packet capture
- network metrics

This is important because the first distillation study should compare model and
target-construction choices, not accidentally introduce environment changes.

## What Is Not Implemented Yet

The following are intentionally outside the current starter framework:

- KL-divergence distillation.
- Feature-matching distillation.
- Relational distillation.
- Architecture search.
- TFLite post-training quantization.
- Automatic teacher checkpoint discovery.
- Multi-experiment scheduling from one script.
- Full Keras model or SavedModel artifact saving.

Those can be added later, but the current framework is scoped to the initial
observation plan: teacher baseline, Student A with Q-blend, and Student B with
Q-blend.
