from types import SimpleNamespace

import numpy as np
import pytest

from DdqnAgent import DoubleDeepQNetwork
from ModelFactory import build_model


def make_runtime_config(tmp_path):
    return SimpleNamespace(
        epsilon_decay=0.999,
        rl_stats_folder=str(tmp_path),
        prefilled_actions_file=str(tmp_path / "prefilled-actions.txt"),
    )


def make_experiment(teacher_weights):
    return SimpleNamespace(
        model_type="student_a",
        distillation_method="q_blend",
        distillation={"alpha": 0.5},
        teacher_weights=str(teacher_weights),
    )


def test_teacher_weights_remain_unchanged_after_student_replay(tmp_path):
    state_size = 4
    action_size = 3
    teacher_path = tmp_path / "teacher.weights.h5"
    build_model("teacher", state_size, action_size, 0.01).save_weights(teacher_path)

    env = SimpleNamespace(INPUT_SHAPE=state_size, OUTPUT_SHAPE=action_size)
    agent = DoubleDeepQNetwork(
        make_runtime_config(tmp_path),
        env,
        http_client=None,
        is_controlled=False,
        is_prefilled_actions=False,
        experiment_config=make_experiment(teacher_path),
    )
    before = [weight.copy() for weight in agent.teacher_model.get_weights()]

    for index in range(9):
        state = np.full(state_size, index / 10.0)
        next_state = np.full(state_size, (index + 1) / 10.0)
        agent.store(state, index % action_size, float(index), next_state, False)

    metrics = agent.experience_replay(agent.batch_size, episode=1, step=9)
    after = agent.teacher_model.get_weights()

    for expected, actual in zip(before, after):
        np.testing.assert_array_equal(actual, expected)
    assert all(np.isfinite(value) for value in metrics.values())
    assert (tmp_path / "distillation_metrics.csv").is_file()


def test_incompatible_teacher_checkpoint_fails_clearly(tmp_path):
    teacher_path = tmp_path / "incompatible.weights.h5"
    build_model("teacher", 5, 3, 0.01).save_weights(teacher_path)
    env = SimpleNamespace(INPUT_SHAPE=4, OUTPUT_SHAPE=3)

    with pytest.raises(ValueError, match="Incompatible teacher checkpoint"):
        DoubleDeepQNetwork(
            make_runtime_config(tmp_path),
            env,
            http_client=None,
            is_controlled=False,
            is_prefilled_actions=False,
            experiment_config=make_experiment(teacher_path),
        )
