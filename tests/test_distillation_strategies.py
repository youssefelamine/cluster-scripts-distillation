import numpy as np
import pytest

from distillation import build_strategy


MINIBATCH = [
    (np.array([1.0, 0.0]), 0, 2.0, np.array([0.0, 1.0]), False),
    (np.array([0.0, 1.0]), 1, -1.0, np.array([1.0, 0.0]), True),
]
STUDENT_Q = np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
NEXT_STUDENT_Q = np.array([[1.0, 3.0, 2.0], [8.0, 7.0, 6.0]])
NEXT_TARGET_Q = np.array([[4.0, 5.0, 6.0], [3.0, 2.0, 1.0]])
TEACHER_Q = np.array([[30.0, 31.0, 32.0], [40.0, 41.0, 42.0]])
GAMMA = 0.85


class PredictableTeacher:
    def predict(self, states, verbose=0):
        assert len(states) == len(TEACHER_Q)
        return TEACHER_Q


def test_existing_teacher_replay_target_behavior_remains_unchanged():
    targets, metrics = build_strategy("none").construct_targets(
        MINIBATCH,
        STUDENT_Q,
        NEXT_STUDENT_Q,
        NEXT_TARGET_Q,
        GAMMA,
    )
    expected = STUDENT_Q.copy()
    expected[0, 0] = 2.0 + GAMMA * 5.0
    expected[1, 1] = -1.0
    np.testing.assert_allclose(targets, expected)
    assert metrics is None


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_q_blend_targets_are_correct(alpha):
    targets, _ = build_strategy("q_blend", {"alpha": alpha}).construct_targets(
        MINIBATCH,
        STUDENT_Q,
        NEXT_STUDENT_Q,
        NEXT_TARGET_Q,
        GAMMA,
        teacher_model=PredictableTeacher(),
        states=np.array([item[0] for item in MINIBATCH]),
    )
    bellman_0 = 2.0 + GAMMA * 5.0
    bellman_1 = -1.0
    assert targets[0, 0] == pytest.approx(alpha * bellman_0 + (1 - alpha) * 30.0)
    assert targets[1, 1] == pytest.approx(alpha * bellman_1 + (1 - alpha) * 41.0)


def test_q_blend_non_taken_actions_equal_teacher_predictions():
    targets, _ = build_strategy("q_blend", {"alpha": 0.5}).construct_targets(
        MINIBATCH,
        STUDENT_Q,
        NEXT_STUDENT_Q,
        NEXT_TARGET_Q,
        GAMMA,
        teacher_model=PredictableTeacher(),
        states=np.array([item[0] for item in MINIBATCH]),
    )
    np.testing.assert_allclose(targets[0, 1:], TEACHER_Q[0, 1:])
    np.testing.assert_allclose(targets[1, [0, 2]], TEACHER_Q[1, [0, 2]])


def test_distillation_metrics_are_finite():
    _, metrics = build_strategy("q_blend", {"alpha": 0.5}).construct_targets(
        MINIBATCH,
        STUDENT_Q,
        NEXT_STUDENT_Q,
        NEXT_TARGET_Q,
        GAMMA,
        teacher_model=PredictableTeacher(),
        states=np.array([item[0] for item in MINIBATCH]),
    )
    assert all(np.isfinite(value) for value in metrics.values())
