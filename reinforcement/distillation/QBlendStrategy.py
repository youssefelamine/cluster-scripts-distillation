import numpy as np

from . import register_strategy
from .BaseStrategy import BaseStrategy


@register_strategy("q_blend")
class QBlendStrategy(BaseStrategy):
    requires_teacher = True

    def __init__(self, parameters):
        super().__init__(parameters)
        try:
            self.alpha = float(parameters["alpha"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("q_blend requires a numeric alpha parameter") from exc
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("q_blend alpha must be in the range [0.0, 1.0]")

    def construct_targets(
        self,
        minibatch,
        student_q_values,
        next_student_q_values,
        next_target_q_values,
        gamma,
        teacher_model=None,
        states=None,
    ):
        if teacher_model is None:
            raise ValueError("q_blend requires a frozen teacher model")
        teacher_q_values = np.asarray(teacher_model.predict(states, verbose=0))
        targets = np.array(teacher_q_values, copy=True)

        for index, (_, action, reward, _, done) in enumerate(minibatch):
            bellman_target = reward
            if not done:
                next_action = np.argmax(next_student_q_values[index])
                bellman_target += gamma * next_target_q_values[index][next_action]
            targets[index][action] = (
                self.alpha * bellman_target
                + (1.0 - self.alpha) * teacher_q_values[index][action]
            )

        metrics = {
            "student_teacher_q_mse": float(np.mean(np.square(student_q_values - teacher_q_values))),
            "mean_abs_student_q": float(np.mean(np.abs(student_q_values))),
            "max_abs_student_q": float(np.max(np.abs(student_q_values))),
        }
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError("Distillation produced non-finite Q-value metrics")
        return targets, metrics
