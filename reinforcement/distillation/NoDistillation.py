import numpy as np

from . import register_strategy
from .BaseStrategy import BaseStrategy


@register_strategy("none")
class NoDistillation(BaseStrategy):
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
        targets = np.array(student_q_values, copy=True)
        for index, (_, action, reward, _, done) in enumerate(minibatch):
            bellman_target = reward
            if not done:
                next_action = np.argmax(next_student_q_values[index])
                bellman_target += gamma * next_target_q_values[index][next_action]
            targets[index][action] = bellman_target
        return targets, None
