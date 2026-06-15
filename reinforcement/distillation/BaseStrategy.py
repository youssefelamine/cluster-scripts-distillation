from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    requires_teacher = False

    def __init__(self, parameters):
        self.parameters = parameters

    @abstractmethod
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
        pass
