import copy
import json
import os

from ModelFactory import available_model_types
from distillation import available_strategies


DEFAULTS = {
    "experiment_name": "teacher_baseline",
    "model_type": "teacher",
    "distillation_method": "none",
    "teacher_weights": None,
    "distillation": {},
    "seed": None,
    "attackers": "[]",
    "episodes": 50,
    "steps": 100,
    "hosts_topo_file": "hosts-toplogy-6hosts",
    "nbr_controlled_switches": 4,
    "epsilon_decay": 0.999,
    "checkpoint_every": 5,
    "keep_last_checkpoints": 10,
}


def _expand_environment(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


class ExperimentConfig:
    def __init__(self, values=None, source_path=None, validate=True):
        self.source_path = source_path
        self._values = copy.deepcopy(DEFAULTS)
        if values:
            unknown = sorted(set(values) - set(DEFAULTS))
            if unknown:
                raise ValueError(f"Unknown experiment configuration field(s): {', '.join(unknown)}")
            self._values.update(_expand_environment(values))
        if validate:
            self.validate()

    @classmethod
    def from_file(cls, path, validate=True):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Experiment configuration file not found: {path}")
        try:
            with open(path) as config_file:
                values = json.load(config_file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in experiment configuration '{path}': {exc}") from exc
        if not isinstance(values, dict):
            raise ValueError("Experiment configuration must contain a JSON object")
        return cls(values, source_path=os.path.abspath(path), validate=validate)

    def with_overrides(self, overrides):
        values = self.to_dict()
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
        return ExperimentConfig(values, source_path=self.source_path)

    def validate(self):
        model_type = self.model_type
        if model_type not in available_model_types():
            raise ValueError(
                f"Invalid model_type '{model_type}'. Supported values: {', '.join(available_model_types())}"
            )

        method = self.distillation_method
        if method not in available_strategies():
            raise ValueError(
                f"Invalid distillation_method '{method}'. Supported values: {', '.join(available_strategies())}"
            )

        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise ValueError("experiment_name must be a non-empty string")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError("seed must be an integer or null")
        if not isinstance(self.distillation, dict):
            raise ValueError("distillation must be a JSON object")

        if method == "q_blend":
            if model_type == "teacher":
                raise ValueError("q_blend requires a student model_type")
            if (
                not isinstance(self.teacher_weights, str)
                or not self.teacher_weights
                or "${" in self.teacher_weights
            ):
                raise ValueError(
                    "q_blend requires teacher_weights; set TEACHER_WEIGHTS or provide a checkpoint path"
                )
            alpha = self.distillation.get("alpha")
            if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
                raise ValueError("distillation.alpha must be a number for q_blend")
            if not 0.0 <= float(alpha) <= 1.0:
                raise ValueError("distillation.alpha must be in the range [0.0, 1.0]")

        for key in ("episodes", "steps", "checkpoint_every", "keep_last_checkpoints"):
            value = self._values[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be an integer greater than 0")

        if (
            isinstance(self.nbr_controlled_switches, bool)
            or not isinstance(self.nbr_controlled_switches, int)
            or not 4 <= self.nbr_controlled_switches <= 99
        ):
            raise ValueError("nbr_controlled_switches must be an integer in the range [4, 99]")
        if (
            isinstance(self.epsilon_decay, bool)
            or not isinstance(self.epsilon_decay, (int, float))
            or not 0.1 < float(self.epsilon_decay) < 1.0
        ):
            raise ValueError("epsilon_decay must be in the range (0.1, 1.0)")
        if not isinstance(self.attackers, str):
            raise ValueError("attackers must be a string such as '[h1]'")
        if not isinstance(self.hosts_topo_file, str) or not self.hosts_topo_file.strip():
            raise ValueError("hosts_topo_file must be a non-empty string")

    def to_dict(self):
        return copy.deepcopy(self._values)

    def write(self, path):
        with open(path, "w") as config_file:
            json.dump(self.to_dict(), config_file, indent=2)

    def __getattr__(self, name):
        if name in self._values:
            return self._values[name]
        raise AttributeError(name)
