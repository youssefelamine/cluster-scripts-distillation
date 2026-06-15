import json
from pathlib import Path

import pytest

from ExperimentConfig import ExperimentConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_invalid_experiment_configurations_fail_clearly():
    with pytest.raises(ValueError, match="model_type"):
        ExperimentConfig({"model_type": "unknown"})
    with pytest.raises(ValueError, match="alpha"):
        ExperimentConfig(
            {
                "model_type": "student_a",
                "distillation_method": "q_blend",
                "teacher_weights": "/tmp/teacher.weights.h5",
                "distillation": {"alpha": 1.5},
            }
        )
    with pytest.raises(ValueError, match="teacher_weights"):
        ExperimentConfig(
            {
                "model_type": "student_a",
                "distillation_method": "q_blend",
                "teacher_weights": "${TEACHER_WEIGHTS}",
                "distillation": {"alpha": 0.5},
            }
        )


def test_environment_variable_expansion(monkeypatch):
    monkeypatch.setenv("TEST_TEACHER", "/tmp/teacher.weights.h5")
    config = ExperimentConfig(
        {
            "model_type": "student_a",
            "distillation_method": "q_blend",
            "teacher_weights": "${TEST_TEACHER}",
            "distillation": {"alpha": 0.5},
        }
    )
    assert config.teacher_weights == "/tmp/teacher.weights.h5"


def test_all_experiment_json_files_validate(monkeypatch):
    monkeypatch.setenv("TEACHER_WEIGHTS", "/tmp/teacher.weights.h5")
    paths = sorted((REPOSITORY_ROOT / "experiments").rglob("*.json"))
    assert paths
    for path in paths:
        config = ExperimentConfig.from_file(path)
        assert config.experiment_name


def test_effective_configuration_can_be_written(tmp_path):
    config = ExperimentConfig({"seed": 42})
    output = tmp_path / "experiment.json"
    config.write(output)
    with output.open() as config_file:
        assert json.load(config_file)["seed"] == 42
