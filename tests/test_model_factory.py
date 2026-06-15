import numpy as np
from tensorflow import keras

from ModelFactory import build_model


STATE_SIZE = 6
ACTION_SIZE = 4
LEARNING_RATE = 0.01


def dense_layers(model):
    return [layer for layer in model.layers if hasattr(layer, "units")]


def layer_signature(model):
    return [
        (layer.name, layer.units, layer.activation.__name__)
        for layer in dense_layers(model)
    ]


def test_teacher_architecture_matches_existing_architecture():
    model = build_model("teacher", STATE_SIZE, ACTION_SIZE, LEARNING_RATE)
    assert layer_signature(model) == [
        ("hidden_1", STATE_SIZE, "relu"),
        ("hidden_2", 2 * STATE_SIZE, "relu"),
        ("expansion", 4 * (STATE_SIZE + ACTION_SIZE) + 2, "relu"),
        ("compression", 2 * STATE_SIZE + 2, "sigmoid"),
        ("q_values", ACTION_SIZE, "linear"),
    ]


def test_student_a_architecture_and_activations():
    model = build_model("student_a", STATE_SIZE, ACTION_SIZE, LEARNING_RATE)
    assert layer_signature(model) == [
        ("hidden_1", STATE_SIZE, "relu"),
        ("hidden_2", STATE_SIZE, "relu"),
        ("expansion", 2 * (STATE_SIZE + ACTION_SIZE) + 1, "relu"),
        ("compression", STATE_SIZE + 1, "sigmoid"),
        ("q_values", ACTION_SIZE, "linear"),
    ]


def test_student_b_architecture_and_activations():
    model = build_model("student_b", STATE_SIZE, ACTION_SIZE, LEARNING_RATE)
    assert layer_signature(model) == [
        ("hidden_1", STATE_SIZE, "relu"),
        ("hidden_2", STATE_SIZE, "relu"),
        ("expansion", 2 * (STATE_SIZE + ACTION_SIZE) + 1, "relu"),
        ("q_values", ACTION_SIZE, "linear"),
    ]


def test_students_have_fewer_parameters_than_teacher():
    teacher = build_model("teacher", STATE_SIZE, ACTION_SIZE, LEARNING_RATE)
    student_a = build_model("student_a", STATE_SIZE, ACTION_SIZE, LEARNING_RATE)
    student_b = build_model("student_b", STATE_SIZE, ACTION_SIZE, LEARNING_RATE)
    assert student_a.count_params() < teacher.count_params()
    assert student_b.count_params() < teacher.count_params()


def test_teacher_weights_can_be_saved_and_reloaded(tmp_path):
    path = tmp_path / "teacher.weights.h5"
    teacher = build_model("teacher", STATE_SIZE, ACTION_SIZE, LEARNING_RATE)
    expected = teacher.predict(np.ones((2, STATE_SIZE)), verbose=0)
    teacher.save_weights(path)

    reloaded = build_model("teacher", STATE_SIZE, ACTION_SIZE, LEARNING_RATE)
    reloaded.load_weights(path)
    actual = reloaded.predict(np.ones((2, STATE_SIZE)), verbose=0)
    np.testing.assert_allclose(actual, expected)


def test_existing_unnamed_teacher_weights_can_be_loaded(tmp_path):
    path = tmp_path / "legacy_teacher.weights.h5"
    legacy = keras.Sequential(
        [
            keras.layers.Input(shape=(STATE_SIZE,)),
            keras.layers.Dense(STATE_SIZE, activation="relu"),
            keras.layers.Dense(2 * STATE_SIZE, activation="relu"),
            keras.layers.Dense(4 * (STATE_SIZE + ACTION_SIZE) + 2, activation="relu"),
            keras.layers.Dense(2 * STATE_SIZE + 2, activation="sigmoid"),
            keras.layers.Dense(ACTION_SIZE, activation="linear"),
        ]
    )
    legacy.save_weights(path)

    current = build_model("teacher", STATE_SIZE, ACTION_SIZE, LEARNING_RATE)
    current.load_weights(path)
    states = np.ones((2, STATE_SIZE))
    np.testing.assert_allclose(
        current.predict(states, verbose=0),
        legacy.predict(states, verbose=0),
    )
