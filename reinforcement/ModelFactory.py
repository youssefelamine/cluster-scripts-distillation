from tensorflow import keras


MODEL_BUILDERS = {}


def register_model(model_type):
    def decorator(builder):
        MODEL_BUILDERS[model_type] = builder
        return builder

    return decorator


def available_model_types():
    return tuple(sorted(MODEL_BUILDERS))


def _compile(model, learning_rate):
    model.compile(
        loss="mean_squared_error",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    )
    return model


@register_model("teacher")
def build_teacher(state_size, action_size, learning_rate):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(state_size,), name="state"),
            keras.layers.Dense(state_size, activation="relu", name="hidden_1"),
            keras.layers.Dense(2 * state_size, activation="relu", name="hidden_2"),
            keras.layers.Dense(
                4 * (state_size + action_size) + 2,
                activation="relu",
                name="expansion",
            ),
            keras.layers.Dense(
                2 * state_size + 2,
                activation="sigmoid",
                name="compression",
            ),
            keras.layers.Dense(action_size, activation="linear", name="q_values"),
        ],
        name="teacher",
    )
    return _compile(model, learning_rate)


@register_model("student_a")
def build_student_a(state_size, action_size, learning_rate):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(state_size,), name="state"),
            keras.layers.Dense(state_size, activation="relu", name="hidden_1"),
            keras.layers.Dense(state_size, activation="relu", name="hidden_2"),
            keras.layers.Dense(
                2 * (state_size + action_size) + 1,
                activation="relu",
                name="expansion",
            ),
            keras.layers.Dense(
                state_size + 1,
                activation="sigmoid",
                name="compression",
            ),
            keras.layers.Dense(action_size, activation="linear", name="q_values"),
        ],
        name="student_a",
    )
    return _compile(model, learning_rate)


@register_model("student_b")
def build_student_b(state_size, action_size, learning_rate):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(state_size,), name="state"),
            keras.layers.Dense(state_size, activation="relu", name="hidden_1"),
            keras.layers.Dense(state_size, activation="relu", name="hidden_2"),
            keras.layers.Dense(
                2 * (state_size + action_size) + 1,
                activation="relu",
                name="expansion",
            ),
            keras.layers.Dense(action_size, activation="linear", name="q_values"),
        ],
        name="student_b",
    )
    return _compile(model, learning_rate)


def build_model(model_type, state_size, action_size, learning_rate):
    if model_type not in MODEL_BUILDERS:
        supported = ", ".join(available_model_types())
        raise ValueError(f"Unknown model type '{model_type}'. Supported model types: {supported}")
    if int(state_size) <= 0 or int(action_size) <= 0:
        raise ValueError("State and action sizes must be greater than zero")
    return MODEL_BUILDERS[model_type](int(state_size), int(action_size), learning_rate)
