STRATEGIES = {}


def register_strategy(name):
    def decorator(strategy_class):
        STRATEGIES[name] = strategy_class
        return strategy_class

    return decorator


def available_strategies():
    return tuple(sorted(STRATEGIES))


def build_strategy(name, parameters=None):
    if name not in STRATEGIES:
        supported = ", ".join(available_strategies())
        raise ValueError(f"Unknown distillation strategy '{name}'. Supported strategies: {supported}")
    return STRATEGIES[name](parameters or {})


from .NoDistillation import NoDistillation  # noqa: E402,F401
from .QBlendStrategy import QBlendStrategy  # noqa: E402,F401
