"""Beach robot simulation and reinforcement-learning environment."""

__all__ = ["Beach"]


def __getattr__(name):
    if name == "Beach":
        from .model import Beach

        return Beach
    raise AttributeError(name)
