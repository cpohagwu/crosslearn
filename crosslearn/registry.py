from typing import Dict, List, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from crosslearn.agents.base import BaseAgent

_AGENT_REGISTRY: Dict[str, Type["BaseAgent"]] = {}


def register_agent(name: str):
    """
    Class decorator to register an agent class under a lowercase string key.

    Args:
        name: Registry key (case-insensitive). Convention: use the algorithm
            name in lowercase, e.g. ``'reinforce'``, ``'ppo'``.

    Usage::

        @register_agent("reinforce")
        class REINFORCE(BaseAgent):
            ...
    """
    def decorator(cls: Type["BaseAgent"]) -> Type["BaseAgent"]:
        _AGENT_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def make_agent(agent_type: str, env, **kwargs) -> "BaseAgent":
    """
    Instantiate a registered agent by name.

    This is a shortcut. For IDE autocomplete and algorithm-specific
    type hints, prefer the direct import::

        from crosslearn import REINFORCE
        agent = REINFORCE(env, learning_rate=0.01)

    Args:
        agent_type: Registered name (case-insensitive).
            Currently: ``'reinforce'``.
        env: A ``gym.Env``-compatible environment.
        **kwargs: Forwarded verbatim to the algorithm constructor.

    Returns:
        Instantiated agent.

    Raises:
        ValueError: If ``agent_type`` is not registered.

    Example::

        agent = make_agent("reinforce", env, learning_rate=0.01)
        # Exactly equivalent to:
        # agent = REINFORCE(env, learning_rate=0.01)
    """
    key = agent_type.lower()
    if key not in _AGENT_REGISTRY:
        available = sorted(_AGENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown agent type '{agent_type}'. "
            f"Registered agents: {available}. "
            f"Tip: import the class directly for full IDE support: "
            f"from crosslearn import {agent_type.upper()}"
        )
    return _AGENT_REGISTRY[key](env, **kwargs)


def list_agents() -> List[str]:
    """Return a sorted list of all registered agent names."""
    return sorted(_AGENT_REGISTRY.keys())
