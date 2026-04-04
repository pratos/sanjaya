"""RLM abstract base class defining the interface for all RLM implementations."""

from abc import ABC, abstractmethod
from typing import Any


class RLM(ABC):
    """Abstract base class for Recursive Language Model implementations."""

    @abstractmethod
    def completion(self, context: Any, query: str = "") -> str:
        """Execute a completion request with the given context and query."""
        pass

    @abstractmethod
    def cost_summary(self) -> dict[str, float]:
        """Retrieve cost summary for API usage."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the internal state of the RLM instance."""
        pass
