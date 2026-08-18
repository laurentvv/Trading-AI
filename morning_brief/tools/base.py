"""Base tool definition for morning brief analysis tools without external agent frameworks."""

from __future__ import annotations
from typing import Any, Dict


class BaseTool:
    name: str = ""
    description: str = ""
    inputs: Dict[str, Any] = {}
    output_type: str = "string"

    def forward(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return self.forward(*args, **kwargs)


# Backward compatibility alias
Tool = BaseTool
