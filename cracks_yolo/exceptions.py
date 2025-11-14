from __future__ import annotations

import typing as t


class CracksYoloError(Exception):
    """Base exception for Cracks YOLO related errors."""

    default_message: t.ClassVar[str] = "An error occurred in Cracks YOLO."
    code: t.ClassVar[int] = 1

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(message)

    def __str__(self) -> str:
        return f"[Error {self.code}] {self.message}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code}, message={self.message!r})"


class RequiredModuleNotFoundError(CracksYoloError):
    """Exception raised when a required module is not found."""

    default_message = "Required module {module_name} not found. Please install it to proceed."
    code: t.ClassVar[int] = 2

    def __init__(self, module_name: str, message: str | None = None) -> None:
        self.module_name = module_name
        full_message = message or self.default_message.format(module_name=module_name)
        super().__init__(full_message)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(module_name={self.module_name!r}, message={self.message!r})"
        )
