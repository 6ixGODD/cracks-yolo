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


class DatasetError(CracksYoloError):
    """Exception raised for dataset-related errors."""

    default_message = "Dataset operation failed."
    code: t.ClassVar[int] = 10


class DatasetNotFoundError(DatasetError):
    """Exception raised when a dataset is not found."""

    default_message = "Dataset not found at path: {path}"
    code: t.ClassVar[int] = 11

    def __init__(self, path: str, message: str | None = None) -> None:
        self.path = path
        full_message = message or self.default_message.format(path=path)
        super().__init__(full_message)


class DatasetFormatError(DatasetError):
    """Exception raised for invalid dataset format."""

    default_message = "Invalid dataset format: {format}"
    code: t.ClassVar[int] = 12

    def __init__(self, format: str, message: str | None = None) -> None:
        self.format = format
        full_message = message or self.default_message.format(format=format)
        super().__init__(full_message)


class DatasetLoadError(DatasetError):
    """Exception raised when dataset loading fails."""

    default_message = "Failed to load dataset: {reason}"
    code: t.ClassVar[int] = 13

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        full_message = message or self.default_message.format(reason=reason)
        super().__init__(full_message)


class DatasetExportError(DatasetError):
    """Exception raised when dataset export fails."""

    default_message = "Failed to export dataset: {reason}"
    code: t.ClassVar[int] = 14

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        full_message = message or self.default_message.format(reason=reason)
        super().__init__(full_message)


class DatasetMergeError(DatasetError):
    """Exception raised when dataset merge fails."""

    default_message = "Failed to merge datasets: {reason}"
    code: t.ClassVar[int] = 15

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        full_message = message or self.default_message.format(reason=reason)
        super().__init__(full_message)


class CategoryConflictError(DatasetError):
    """Exception raised when category conflicts occur."""

    default_message = "Category conflict detected: {category_name}"
    code: t.ClassVar[int] = 16

    def __init__(self, category_name: str, message: str | None = None) -> None:
        self.category_name = category_name
        full_message = message or self.default_message.format(category_name=category_name)
        super().__init__(full_message)


class ValidationError(CracksYoloError):
    """Exception raised for validation errors."""

    default_message = "Validation failed: {reason}"
    code: t.ClassVar[int] = 20

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        full_message = message or self.default_message.format(reason=reason)
        super().__init__(full_message)
