"""Application errors that map cleanly onto HTTP responses."""

from __future__ import annotations


class DirectoryError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details = list(details or [])

    def to_dict(self) -> dict:
        payload = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class InputError(DirectoryError):
    status_code = 400
    code = "invalid_input"


class NotFoundError(DirectoryError):
    status_code = 404
    code = "not_found"


class ConflictError(DirectoryError):
    status_code = 409
    code = "conflict"


class RestoreInProgress(ConflictError):
    status_code = 409
    code = "restore_in_progress"

    def __init__(self, message: str = "备份恢复进行中，新的修改已被拒绝，请稍后重试") -> None:
        super().__init__(message)


class BackupNameTaken(ConflictError):
    status_code = 409
    code = "backup_name_taken"


class BackupValidationError(DirectoryError):
    status_code = 422
    code = "invalid_backup"

    def __init__(self, message: str, *, details: list[str] | None = None) -> None:
        super().__init__(message, details=details)


class DatabaseBusy(DirectoryError):
    status_code = 503
    code = "database_busy"

    def __init__(self) -> None:
        super().__init__("数据库繁忙，请稍后重试")


class StorageFull(DirectoryError):
    """The database hit its page cap (or the disk is full).

    This is deliberately distinct from :class:`DatabaseError`: the request was
    valid, but committing it would exceed the configured page limit.  The
    rejected transaction is rolled back, so the previously committed directory
    and all named backups are intact and the same request can be retried
    unchanged once space has been reclaimed.
    """

    status_code = 507
    code = "storage_full"

    def __init__(
        self,
        message: str = (
            "存储空间不足：数据库已达到页数上限，本次修改未保存。"
            "请释放历史空间或提高页数上限后，使用原内容重试。"
        ),
        *,
        details: list[str] | None = None,
    ) -> None:
        super().__init__(message, details=details)


class DatabaseError(DirectoryError):
    status_code = 500
    code = "database_error"

    def __init__(self, message: str = "数据库操作失败") -> None:
        super().__init__(message)
