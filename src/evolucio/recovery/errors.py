"""Typed errors at recovery boundaries."""


class RecoveryError(ValueError):
    """Recovery cannot proceed."""


class CheckpointNotRecoverableError(RecoveryError):
    pass


class HistoryIntegrityError(RecoveryError):
    pass


class RunStillActiveError(RecoveryError):
    pass


class RecoveryPlanError(RecoveryError):
    pass


class RecoveryPersistenceError(RecoveryError):
    pass
