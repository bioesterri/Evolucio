"""Public tracking errors with messages safe to persist and display."""


class MlflowSyncError(RuntimeError):
    """Tracking could not be synchronized."""


class MlflowUnavailableError(MlflowSyncError):
    """The tracking client or backend is unavailable."""


class MlflowExperimentBindingError(MlflowSyncError):
    """An experiment could not be bound."""


class MlflowRunBindingError(MlflowSyncError):
    """A run could not be bound."""


class MlflowTrackingIntegrityError(MlflowSyncError):
    """Canonical data and tracking data disagree."""


class MlflowInputError(MlflowSyncError):
    """The input run cannot be used for tracking operations."""
