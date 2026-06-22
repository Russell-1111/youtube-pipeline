class PipelineError(Exception):
    """Base class for expected pipeline failures."""


class ConfigError(PipelineError):
    """Raised when the YAML config is missing or invalid."""


class InputFileError(PipelineError):
    """Raised when required input files are missing."""


class SRTParseError(PipelineError):
    """Raised when the SRT transcript is malformed."""


class TimingError(PipelineError):
    """Raised when transcript and audio timing cannot be reconciled."""
