"""Domain errors mapped to HTTP status codes by the API layer."""


class VerifAIError(Exception):
    http_status = 500
    code = "internal_error"

    def __init__(self, message: str, code: str = ""):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class InvalidMediaError(VerifAIError):
    """Corrupt, unsupported or out-of-limits media (HTTP 422)."""
    http_status = 422
    code = "invalid_media"


class PayloadTooLargeError(VerifAIError):
    http_status = 413
    code = "payload_too_large"


class UnsupportedMediaTypeError(VerifAIError):
    http_status = 415
    code = "unsupported_media_type"


class NotFoundError(VerifAIError):
    http_status = 404
    code = "not_found"


class JobCancelled(VerifAIError):
    http_status = 409
    code = "cancelled"


class ModelUnavailableError(VerifAIError):
    http_status = 503
    code = "model_unavailable"
