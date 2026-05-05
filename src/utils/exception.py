"""
exception.py
============
Custom exception class that enriches error messages with the exact
source file and line number where the exception was raised.

Mirrors the pattern used in the reference repo.

Usage
-----
    from src.utils.exception import LeachingException
    import sys

    try:
        risky_operation()
    except Exception as e:
        raise LeachingException(e, sys) from e
"""

import sys


def _get_error_message(error: Exception, error_detail: "module") -> str:
    """
    Build a descriptive error string including file path and line number.

    Parameters
    ----------
    error        : the caught exception
    error_detail : the ``sys`` module (passed by caller)
    """
    _, _, traceback = error_detail.exc_info()

    # Walk to the innermost frame (where the error actually occurred)
    while traceback.tb_next is not None:
        traceback = traceback.tb_next

    file_name   = traceback.tb_frame.f_code.co_filename
    line_number = traceback.tb_lineno

    return (
        f"\n"
        f"  File : {file_name}\n"
        f"  Line : {line_number}\n"
        f"  Error: {type(error).__name__}: {error}"
    )


class LeachingException(Exception):
    """
    Domain-specific exception for the leaching ML pipeline.

    Wraps any caught exception and attaches file + line context
    so debugging is faster.
    """

    def __init__(self, error: Exception, error_detail: "module"):
        super().__init__(str(error))
        self.error_message = _get_error_message(error, error_detail)

    def __str__(self) -> str:
        return self.error_message
