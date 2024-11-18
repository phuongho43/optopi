import contextlib
import os


def makedirs(dp):
    """Create directories as needed for the desired directory path.

    Args:
        dp (str): directory path desired
    """
    with contextlib.suppress(OSError):
        os.makedirs(dp)
