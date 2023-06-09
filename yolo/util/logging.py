#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Logging."""

import builtins
import logging
import os
import sys
from typing import Any, Optional, Dict


def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


def setup_logging(local_rank, output_dir=None):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    if local_rank == 0:
        # Enable logging for the master process.
        logging.root.handlers = []
    else:
        # Suppress logging for non-master processes.
        _suppress_print()
        return EmptyLogger('ignore')

    level = logging.INFO if local_rank == 0 else logging.ERROR

    logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(level)
    logger.propagate = False
    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)3d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    # ch.setLevel(logging.DEBUG)
    ch.setLevel(level)
    ch.setFormatter(plain_formatter)
    logger.addHandler(ch)

    # if output_dir is not None and du.is_master_proc():
    if output_dir is not None and local_rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, "stdout.log")
        fh = logging.FileHandler(filename)
        # fh.setLevel(logging.DEBUG)
        fh.setLevel(level)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


class EmptyLogger(logging.Logger):

    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        pass

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        pass

    def warn(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        pass

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        pass
