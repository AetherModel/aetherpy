#!/usr/bin/env python
# Copyright 2021, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md
"""Utilities for test classes."""

import os


def sys_agnostic_rename(inname, outname, max_attemps=100):
    """Wrap os.rename, Windows OS sometimes needs time to allow rename to work.

    Parameters
    ----------
    inname : str
        Input filename or directory
    outname : str
        Output filename or directory
    max_attemps : int
        Maximum rename attemps (default=100)

    """
    for retry in range(max_attemps):
        try:
            os.rename(inname, outname)
            break
        except Exception:
            pass

    return


def sys_agnostic_remove(fname, max_attemps=100):
    """Wrap os.remove, Windows OS sometimes needs time to allow remove to work.

    Parameters
    ----------
    fname : str
        Filename to be removed
    max_attemps : int
        Maximum rename attemps (default=100)

    """
    for retry in range(max_attemps):
        if os.path.isfile(fname):
            try:
                os.remove(fname)
                break
            except Exception:
                pass

    return
