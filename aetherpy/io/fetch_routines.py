#!/usr/bin/env python
# Copyright 2020, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md
"""Routines to find and retrieve files."""

from glob import glob
import os

from aetherpy import logger


def get_filelist(file_dir, file_type='ALL', file_ext="bin"):
    """Get a list of Aether-style model filenames from a specified directory.

    Parameters
    ----------
    file_dir : str
        File directory to search or a glob search string
    file_type : str
        Desired file type, accepts 'ALL', 'ION', and 'NEU' (default='ALL')
    file_ext : str
        File extenstion, without period (default='bin')

    Returns
    -------
    filelist : list
        List of Aether-style model filenames

    Notes
    -----
    Only retrieves 1D files if 3D files are not present

    """

    # Check the file type and warn user for untested use cases
    file_type = file_type.upper()
    if file_type not in ['ALL', 'ION', 'NEU']:
        logger.warning(''.join(['unexpected file type [', file_type,
                                '], routine may not work']))

    # Check to see if the directory exists
    if os.path.isdir(file_dir):
        # Get a list of 3D files or 1D files
        filelist = glob(os.path.join(
            file_dir, '3D{:s}*.{:s}'.format(file_type, file_ext)))

        if len(filelist) == 0:
            logger.info("".join(["No 3D", file_type, " files found in ",
                                 file_dir, ", checking for 1D", file_type]))
            filelist = glob(os.path.join(
                file_dir, '1D{:s}*.{:s}'.format(file_type, file_ext)))

            if len(filelist) == 0:
                logger.warning('No 1D{:s} files found in {:s}'.format(
                    file_type, file_dir))
    else:
        # This may be a glob search string
        filelist = glob(file_dir)

        if len(filelist) == 0:
            logger.warning('No files found using search string: {:s}'.format(
                file_dir))

    return filelist
