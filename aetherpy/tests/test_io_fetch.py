#!/usr/bin/env python
# Copyright 2021, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md
"""Unit tests for I/O fetch utilities."""

import logging
import os
import pytest
import sys
import tempfile

from aetherpy.io import fetch_routines
from aetherpy.tests.utils import sys_agnostic_remove


class TestLocalFetch(object):
    """Unit tests for local file fetching routines."""

    def setup(self):
        """Initialize clean test environment."""
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # TODO #9: remove if-statement when it is always triggered
        tkwargs = {}
        if sys.version_info.major >= 3 and sys.version_info.minor >= 10:
            tkwargs = {"ignore_cleanup_errors": True}
        self.tempdir = tempfile.TemporaryDirectory(**tkwargs)
        self.tempfiles = []
        self.file_type = 'ALL'
        self.file_ext = 'bin'

        return

    def teardown(self):
        """Clean up the test environment."""
        # Remove the created files and directories
        for filename in self.tempfiles:
            sys_agnostic_remove(filename)

        # Remove the temporary directory
        # TODO #9: Remove try/except when Python 3.10 is the lowest version
        try:
            self.tempdir.cleanup()
        except Exception:
            pass

        # Clear the test environment attributes
        del self.file_type, self.file_ext, self.tempdir, self.tempfiles
        return

    def make_files(self, dims=1):
        """Create four temporary files in an existing directory.

        Parameters
        ----------
        dims : int
            Number of dimensions (expects 1 or 3)

        """

        # Create the file base
        filebase = "{:d}D{:s}".format(dims, self.file_type)
        fileext = ".{:s}".format(self.file_ext)

        # Create an empty temporary file and save the filename
        for i in range(4):
            out = tempfile.mkstemp(suffix=fileext, prefix=filebase,
                                   dir=self.tempdir.name)
            self.tempfiles.append(out[1])

        return

    @pytest.mark.parametrize("ftype", ["ALL", "ION", "NEU"])
    @pytest.mark.parametrize("ext", ["bin", "nc"])
    @pytest.mark.parametrize("dims", [1, 3])
    def test_get_filelist(self, ftype, ext, dims):
        """Test successful retrieval of local files for standard types.

        Parameters
        ----------
        ftype : str
            Desired file type, accepts 'ALL', 'ION', and 'NEU'
        ext : str
            File extenstion, without period
        dims : int
            Number of dimensions, accepts 1 or 3

        """

        # Create temporary files
        self.file_type = ftype
        self.file_ext = ext
        self.make_files(dims=dims)

        # Get the list of files
        filelist = fetch_routines.get_filelist(self.tempdir.name,
                                               file_type=ftype, file_ext=ext)

        # Evaluate the retrieved list
        assert len(filelist) == len(self.tempfiles)

        for fname in filelist:
            assert fname in self.tempfiles

        return

    def test_get_filelist_from_multidim_set(self):
        """Test retrieval of 3D local files from dir with 1D and 3D."""

        # Create temporary 1D and 3D files
        self.make_files(dims=1)
        self.make_files(dims=3)

        # Get the list of files
        filelist = fetch_routines.get_filelist(self.tempdir.name,
                                               file_type=self.file_type,
                                               file_ext=self.file_ext)

        # Evaluate the retrieved list
        assert len(filelist) == len(self.tempfiles) / 2

        for fname in filelist:
            assert fname in self.tempfiles

        return

    def test_get_filelist_from_unknown_dim_set(self, caplog):
        """Test unsuccessful retrieval of local files with 2D and 4D files."""

        # Create temporary 2D and 4D files
        self.make_files(dims=2)
        self.make_files(dims=4)

        # Get the list of files
        with caplog.at_level(logging.INFO, logger="aetherpy_logger"):
            filelist = fetch_routines.get_filelist(self.tempdir.name,
                                                   file_type=self.file_type,
                                                   file_ext=self.file_ext)

        # Evaluate the retrieved list
        assert len(filelist) == 0

        # Evaluate the logger output
        ordered_msgs = ["No 3D", "No 1D"]
        ordered_lvl = ['INFO', 'WARNING']
        assert len(caplog.records) == len(ordered_msgs)

        for i, record in enumerate(caplog.records):
            # Evaluate the logging message
            assert record.message.find(ordered_msgs[i]) >= 0, \
                "unexpected log output: {:s}".format(record.message)

            # Evaluate the logging output level
            assert record.levelname == ordered_lvl[i]

        return

    def test_get_filelist_from_unknown_file_type(self, caplog):
        """Test retrieval of local files with unknown file type."""

        # Create temporary 3D files
        self.file_type = "LOCAL"
        self.make_files(dims=3)

        # Get the list of files
        with caplog.at_level(logging.WARNING, logger="aetherpy"):
            filelist = fetch_routines.get_filelist(self.tempdir.name,
                                                   file_type=self.file_type,
                                                   file_ext=self.file_ext)

        # Evaluate the retrieved list
        assert len(filelist) == len(self.tempfiles)

        for fname in filelist:
            assert fname in self.tempfiles

        # Evaluate the logger output
        captured = caplog.text
        assert captured.find("unexpected file type") >= 0, \
            "unexpected log output: {:s}".format(captured)

        return

    def test_get_filelist_from_bad_glob_str(self, caplog):
        """Test unsuccessful retrieval of local files with bad glob string."""

        # Create temporary 3D files
        self.make_files(dims=3)

        # Create a bad glob string
        glob_str = os.path.join(self.tempdir.name, "not_a_file*")

        # Get the list of files
        with caplog.at_level(logging.WARNING, logger="aetherpy"):
            filelist = fetch_routines.get_filelist(glob_str)

        # Evaluate the retrieved list
        assert len(filelist) == 0

        # Evaluate the logger output
        captured = caplog.text
        assert captured.find("No files found using search string") >= 0, \
            "unexpected log output: {:s}".format(captured)

        return

    def test_get_filelist_from_glob_str(self):
        """Test successful retrieval of local files with a good glob string."""

        # Create temporary 3D files
        self.make_files(dims=3)

        # Create a good glob string
        glob_str = os.path.join(self.tempdir.name, "3D{:s}*.{:s}".format(
            self.file_type, self.file_ext))

        # Get the list of files
        filelist = fetch_routines.get_filelist(glob_str)

        # Evaluate the retrieved list
        assert len(filelist) == len(self.tempfiles)

        for fname in filelist:
            assert fname in self.tempfiles

        return
