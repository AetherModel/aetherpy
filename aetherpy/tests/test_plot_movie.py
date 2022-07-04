#!/usr/bin/env python
# Copyright 2021, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md
"""Unit tests for plot-to-movie utilities."""

import logging
import numpy as np
import os
import pytest
import sys
import tempfile

from aetherpy.plot import movie_routines as mr
from aetherpy.tests.utils import sys_agnostic_remove
from aetherpy.tests.utils import sys_agnostic_rename


class TestMovie(object):
    """Unit tests for plot-to-movie functions."""

    def setup(self):
        """Initialize clean test environment."""

        # Create a temporary directory
        tkwargs = {}
        # TODO #9: remove if-statement when it is always triggered
        if sys.version_info.major >= 3 and sys.version_info.minor >= 10:
            tkwargs = {"ignore_cleanup_errors": True}
        self.tempdir = tempfile.TemporaryDirectory(**tkwargs)
        self.movie_dir = os.path.join(self.tempdir.name, "movie_dir")
        self.fileext = ".png"
        self.filebase = "test_"
        self.tempfiles = []
        self.moviename = "test.mp4"

        return

    def teardown(self):
        """Clean up the test environment."""
        # Remove the created files and directories
        for filename in self.tempfiles:
            sys_agnostic_remove(filename)

        # TODO #9: Remove try/except when Python 3.10 is the lowest version
        if os.path.isdir(self.movie_dir):
            try:
                os.rmdir(self.movie_dir)
            except Exception:
                pass

        # Remove the temporary directory
        # TODO #9: Remove try/except when Python 3.10 is the lowest version
        try:
            self.tempdir.cleanup()
        except Exception:
            pass

        # Clear the test environment attributes
        del self.movie_dir, self.tempdir, self.tempfiles, self.moviename
        del self.fileext, self.filebase
        return

    def make_files(self, data=False):
        """Create a directory with temporary files.

        Parameters
        ----------
        data : bool
            Add data to the temporary file
        """
        os.makedirs(self.movie_dir)

        for i in range(4):
            # Create a temporary file that must be removed
            out = tempfile.mkstemp(suffix=self.fileext, prefix=self.filebase,
                                   dir=self.movie_dir)

            # Rename the temporary file to match the necessary format
            goodname = os.path.join(self.movie_dir, "{:s}{:04d}{:s}".format(
                self.filebase, i, self.fileext))

            # Windows OS sometimes needs time to allow a rename
            sys_agnostic_rename(out[1], goodname)

            # Add data to the temporary file
            if data:
                with open(goodname, "wb") as fout:
                    fout.write(b"AAAn")

            # Save the good filename
            self.tempfiles.append(goodname)

        return

    def test_setup_movie_dir_newdir(self):
        """Test sucessful creation of a new directory for movie files."""
        assert not os.path.isdir(self.movie_dir)
        mr.setup_movie_dir(self.movie_dir)
        assert os.path.isdir(self.movie_dir)

        return

    @pytest.mark.skipif(sys.platform in ["win32", "cygwin", "windows"],
                        reason="Windows can't remove directories")
    @pytest.mark.parametrize("wcard", ["*", "????"])
    def test_setup_movie_dir_olddir(self, wcard):
        """Test sucessful creation of a new directory for movie files.

        Parameters
        ----------
        wcard : str
            Accepted wildcard strings for the test files

        """
        self.make_files()
        file_glob = "".join([self.filebase, wcard, self.fileext])

        img_names = mr.setup_movie_dir(self.movie_dir, file_glob=file_glob)
        assert os.path.isdir(self.movie_dir)
        for filename in self.tempfiles:
            assert not os.path.isfile(filename), "old file not removed"

        assert img_names.find(self.movie_dir) >= 0, "unexpected dest directory"
        assert img_names.find(self.filebase) >= 0, "unexpected file prefix"
        assert img_names.find(self.fileext) >= 0, "unexpected file extension"
        return

    def test_setup_movie_dir_no_overwrite(self):
        """Test raises IOError when conflicting files are present."""
        self.make_files()
        file_glob = "*".join([self.filebase, self.fileext])

        with pytest.raises(IOError) as ierr:
            mr.setup_movie_dir(self.movie_dir, file_glob=file_glob,
                               overwrite=False)

        assert str(ierr).find("files present in movie directory") >= 0
        return

    @pytest.mark.parametrize("rate", [30, 60])
    def test_save_movie_nooverwrite_success(self, rate):
        """Test the creation of a movie file in a clean directory.

        Parameters
        ----------
        rate : int
            Frame rate

        """
        # Set up the movie file directory
        self.make_files(data=True)
        image_files = os.path.join(self.movie_dir, "".join([
            self.filebase, "%04d", self.fileext]))

        # Create the movie file
        outfile = mr.save_movie(self.movie_dir, movie_name=self.moviename,
                                image_files=image_files, rate=rate)

        # Test the output
        assert os.path.isfile(outfile), "movie file not created"

        # Prepare for cleanup
        self.tempfiles.append(outfile)
        return

    @pytest.mark.skipif(sys.platform in ['win32', 'cygwin', 'windows'],
                        reason="Windows can't remove directories")
    def test_save_movie_overwrite_success(self):
        """Test the creation of a movie file when overwriting the old movie."""
        # Set up the movie file directory
        self.make_files(data=True)
        image_files = os.path.join(self.movie_dir, "".join([
            self.filebase, "%04d", self.fileext]))

        # Make an output file with the same name as the movie file
        out = tempfile.mkstemp(suffix=self.fileext, prefix=self.filebase,
                               dir=self.movie_dir)
        sys_agnostic_rename(out[1],
                            os.path.join(self.movie_dir, self.moviename))

        # Create the move file
        outfile = mr.save_movie(self.movie_dir, movie_name=self.moviename,
                                image_files=image_files, overwrite=True)

        # Test the output
        assert os.path.isfile(outfile), "movie file not created"

        # Prepare for cleanup
        self.tempfiles.append(outfile)
        return

    @pytest.mark.skipif(sys.platform in ['win32', 'cygwin', 'windows'],
                        reason="Windows create the test file")
    def test_save_movie_overwrite_blocking(self):
        """Test raises IOError when a movie exists and overwrite is blocked."""
        # Set up the movie file directory
        self.make_files(data=True)
        image_files = os.path.join(self.movie_dir, "".join([
            self.filebase, "%04d", self.fileext]))

        # Make an output file with the same name as the movie file
        out = tempfile.mkstemp(suffix=self.fileext, prefix=self.filebase,
                               dir=self.movie_dir)
        outfile = os.path.join(self.movie_dir, self.moviename)
        sys_agnostic_rename(out[1], outfile)
        assert os.path.isfile(outfile), \
            "Unable to create file {:} on {:}".format(outfile, sys.platform)

        # Create the move file
        with pytest.raises(IOError) as ierr:
            mr.save_movie(self.movie_dir, movie_name=self.moviename,
                          image_files=image_files, overwrite=False)

        # Test the output
        assert str(ierr).find('already exists') >= 0, "unexpected IOError"

        # Prepare for cleanup
        self.tempfiles.append(outfile)
        return
