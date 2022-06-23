#!/usr/bin/env python
# Copyright 2021, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md
"""Unit tests for I/O reading utilities."""

import datetime as dt
import logging
from glob import glob
import os
import numpy as np
import pytest
import sys
import tempfile

import aetherpy
from aetherpy.io import read_routines


class TestIORead(object):
    """Unit tests for file reading routines."""

    def setup(self):
        """Initialize clean test environment."""
        self.test_dir = os.path.join(os.path.split(aetherpy.__file__)[0],
                                     "tests", "test_data")
        self.header = {}
        return

    def teardown(self):
        """Clean up the test environment."""
        del self.test_dir, self.header
        return

    def eval_header(self, file_list=True):
        """Evaluate header output.

        Parameters
        ----------
        file_list : bool
           'filename' key is a list if True and a string if False (default=True)

        """

        # Ensure the base keys are present, more keys are allowed
        base_keys = ['vars', 'time', 'filename', 'nlons', 'nlats', 'nalts']

        assert np.all([bkey in self.header.keys() for bkey in base_keys]), \
            "missing required keys from header output."

        # Evalute the header output formats and types
        for hkey in self.header.keys():
            if hkey in ['version']:
                # Ensure floats are the correct type
                assert isinstance(self.header[hkey], float)
            elif hkey in base_keys[:3]:
                # Ensure lists are the correct type, length, and have the
                # expected value types.
                if file_list or hkey == 'vars':
                    assert isinstance(self.header[hkey], list)

                if hkey == 'time':
                    if file_list:
                        assert len(self.header[hkey]) <= len(
                            self.header['filename']), \
                            "header times cannot be more than number of files"
                    else:
                        self.header[hkey] = [self.header[hkey]]

                    for htime in self.header[hkey]:
                        assert isinstance(htime, dt.datetime), \
                            "unexpected time format for: {:} from {:}".format(
                                htime, self.header[hkey])
                elif hkey == 'filename':
                    if file_list:
                        assert len(self.header[hkey]) > 0
                    else:
                        self.header[hkey] = [self.header[hkey]]

                    for fname in self.header[hkey]:
                        assert os.path.isfile(fname), \
                            "header filename {:} is not a file".format(fname)
                else:
                    assert np.all(np.unique(self.header[hkey])
                                  == self.header[hkey]), \
                        "duplicate variables in header list"

                    for var in self.header[hkey]:
                        assert isinstance(var, str), \
                            "variable name {:} is not a string".format(var)
            elif hkey in ['nlons', 'nlats', 'nalts'] or hkey[0] == 'n':
                # Ensure the counters are all integers
                assert isinstance(self.header[hkey], int), \
                    "counter {:} is not an int: {:}".format(hkey,
                                                            self.header[hkey])
            elif hkey == 'filename':
                # The filename is a string and not a list
                assert os.path.isfile(self.header[hkey])
        return

    @pytest.mark.parametrize("in_line, out_int",
                             [("1 2 3 4", 1),
                              ("42 NEU farm SpRiNg 45.0", 42)])
    @pytest.mark.parametrize("parse", [True, False])
    def test_parse_line_into_int_and_string(self, in_line, out_int, parse):
        """Test successful int/string line parsing.

        Parameters
        ----------
        in_line : str
            Input line with correct formatting
        out_int : int
            Expected integer output
        parse : bool
            `parse_string` kwarg input

        """

        # Parse the line
        lnum, lstr = read_routines.parse_line_into_int_and_string(in_line,
                                                                  parse)

        # Evaluate the retrieved values
        assert lnum == out_int, "unexpected integer value for first column"

        if parse:
            assert in_line.find(lstr) > 0, "unexpected line values"
        else:
            assert in_line == lstr, "unexpected line values"
        return

    @pytest.mark.parametrize("in_line",
                             ["1.0 2 3 4", "NEU farm SpRiNg 45.0"])
    def test_parse_line_into_int_and_string_bad_format(self, in_line):
        """Test successful int/string line parsing.

        Parameters
        ----------
        in_line : str
            Input line with incorrect formatting

        """

        # Parse the line and check error output
        with pytest.raises(ValueError) as verr:
            read_routines.parse_line_into_int_and_string(in_line)

        # Evaluate the retrieved values
        assert str(verr).find("invalid literal for int") >= 0
        return

    @pytest.mark.parametrize('fname', ['3DALL_20110320_003000_g0001.nc',
                                       '3DBFI_20110320_000000_g0000.nc'])
    def test_read_aether_netcdf_header(self, fname):
        """Test successful Aether netCDF header reading.

        Parameters
        ----------
        fname : str
            File base name

        """
        filename = os.path.join(self.test_dir, fname)
        assert os.path.isfile(filename), "missing test file: {:}".format(
            filename)

        self.header = read_routines.read_aether_netcdf_header(filename)
        self.eval_header(file_list=False)
        return

    def test_read_aether_netcdf_header_bad_file(self):
        """Test raises IOError with bad filename."""

        with pytest.raises(IOError) as verr:
            read_routines.read_aether_netcdf_header("not_a_file")

        assert str(verr).find("unknown aether netCDF file") >= 0
        return

    @pytest.mark.parametrize('fbase', ['3DALL_*.nc', '3DBFI_*.nc'])
    @pytest.mark.parametrize('ftype', ['netcdf'])
    @pytest.mark.parametrize('finds, is_single', [(-1, True), (None, True),
                                                  (-1, False), (None, False),
                                                  ([0, 1], False),
                                                  (slice(1), True),
                                                  (slice(1), False)])
    def test_read_aether_headers(self, fbase, ftype, finds, is_single):
        """Test successful Aether header reading.

        Parameters
        ----------
        fbase : str
            File base name, with glob support
        ftype : str
            File type
        finds : int, NoneType, list, slice
            File indexers
        is_single : bool
            Uses a single file as a string input if True, or a list if False

        """
        filenames = glob(os.path.join(self.test_dir, fbase))

        if is_single:
            filenames = filenames[0]

        self.header = read_routines.read_aether_headers(filenames, finds,
                                                        ftype)

        self.eval_header(file_list=True)
        return

    @pytest.mark.parametrize('fname', ['3DALL_20110320_003000_g0001.nc',
                                       '3DBFI_20110320_000000_g0000.nc'])
    @pytest.mark.parametrize('fvars', [None, (['z'])])
    def test_read_aether_file_dict(self, fname, fvars):
        """Test successful Aether NetCDF data reading into dict output.

        Parameters
        ----------
        fname : str
            File base name
        fvars : NoneType or list
            List of variable names to read in or NoneType to read all

        """
        filename = os.path.join(self.test_dir, fname)
        assert os.path.isfile(filename), "missing test file: {:}".format(
            filename)

        data = read_routines.read_aether_file(filename, file_vars=fvars)

        # Evaluate the output keys
        assert isinstance(data, dict)
        assert 'time' in data.keys(), "'time' missing from output"
        assert 'units' in data.keys(), "'units' missing from output"
        assert 'long_name' in data.keys(), "'long_name' missing from output"
        assert 'vars' in data.keys(), "'vars' missing from output"

        # Evaluate the data variables and attributes
        for i in range(len(data['vars'])):
            assert i in data.keys(), \
                'missing data index {:d} in output'.format(i)
            assert data[i].dtype in [np.float16, np.float32, np.float64]
            assert i < len(data['units'])
            assert i < len(data['long_name'])
            assert isinstance(data['units'][i], str)
            assert isinstance(data['long_name'][i], str)

        return
