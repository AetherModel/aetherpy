#!/usr/bin/env python
# Copyright 2021, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md
"""Unit tests for input utilities."""

import pytest
import sys

from aetherpy.utils import inputs as input_utils


class TestInputs(object):
    """Unit tests for input utilities."""

    @pytest.mark.parametrize("in_str, out_val", [
        ("true", True), ("false", False), ("TRUE", True), ("FALSE", False),
        ("T", True), ("F", False), ("t", True), ("f", False), ("1", True),
        ("0", False), ("", True), ("True", True), ("False", False),
        ("tRuE", True), ("fAlSe", False)])
    def test_bool_string_success(self, in_str, out_val):
        """Test conversion of a string to a Boolean.

        Parameters
        ----------
        in_str : str
            Input string with acceptable boolean-interpretable values
        out_val : bool
            Expected Boolean output

        """

        out = input_utils.bool_string(in_str)

        assert out == out_val
        return

    @pytest.mark.parametrize("in_val", ["Tru", "Fals", "3", "1.0", "0.0"])
    def test_bool_string_failure(self, in_val):
        """Test conversion from datetime to epoch seconds.

        Parameters
        ----------
        in_val : str
            Input value that will not be recognized as a string that may be
            converted to a boolean

        """

        with pytest.raises(ValueError) as verr:
            input_utils.bool_string(in_val)

        assert str(verr).find("input not interpretable as a boolean") >= 0
        return

    @pytest.mark.parametrize("in_str", ["none", "None", "nOne", ""])
    def test_none_string_success(self, in_str):
        """Test conversion of a string to NoneType.

        Parameters
        ----------
        in_str : str
            Input string with acceptable NoneType conversion values

        """

        assert input_utils.none_string(in_str) is None
        return

    @pytest.mark.parametrize("in_str", ["non", "Non", "nOn", " "])
    def test_none_string_failure(self, in_str):
        """Test conversion of a string to NoneType.

        Parameters
        ----------
        in_str : str
            Input string with unacceptable NoneType conversion values

        """

        assert input_utils.none_string(in_str) == in_str
        return

    @pytest.mark.parametrize("in_args, num_args", [(["ipython", "one"], 0),
                                                   (["module_name", "one"], 1)])
    def test_process_command_line_input(self, in_args, num_args):
        """Test `process_command_line_input` using the command line."""

        # Set and process the system arguments
        sys.argv = in_args
        out_args = input_utils.process_command_line_input()

        # Test the processed output
        assert len(out_args) == num_args
        for i, in_val in enumerate(in_args[len(in_args) - num_args:]):
            assert out_args[i] == in_val

        return
