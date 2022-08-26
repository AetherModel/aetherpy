#!/usr/bin/env python
# Copyright 2021, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md
"""Unit tests for time conversion utilities."""

import datetime as dt
import numpy as np
import pytest

from aetherpy.utils import time_conversion as tc


class TestEpochTime(object):
    """Unit tests for epoch time conversions."""

    @pytest.mark.parametrize("esec, dtime", [
        (-10, dt.datetime(1964, 12, 31, 23, 59, 50)),
        (0, dt.datetime(1965, 1, 1)),
        (1000.5, dt.datetime(1965, 1, 1, 0, 16, 40, 500000))])
    def test_epoch_to_datetime(self, esec, dtime):
        """Test conversion from epoch seconds to datetime.

        Parameters
        ----------
        esec : int or float
            Input epoch seconds
        dtime : dt.datetime
            Output datetime object

        """

        out_time = tc.epoch_to_datetime(esec)

        assert isinstance(out_time, dt.datetime)
        assert out_time == dtime
        return

    @pytest.mark.parametrize("dtime, esec", [
        (dt.datetime(1950, 1, 1), -473385600.0),
        (dt.datetime(1965, 1, 1), 0.0),
        (dt.datetime(2010, 2, 3, 4, 5, 6, 7), 1422936306.000007)])
    def test_datetime_to_epoch(self, dtime, esec):
        """Test conversion from datetime to epoch seconds.

        Parameters
        ----------
        dtime : dt.datetime
            Input datetime object
        esec : float
            Output epoch seconds

        """

        out_time = tc.datetime_to_epoch(dtime)

        assert isinstance(out_time, float)
        assert out_time == esec
        return


class TestUT(object):
    """Unit tests for conversions based around UT."""

    def setup(self):
        """Create a clean testing environment."""

        self.times = [dt.datetime(2001, 1, 1) + dt.timedelta(seconds=i * 900)
                      for i in range(96)]
        self.lon = 180.0
        self.lt = [12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0,
                   14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75, 16.0, 16.25,
                   16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5,
                   18.75, 19.0, 19.25, 19.5, 19.75, 20.0, 20.25, 20.5, 20.75,
                   21.0, 21.25, 21.5, 21.75, 22.0, 22.25, 22.5, 22.75, 23.0,
                   23.25, 23.5, 23.75, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
                   1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25,
                   4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0,
                   7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75,
                   10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75]
        return

    def teardown(self):
        """Clean up the test environment."""

        del self.times, self.lon, self.lt

    def update_lon(self, multi_lon):
        """Update the `lon` attribute.

        Parameters
        ----------
        multi_lon : bool
           Include a single longitude value or an array-like object

        """

        if multi_lon:
            self.lon = np.full(shape=len(self.times), fill_value=self.lon)
        return

    @pytest.mark.parametrize("lon_sign", [1, -1])
    @pytest.mark.parametrize("multi_lon", [True, False])
    def test_ut_to_lt(self, lon_sign, multi_lon):
        """Test the datetime in UT to solar local time conversion.

        Parameters
        ----------
        lon_sign : int
            One or negative one, specifies whether longitude should be
            positive or negative
        multi_lon : bool
           Include a single longitude value or an array-like object

        """

        # Update the longitude
        self.update_lon(multi_lon)
        self.lon *= lon_sign

        # Get the solar local time
        out = tc.ut_to_lt(self.times, self.lon)

        # Test the time output
        assert out.shape[0] == len(self.times)
        assert np.all(out == np.array(self.lt))
        return

    @pytest.mark.parametrize("multi_lon", [True, False])
    def test_lt_to_ut(self, multi_lon):
        """Test the solar local time to UT in hours of day conversion.

        Parameters
        ----------
        multi_lon : bool
           Include a single longitude value or an array-like object

        """

        self.update_lon(multi_lon)
        out = tc.lt_to_ut(self.lt, self.lon)

        assert out.shape[0] == len(self.lt)

        for i, uth in enumerate(out):
            test_uth = self.times[i].hour + self.times[i].minute / 60.0 \
                + (self.times[i].second
                   + self.times[i].microsecond * 1e-6) / 3600.0
            assert test_uth == uth, \
                "{:d} time element doesn't match {:f} != {:f}".format(
                    i, test_uth, uth)
        return

    def test_calc_time_shift(self):
        """Test the polar dial shift calculation."""

        # The output will be 15 degrees times the UT hours of day
        out = tc.lt_to_ut(self.lt, self.lon) * 15.0

        for i, test_shift in enumerate(out):
            shift = tc.calc_time_shift(self.times[i])

            assert shift == test_shift, \
                "{:d} time element has unexpected offset {:f} != {:f}".format(
                    i, shift, test_shift)
        return
