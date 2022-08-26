#!/usr/bin/env python
# Copyright 2020, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md
""" Utilities for slicing and preparing data for plotting
"""

import numpy as np

from aetherpy import logger


def get_cut_index(lons, lats, alts, cut_val, isgrid=False, cut_coord='alt'):
    """Select indices needed to obtain a slice in the remaining two coords.

    Parameters
    ----------
    lons : array-like
        1D array of longitudes in degrees
    lats : array-like
        1D array of latitudes in degrees
    alts : array-like
        1D array of altitudes in km
    cut_val : int or float
        Data value or grid number along that will be held constant
    isgrid : bool
        Flag that indicates `cut_val` is a grid index if True or that it is a
        data value if False (default=False)
    cut_coord : str
        Expects one of 'lat', 'lon', or 'alt' and will return an index for
        that data, allowing a 2D slice to be created along other two
        coordinates (default='alt')

    Returns
    -------
    icut : int
        Cut index
    cut_data : tuple
        Tuple to select data of the expected dimensions along `icut`
    x_coord : array-like
        Array of data to include along the x-axis
    y_coord : array-like
        Array of data to include along the y-axis
    z_val : float
        Data value for cut index

    Notes
    -----
    `lons`, `lats`, and `alts` do not need to be the same shape

    Raises
    ------
    ValueError
        If `cut_val` is outside the possible range of values

    """
    # Ensure inputs are array-like
    lons = np.asarray(lons)
    lats = np.asarray(lats)
    alts = np.asarray(alts)

    # Initialize the output slices
    out_slice = {"lon": slice(0, lons.shape[0], 1),
                 "lat": slice(0, lats.shape[0], 1),
                 "alt": slice(0, alts.shape[0], 1)}

    # Set the x, y, and z coordinates
    x_coord = lons if cut_coord in ['alt', 'lat'] else lats
    y_coord = alts if cut_coord in ['lat', 'lon'] else lats

    if cut_coord == 'alt':
        z_coord = alts
    else:
        if cut_coord == 'lat':
            z_coord = lats
        else:
            z_coord = lons

    # Find the desired index for the z-coordinate value
    if isgrid:
        icut = cut_val
    else:
        if cut_val < z_coord.min() or cut_val > z_coord.max():
            raise ValueError('Requested cut is outside the coordinate range')

        icut = abs(z_coord - cut_val).argmin()

    # Get the z-value if possible.
    if icut < 0 or icut >= len(z_coord):
        raise ValueError('Requested cut is outside the index range')

    # Warn the user if they selected a suspect index
    if cut_coord == "alt":
        if icut > len(z_coord) - 3:
            logger.warning(''.join(['Requested altitude slice is above ',
                                    'the recommended upper limit']))
    else:
        if icut == 0 or icut == len(z_coord) - 1:
            logger.warning(''.join(['Requested ', cut_coord, ' slice is ',
                                    'beyond the recommended limits']))

    z_val = z_coord[icut]

    # Finalize the slicing tuple
    out_slice[cut_coord] = icut
    cut_data = tuple([out_slice[coord] for coord in ['lon', 'lat', 'alt']])

    return icut, cut_data, x_coord, y_coord, z_val


def calc_tec(alt, ne, ialt_min=2, ialt_max=-4):
    """Calculate TEC for the specified altitude range from electron density.

    Parameters
    ----------
    alt : array-like
        1D Altitude data in km
    ne : array-like
        Electron density in cubic meters, with altitude as the last axis
    ialt_min : int
        Lowest altitude index to use, may be negative (default=2)
    ialt_max : int
        Highest altitude index to use, may be negative (default=-4)

    Returns
    -------
    tec : array-like
        Array of TEC values in TECU

    Notes
    -----
    TEC = Total Electron Content
    TECU = 10^16 m^-2 (TEC Units)

    Raises
    ------
    ValueError
        If the altitude integration range is poorly defined.

    """
    alts = np.asarray(alt)
    nes = np.asarray(ne)

    # Initialize the TEC
    tec = np.zeros(shape=nes[..., 0].shape)

    # Get the range of altitudes to cycle over
    if ialt_max < 0:
        ialt_max += alt.shape[0]

    if ialt_min < 0:
        ialt_min += alt.shape[0]

    if ialt_max <= ialt_min:
        raise ValueError('`ialt_max` must be greater than `ialt_min`')

    # Cycle through each altitude bin, summing the contribution from the
    # electron density
    for ialt in np.arange(ialt_min, ialt_max, 1):
        tec += 1000.0 * nes[..., ialt] * (alts[ialt + 1]
                                          - alts[ialt - 1]) / 2.0

    # Convert TEC from per squared meters to TECU
    tec /= 1.0e16

    return tec
