#!/usr/bin/env python
# Copyright 2020, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md
"""Utilities for slicing and preparing data for plotting."""

import numpy as np

from aetherpy import logger
from aetherpy.io import read_routines


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
            raise ValueError(''.join(['Requested cut is outside the ',
                                      'coordinate range [{:} '.format(cut_val),
                                      'not in {:} to {:}]'.format(
                                          z_coord.min(), z_coord.max())]))

        icut = abs(z_coord - cut_val).argmin()

    # Get the z-value if possible.
    if icut < 0 or icut >= len(z_coord):
        raise ValueError('Requested cut is outside the index range')

    # Warn the user if they selected a suspect index
    if cut_coord == "alt":
        if icut > len(z_coord) - 3:
            logger.warning(''.join(['Requested altitude slice is above the ',
                                    'recommended upper limit of {:}'.format(
                                        len(z_coord) - 3)]))
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


def load_data_for_plotting(filelist, plot_var, cut_var='alt', cut_val=400,
                           has_header=False, is_gitm=False, winds=False,
                           tec=False):
    """Load model data for plotting.

    Parameters
    ----------
    filelist : list-like
        List of model filenames to load
    plot_var : str or int
        Variable name or index of data to plot
    cut_var : str
        Variable name along which data should be sliced (default='alt')
    cut_val : int or float
        Data value along that will be held constant
    has_header : bool
        Flag indicating that a separate header file contains the header data,
        as is the case for binary files (default=False)
    is_gitm : bool
        Flag indicating whether this is a GITM file, if True, or and Aether
        file, if False (default=False)
    winds : bool
        If True prepare winds or drifts for quiver-style plotting
        (default=False)
    tec : bool
        If True calculate the TEC for plotting (default=False)

    Returns
    -------
    all_times : array-like
        1D array of datetimes to plot
    all_2dim_data : array-like
        3D array of data variables (time, x, and y)
    all_winds_x : array-like or NoneType
        1D array of winds along x-axis or None if `winds` is False
    all_winds_y : array-like or NoneType
        1D array of winds along y-axis or None if `winds` is False
    icut : int
        Cut index
    x_coord : array-like
        Array of data to include along the x-axis
    y_coord : array-like
        Array of data to include along the y-axis
    z_val : float
        Data value for cut index
    var_name : str
        Long name of the data variable

    Raises
    ------
    ValueError
        If a bad `plot_var` value is provided

    """

    # Load the header data
    header = read_routines.read_headers(filelist, has_header=has_header,
                                        is_gitm=is_gitm)

    if is_gitm and has_header:
        is_gitm = False

    if isinstance(plot_var, int):
        if plot_var >= len(header["vars"]):
            raise ValueError("requested bad variable index: {:d}>={:d}".format(
                plot_var, len(header["vars"])))
    elif plot_var not in header["vars"]:
        raise ValueError("unknown variable requested: {:s} not in {:}".format(
            plot_var, header["vars"]))

    # Define the plotting inputs
    var_list = ['lon', 'lat', 'z', plot_var]
    plot_vars = [0, 1, 2, plot_var]

    # Update plotting variables to include the wind, if desired
    if winds:
        if cut_var in ['alt', 'lat']:
            plot_vars.append(16)
            var_list.append('Zonal Wind')
        else:
            plot_vars.append(17)
            var_list.append('Meridional Wind')

        if cut_var in ['lat', 'lon']:
            plot_vars.append(18)
            var_list.append('Vertical Wind')
        else:
            plot_vars.append(17)
            var_list.append('Meridional Wind')

        all_winds_x = []
        all_winds_y = []
    else:
        all_winds_x = None
        all_winds_y = None

    # Prepare to load the desired file data
    all_2dim_data = []
    all_times = []
    var_name = None
    convert_lat = True
    convert_lon = True

    for j, filename in enumerate(filelist):
        # Read in the data file
        if is_gitm:
            data = read_routines.read_gitm_file(filename, plot_vars)
            var_list[3] = data['vars'][3]
        else:
            if has_header:
                data = read_routines.read_aether_one_binary_file(header, j,
                                                                 plot_vars)
                var_list[3] = data['vars'][3]
            else:
                data = read_routines.read_aether_file(filename, var_list)
                plot_vars[3] = 3

                if data['units'][0].find('degree') >= 0:
                    convert_lon = False

                if data['units'][1].find('degree') >= 0:
                    convert_lat = False

        # Get the z-variable name, if needed
        if var_name is None:
            vkey = 'long_name' if 'long_name' in data.keys() else 'vars'
            var_name = data[vkey][plot_vars[3]]

        # For the first file, initialize the necessary plotting data
        if j == 0:
            # Get 1D arrays for the coordinates
            alts = data[2][0, 0] / 1000.0  # Convert from m to km

            # Convert from rad to deg, if necessary, and reshape lat and lon
            lons = data[0][:, 0, 0]
            lats = data[1][0, :, 0]

            if convert_lat:
                lats = np.degrees(lats)

            if convert_lon:
                lons = np.degrees(lons)

            # Find the desired index to cut along to get a 2D slice
            icut, cut_data, x_coord, y_coord, z_val = get_cut_index(
                lons, lats, alts, cut_val, cut_coord=cut_var, isgrid=False)

        # Save the time data
        all_times.append(data["time"])

        # Save the z-axis data
        if tec:
            all_2dim_data.append(calc_tec(alts, data[plot_vars[3]], 2, -4))
        else:
            all_2dim_data.append(data[plot_vars[3]][cut_data])

            if winds:
                all_winds_x.append(data[plot_vars[-1]][cut_data])
                all_winds_y.append(data[plot_vars[-1]][cut_data])

    # Convert data list to a numpy array
    all_2dim_data = np.array(all_2dim_data)

    if winds:
        all_winds_x = np.array(all_winds_x)
        all_winds_y = np.array(all_winds_y)

    # Return data for plotting
    return(all_times, all_2dim_data, all_winds_x, all_winds_y, icut, x_coord,
           y_coord, z_val, var_name)
