#!/usr/bin/env python
# Copyright 2020, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md
"""Block-based model visualization routines."""

from glob import glob
import os
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import numpy as np
import cartopy.crs as ccrs

from aetherpy import logger
from aetherpy.io import read_routines
from aetherpy.plot import data_prep
from aetherpy.plot import movie_routines
from aetherpy.utils import inputs
from aetherpy.utils import time_conversion


# ----------------------------------------------------------------------------
# Define the support routines

def get_help(file_vars=None):
    """Provide string explaining how to run the command line interface.

    Parameters
    ----------
    file_vars : list or NoneType
        List of file variables or None to exclude this output (default=None)

    Returns
    -------
    help_str : str
        String with formatted help statement

    """

    mname = os.path.join(
        os.path.commonpath([inputs.__file__, data_prep.__file__]),
        'plot_block_model_results.py') if __name__ == '__main__' else __name__

    # TODO: Update help string
    help_str = ''.join(['Usage:\n{:s} -[flags] [filenames]\n'.format(mname),
                        'Flags:\n',
                        '       -help : print this message, include filename ',
                        'for variable names and indices\n',
                        '       -var=number : index of variable to plot\n',
                        '       -cut=alt, lat, or lon : which cut you would ',
                        'like\n',
                        '       -alt=number : alt in km or grid number ',
                        '(closest)\n',
                        '       -lat=number : latitude in degrees (closest)\n',
                        '       -lon=number: longitude in degrees (closest)\n',
                        '       -log : plot the log of the variable\n',
                        '       -winds : overplot winds\n',
                        '       -tec : plot the TEC variable\n',
                        '       -movie=number : provide a positive frame rate',
                        ' to create a movie\n',
                        '       -ext=str : figure or movie extension\n',
                        'At end, list the files you want to plot. This code ',
                        'should work with either GITM files (*.bin) or Aether',
                        ' netCDF files (*.nc)'])

    if file_vars is not None:
        help_str += "File Variables (index, name):\n"
        for ivar, var in enumerate(file_vars):
            help_str += "               ({:d}, {:s})\n".format(ivar, var)

    return help_str


# I wrote this because the old function error / type checks and I want to
# pass whichever args I want and access them easily (var primarily)
def get_command_line_args(argv):
    # Initialize the arguments to their default values
    # TODO: Change default alt value from grid index to physical altitude
    args = {'filelist': [], 'log': False, 'var': None, 'alt': 10, 'tec': False,
            'lon': np.nan, 'lat': np.nan, 'cut': 'alt', 'winds': False,
            'diff': False, 'is_gitm': False, 'has_header': False, 'movie': 0,
            'ext': 'png'}
    args['help'] = (len(argv) == 0)

    for arg in argv:
        # Is cmdline option
        if arg[0] == '-':
            split_arg = arg.split('=')
            akey = split_arg[0][1:]

            if akey == 'help':
                return args

            if len(split_arg) == 1:
                args[akey] = True
            else:
                aval = split_arg[1]
                # Convert to int when necessary
                if (akey in ['alt', 'lon', 'lat']):
                    args[akey] = int(aval)
                else:
                    args[akey] = aval
        # Is filename
        else:
            args['filelist'].append(arg)

            match_bin = re.match(r'(.*)bin', arg)
            if match_bin:
                args['is_gitm'] = True
                args['has_header'] = False

                # Check for a header file:
                check_file = glob(match_bin.group(1) + "header")
                if len(check_file) > 0 and len(check_file[0]) > 1:
                    args['has_header'] = True
            else:
                args['is_gitm'] = False

    # Update default movie extension for POSIX systems
    if args['movie'] > 0 and args['ext'] == 'png':
        if os.name == "posix":
            args['ext'] = "mkv"
        else:
            args['ext'] = "mp4"

    return args


def determine_min_max_within_range(all_block_data, var, alt,
                                   min_lon=-np.inf, max_lon=np.inf,
                                   min_lat=-np.inf, max_lat=np.inf):
    """Determines the minimum and maximum values of var at a given altitude
    within a rectangular latitude / longitude range.

    Parameters
    ----------
    all_block_data : list
        List of dictionaries containing data for each block
    var : str
        Name of the variable to find the min/max of
    alt : int
        Index of the altitude to find the min/max at
    min_lon : float
        Minimum longitude of desired range (inclusive), defaults to -inf
    max_lon : float
        Maximum longitude of desired range (inclusive), defaults to +inf
    min_lat : float
        Minimum latitude of desired range (inclusive), defaults to -inf
    max_lat : float
        Maximum latitude of desired range (inclusive), defaults to +inf

    Returns
    ----------
    mini, maxi : tuple
        Tuple containing min and max
    """
    mini = np.inf
    maxi = -np.inf
    for block_data in all_block_data:
        lon = block_data['lon'][2:-2, 2:-2, alt]
        lat = block_data['lat'][2:-2, 2:-2, alt]
        v = block_data[var][2:-2, 2:-2, alt]
        cond = (lon >= min_lon) & (lon <= max_lon) \
            & (lat >= min_lat) & (lat <= max_lat)
        mini = min(mini, np.min(v[cond], initial=np.inf))
        maxi = max(maxi, np.max(v[cond], initial=-np.inf))
    return mini, maxi


def determine_min_max(all_block_data, var, alt):
    """Determines the minimum and maximum values of var at a given altitude;
    convenience function for determine_min_max_within_range.

    Parameters
    ----------
    all_block_data : list
        List of dictionaries containing data for each block
    var : str
        Name of the variable to find the min/max of
    alt : int
        Index of the altitude to find the min/max at

    Returns
    ----------
    mini, maxi : tuple
        Tuple containing min and max
    """
    return determine_min_max_within_range(
        all_block_data, var, alt
    )


def get_plotting_bounds(all_block_data, var, alt):
    mini, maxi = determine_min_max(all_block_data, var, alt)
    if mini < 0:
        maxi = max(np.abs(mini), np.abs(maxi))
        mini = -maxi
    return mini, maxi


def extract_block_data(data, block_index):
    block_data = {
        'vars': data['vars'],
        'time': data['time']
    }
    for k, v in data.items():
        if hasattr(v, 'shape'):
            block_data[k] = v[block_index]
    return block_data

# ----------------------------------------------------------------------------
# Define the main plotting routines


def plot_block_data(block_data, var_to_plot, alt_to_plot, fig, ax_list,
                    mini, maxi, split_block=False,
                    debug_filename=None, debug_blockindex=None):
    # Extract plotting data
    lonkey = 'COR_lon' if 'COR_lon' in block_data.keys() else 'lon'
    latkey = 'COR_lat' if 'COR_lat' in block_data.keys() else 'lat'
    lons = block_data[lonkey][2:-2, 2:-2, alt_to_plot]
    lats = block_data[latkey][2:-2, 2:-2, alt_to_plot]
    v = block_data[var_to_plot][2:-2, 2:-2, alt_to_plot]
    lons = np.unwrap(
        np.unwrap(lons, period=360, axis=0),
        period=360, axis=1
    )
    cmap = cm.plasma if mini >= 0 else cm.bwr
    plot_kwargs = {
        'vmin': mini,
        'vmax': maxi,
        'cmap': cmap,
        'transform': ccrs.PlateCarree()
    }
    use_centers = (lons.shape == v.shape)

    # Plot data on each Axes instance in axList
    for ax in ax_list:
        plot_on_ax(ax, lons, lats, v, split_block, use_centers, **plot_kwargs)

    # Step by step debug file generation
    if debug_filename and debug_blockindex:
        fname = f"{debug_filename}_block{debug_blockindex}"
        print(f"  Outputting file: {fname}.png")
        fig.savefig(fname, bbox_inches='tight')


def plot_on_ax(ax, lons, lats, v, split_block, use_centers, **kwargs):
    # Block doesn't cover pole
    if not split_block:
        ax.pcolor(lons, lats, v, **kwargs)
        return
    # Block does cover pole -- slice
    x_mid = int(lons.shape[0] / 2)
    y_mid = int(lons.shape[1] / 2)
    corner_offset = 0 if use_centers else 1
    coordslices = [
        np.s_[:x_mid + corner_offset, :y_mid + corner_offset],
        np.s_[:x_mid + corner_offset, y_mid:],
        np.s_[x_mid:, y_mid:],
        np.s_[x_mid:, :y_mid + corner_offset]
    ]
    valslices = [
        np.s_[:x_mid, :y_mid],
        np.s_[:x_mid, y_mid:],
        np.s_[x_mid:, y_mid:],
        np.s_[x_mid:, :y_mid]
    ]
    for cslice, vslice in zip(coordslices, valslices):
        # Get temp slices of coords and values
        t_lons = lons[cslice]
        t_lats = lats[cslice]
        t_v = v[vslice]
        # Fix slices
        if not use_centers:
            t_lons = fix_pole_corner_longitude(t_lons, t_lats)
        t_lons = np.unwrap(
            np.unwrap(t_lons, period=360, axis=0), period=360, axis=1
        )
        # Plot slices, fix plots
        coll = ax.pcolor(t_lons, t_lats, t_v, **kwargs)
        if not use_centers:
            patch = get_pole_corner_patch(coll)
            ax.add_patch(patch)


def fix_pole_corner_longitude(t_lons, t_lats):
    corners = [
        np.s_[0, 0],
        np.s_[0, -1],
        np.s_[-1, -1],
        np.s_[-1, 0]
    ]
    minlati = 0
    minlat = np.inf
    polelon = 0
    for i, corner in enumerate(corners):
        if abs(t_lats[corner]) < minlat:
            minlat = abs(t_lats[corner])
            minlati = i
    polelon = t_lons[corners[minlati]]
    polecorner = corners[(minlati + 2) % 4]
    t_lons[polecorner] = polelon
    return t_lons


def get_pole_corner_patch(coll):
    pole_i, pole_path = [
        (i, path) for i, path in enumerate(coll.get_paths())
        if 90 in abs(path.vertices[:, 1])][0]
    poly = pole_path.to_polygons()[0]
    # Insert point after first pole point
    for i, row in enumerate(poly):
        if row[1] in [90, -90]:
            polepoint = row
            poly = np.insert(poly, i + 1,
                             [[poly[i + 1, 0], row[1]]], axis=0)
            break
    # Insert point before last pole point
    for i, row in reversed(list(enumerate(poly))):
        if np.array_equal(row, polepoint):
            poly = np.insert(poly, i,
                             [[poly[i - 1, 0], row[1]]], axis=0)
            break
    coll.update_scalarmappable()
    patch = mpatches.Polygon(poly, transform=ccrs.PlateCarree(),
                             color=coll.get_facecolors()[pole_i])
    return patch


# TODO: Streamline code/logic duplication
def plot_with_centers(ax, lons, lats, v, split_block, **kwargs):
    if split_block:
        x_mid = int(lons.shape[0] / 2)
        y_mid = int(lons.shape[1] / 2)
        valslices = [
            np.s_[:x_mid, :y_mid],
            np.s_[:x_mid, y_mid:],
            np.s_[x_mid:, y_mid:],
            np.s_[x_mid:, :y_mid]
        ]
        for vslice in valslices:
            t_lons = lons[vslice]
            t_lats = lats[vslice]
            t_v = v[vslice]
            t_lons = np.unwrap(
                np.unwrap(t_lons, period=360, axis=0), period=360, axis=1
            )
            ax.pcolor(t_lons, t_lats, t_v, **kwargs)
    else:
        ax.pcolor(lons, lats, v, **kwargs)


def plot_with_corners(ax, lons, lats, v, split_block, **kwargs):
    if split_block:
        x_mid = int(lons.shape[0] / 2)
        y_mid = int(lons.shape[1] / 2)
        coordslices = [
            np.s_[:x_mid + 1, :y_mid + 1],
            np.s_[:x_mid + 1, y_mid:],
            np.s_[x_mid:, y_mid:],
            np.s_[x_mid:, :y_mid + 1]
        ]
        valslices = [
            np.s_[:x_mid, :y_mid],
            np.s_[:x_mid, y_mid:],
            np.s_[x_mid:, y_mid:],
            np.s_[x_mid:, :y_mid]
        ]
        for cslice, vslice in zip(coordslices, valslices):
            t_lons = lons[cslice]
            t_lats = lats[cslice]
            t_v = v[vslice]

            corners = [
                np.s_[0, 0],
                np.s_[0, -1],
                np.s_[-1, -1],
                np.s_[-1, 0]
            ]
            minlati = 0
            minlat = np.inf
            polelon = 0
            for i, corner in enumerate(corners):
                if abs(t_lats[corner]) < minlat:
                    minlat = abs(t_lats[corner])
                    minlati = i
            polelon = t_lons[corners[minlati]]
            polecorner = corners[(minlati + 2) % 4]
            t_lons[polecorner] = polelon

            t_lons = np.unwrap(
                np.unwrap(t_lons, period=360, axis=0), period=360, axis=1
            )

            coll = ax.pcolor(t_lons, t_lats, t_v, **kwargs)
            pole_i, pole_path = [
                (i, path) for i, path in enumerate(coll.get_paths())
                if 90 in abs(path.vertices[:, 1])][0]
            poly = pole_path.to_polygons()[0]
            # Insert point after first pole point
            for i, row in enumerate(poly):
                if row[1] in [90, -90]:
                    polepoint = row
                    poly = np.insert(poly, i + 1,
                                     [[poly[i + 1, 0], row[1]]], axis=0)
                    break
            # Insert point before last pole point
            for i, row in reversed(list(enumerate(poly))):
                if np.array_equal(row, polepoint):
                    poly = np.insert(poly, i,
                                     [[poly[i - 1, 0], row[1]]], axis=0)
                    break
            coll.update_scalarmappable()
            patch = mpatches.Polygon(poly, transform=ccrs.PlateCarree(),
                                     color=coll.get_facecolors()[pole_i])
            ax.add_patch(patch)
    else:
        ax.pcolor(lons, lats, v, **kwargs)


def plot_all_blocks(all_block_data, var_to_plot, alt_to_plot, plot_filename):
    # Initialize colorbar information
    mini, maxi = get_plotting_bounds(
        all_block_data, var_to_plot, alt_to_plot)
    norm = colors.Normalize(vmin=mini, vmax=maxi)
    cmap = cm.plasma if mini >= 0 else cm.bwr
    col = 'white' if mini >= 0 else 'black'

    # Initialize figure to plot on
    fig = plt.figure(figsize=(11, 10), constrained_layout=True)
    altitude = round(
        all_block_data[0]['z'][0, 0, alt_to_plot] / 1000.0, 2)
    time = all_block_data[0]['time']
    title = f"{time}; var: {var_to_plot}; alt: {altitude} km"

    # Calculate circle plot rotations to place sun at top
    hours = time.hour + time.minute / 60 + time.second / 3600
    north_longitude = -15 * hours
    south_longitude = 180 - 15 * hours

    # Define subplot projections and gridspecs, create subplots
    north_proj = ccrs.Stereographic(
        central_latitude=90, central_longitude=north_longitude)
    south_proj = ccrs.Stereographic(
        central_latitude=-90, central_longitude=south_longitude)
    world_proj = ccrs.PlateCarree(central_longitude=0)

    # Create subplots
    gs = fig.add_gridspec(nrows=2, ncols=2, hspace=-0.2)

    north_ax = fig.add_subplot(gs[0, 0], projection=north_proj)
    south_ax = fig.add_subplot(gs[0, 1], projection=south_proj)
    world_ax = fig.add_subplot(gs[1, :], projection=world_proj)
    world_ax.title.set_text(title)

    ax_list = [north_ax, south_ax, world_ax]

    # Set subplot extents
    world_ax.set_global()

    # Limit latitudes of circle plots to >45 degrees N/S
    border_latitude = 45
    r_limit, _ = north_proj.transform_point(90, border_latitude,
                                            ccrs.PlateCarree())
    r_limit = abs(r_limit)
    r_extent = r_limit * 1.00001
    circle_bound = mpath.Path.unit_circle()
    circle_bound = mpath.Path(
        circle_bound.vertices.copy() * r_limit, circle_bound.codes.copy())
    for ax in [north_ax, south_ax]:
        ax.set_xlim(-r_extent, r_extent)
        ax.set_ylim(-r_extent, r_extent)
        ax.set_boundary(circle_bound)

    # Plot all block data on figure
    for i, block_data in enumerate(all_block_data):
        print(f"  Computing block {i}")
        split_block = len(all_block_data) == 6 and i in [4, 5]
        plot_block_data(block_data, var_to_plot, alt_to_plot, fig,
                        ax_list, mini, maxi, split_block)

    # Add elements affecting all subplots
    for ax in ax_list:
        ax.coastlines(color=col)

    # Configure colorbar
    power = int(np.log10(max(maxi, 1)))
    create_colorbar(fig, norm, cmap, ax_list, var_to_plot, power)

    # Add labels to circle plots
    north_minmax = determine_min_max_within_range(
        all_block_data, var_to_plot, alt_to_plot,
        min_lat=45, max_lat=90
    )
    south_minmax = determine_min_max_within_range(
        all_block_data, var_to_plot, alt_to_plot,
        min_lat=-90, max_lat=-45
    )
    label_circle_plots(north_ax, south_ax, *north_minmax, *south_minmax)

    # Save plot
    print(f"Outputting file: {plot_filename}.png")
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.close(fig)


def label_circle_plots(north_ax, south_ax, north_min, north_max,
                       south_min, south_max):
    # Add local time labels to circle plots
    north_kwargs = {
        'horizontalalignment': 'center',
        'verticalalignment': 'center',
        'fontsize': 'small',
        'transform': north_ax.transAxes
    }
    south_kwargs = {
        'horizontalalignment': 'center',
        'verticalalignment': 'center',
        'fontsize': 'small',
        'transform': south_ax.transAxes
    }

    north_ax.text(0.5, -0.03, '00', **north_kwargs)     # Bottom
    north_ax.text(1.03, 0.5, '06', **north_kwargs)      # Right
    north_ax.text(0.5, 1.03, '12', **north_kwargs)      # Top
    north_ax.text(-0.03, 0.5, '18', **north_kwargs)     # Left

    south_ax.text(0.5, -0.03, '00', **south_kwargs)     # Bottom
    south_ax.text(-0.03, 0.5, '06', **south_kwargs)     # Left
    south_ax.text(0.5, 1.03, '12', **south_kwargs)      # Top
    south_ax.text(1.03, 0.5, '18', **south_kwargs)      # Right

    # Add min/max labels to circle plots
    north_mintext = f"Min: {north_min:.3e}".replace('+', '')
    north_maxtext = f"Max: {north_max:.3e}".replace('+', '')
    south_mintext = f"Min: {south_min:.3e}".replace('+', '')
    south_maxtext = f"Max: {south_max:.3e}".replace('+', '')
    north_ax.text(0.125, 0.125, north_mintext, **north_kwargs, rotation=-45)
    north_ax.text(0.875, 0.125, north_maxtext, **north_kwargs, rotation=45)
    south_ax.text(0.125, 0.125, south_mintext, **south_kwargs, rotation=-45)
    south_ax.text(0.875, 0.125, south_maxtext, **south_kwargs, rotation=45)


def create_colorbar(fig, norm, cmap, ax_list, var_to_plot, power):
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax_list, shrink=0.5, pad=0.01)
    cbar.formatter.set_useMathText(True)
    cbar.ax.yaxis.get_offset_text().set_rotation('vertical')
    cbar_label = f"{var_to_plot} (x$10^{{{power}}} / $m$^3$)"
    cbar.set_label(cbar_label, rotation='vertical')
    cbar.formatter.set_powerlimits((0, 0))


def plot_model_block_results():
    # Get the input arguments
    args = get_command_line_args(inputs.process_command_line_input())

    # Read headers for input files (assumes all files have same header)
    header = read_routines.read_blocked_netcdf_header(args['filelist'][0])

    # Output help
    if (len(args['filelist']) == 0 or args['help']):
        help_str = get_help(header['vars'] if args['help'] else None)
        print(help_str)
        return

    # Determine variables to plot (currently hardcoded)
    # TODO: handle winds correctly
    alt_to_plot = args['alt']
    file_vars = ['lon', 'lat', 'z', args['var']] if args['var'] else None

    # Generate plots for each file
    for filename in args['filelist']:
        # Retrieve all block data
        data = read_routines.read_blocked_netcdf_file(filename, file_vars)

        # Search for compatible 3DCOR files, add to data if found
        filename_list = filename.split('/')[:-1]
        searchstr = f"{'/'.join(filename_list)}/3DCOR*.nc"
        corner_files = glob(searchstr)
        for f in corner_files:
            corner_data = read_routines.read_aether_file(f)
            correct_shape = (
                data['nblocks'],
                data['nlons'] + 1,
                data['nlats'] + 1,
                data['nalts'] + 1
            )
            # TODO: change this to ['lon', 'lat', 'z'] after updating functions
            valid_file = all(
                corner_data[var].shape == correct_shape
                for var in [1, 2, 3]
            )
            if valid_file:
                data['COR_lon'] = corner_data[1]
                data['COR_lat'] = corner_data[2]
                data['COR_z'] = corner_data[3]
                break

        # Separate data dict into all_block_data list of dicts
        all_block_data = []
        for i in range(data['nblocks']):
            block_data = extract_block_data(data, i)
            all_block_data.append(block_data)

        # Plot desired variable if given, plot all variables if not
        all_vars = [v for v in data['vars']
                    if v not in ['time', 'lon', 'lat', 'z']]
        plot_vars = [args['var']] if args['var'] else all_vars

        # Generate plots for each variable requested
        for var_to_plot in plot_vars:
            print(f"Plotting variable: {var_to_plot}")
            plot_filename = f"{filename.split('.')[0]}_{var_to_plot}"
            plot_all_blocks(
                all_block_data, var_to_plot, alt_to_plot, plot_filename)


# Needed to run main script as the default executable from the command line
if __name__ == '__main__':
    plot_model_block_results()
