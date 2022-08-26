#!/usr/bin/env python
# Copyright 2020, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md
"""Block-based model visualization routines."""

from glob import glob
import os
import re
import argparse
from xml.dom import minicompat
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import numpy as np
import cartopy.crs as ccrs
from scipy.spatial import KDTree

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


def argparse_command_line_args():
    parser = argparse.ArgumentParser(
        description='Plotting script for Aether model output files',
        epilog='This code should work with either GITM files (*.bin) \
            or Aether netcdf files (*.nc).',
        add_help=False)
    parser.add_argument('filelist', nargs='+', help='file(s) to plot')
    parser.add_argument('-h', '--help', action='store_true',
                        help='show help message and list of variables, exit')
    parser.add_argument('-v', '--var', nargs='*', default=[],
                        help='name of variable(s) to plot')
    parser.add_argument('-n', '--varn', nargs='*', type=int, default=[],
                        help='index of variable(s) to plot')
    parser.add_argument('-c', '--cut', default='alt',
                        choices=['alt', 'lat', 'lon'],
                        help='which cut you would like')
    parser.add_argument('-a', '--alt', default=np.nan, type=float,
                        help='alt in km (closest)')
    parser.add_argument('--lat', default=np.nan, type=float,
                        help='latitude in degrees (closest)')
    parser.add_argument('--lon', default=np.nan, type=float,
                        help='longitude in degrees (closest)')
    parser.add_argument('-l', '--log', action='store_true',
                        help='plot the log of the variable')
    parser.add_argument('-t', '--tec', action='store_true',
                        help='plot the TEC variable')
    parser.add_argument('-w', '--winds', action='store_true',
                        help='overplot neutral winds (only if --ions is off)')
    parser.add_argument('-i', '--ions', action='store_true',
                        help='overplot ion winds')
    parser.add_argument('-d', '--diff', action='store_true',
                        help='flag for difference with other plots')
    parser.add_argument('--is_gitm', action='store_true',
                        help='flag for plotting gitm files')
    parser.add_argument('--has_header', action='store_true',
                        help='flag for if file headers exist')
    parser.add_argument('--movie', default=0, type=int,
                        help='provide a positive framerate to create a movie')
    parser.add_argument('--ext', default='png',
                        help='figure or movie extension')
    parser.add_argument('--original', action='store_true',
                        help='create exact plots (warning: very slow)')
    args = parser.parse_args()
    # Output generic help if no files
    if (len(args.filelist) == 0):
        parser.print_help()
        return False
    # Process var and varn into one list, output indices and variables if bad
    header = read_routines.read_file(args.filelist[0])
    if (args.help):
        parser.print_help()
        list_header_variables(header)
        return False
    varnames = args.var
    for varn in args.varn:
        try:
            varnames.append(header['vars'][varn])
        except IndexError:
            print(f"{varn} is an invalid variable index")
            list_header_variables(header)
            return False
    for name in varnames:
        if name not in header['vars']:
            print(f"{name} is an invalid variable name")
            list_header_variables(header)
            return False
    args.var = varnames
    return args.__dict__


def list_header_variables(header):
    print("File Variables: ")
    print("\tIndex\tName")
    for i, var in enumerate(header['vars']):
        print(f"\t{i}\t{var}")


def determine_min_max_within_range(data, var, alt,
                                   min_lon=-np.inf, max_lon=np.inf,
                                   min_lat=-np.inf, max_lat=np.inf):
    """Determines the minimum and maximum values of var at a given altitude
    within a rectangular latitude / longitude range.

    Parameters
    ----------
    data : dict
        Dictionary containing each block's data
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
    all_lons = data['lon'][:, 2:-2, 2:-2, alt]
    all_lats = data['lat'][:, 2:-2, 2:-2, alt]
    all_v = data[var][:, 2:-2, 2:-2, alt]
    cond = (all_lons >= min_lon) & (all_lons <= max_lon) \
        & (all_lats >= min_lat) & (all_lats <= max_lat)
    mini = np.nanmin(all_v[cond], initial=np.inf)
    maxi = np.nanmax(all_v[cond], initial=-np.inf)
    return mini, maxi


def determine_min_max(data, var, alt):
    """Determines the minimum and maximum values of var at a given altitude;
    convenience function for determine_min_max_within_range.

    Parameters
    ----------
    data : dict
        Dictionary containing each block's data
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
        data, var, alt
    )


def get_plotting_bounds(data, var, alt):
    mini, maxi = determine_min_max(data, var, alt)
    if mini < 0:
        maxi = max(np.abs(mini), np.abs(maxi))
        mini = -maxi
    return mini, maxi


# ----------------------------------------------------------------------------
# Define the main plotting routines


def lon_lat_to_cartesian(lon, lat, R=1):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x, y, z


def generate_spherical_mesh(lon_cells, lat_cells):
    lon_halfdim = 360 / (2 * lon_cells)
    lat_halfdim = 180 / (2 * lat_cells)
    x = np.linspace(0 + lon_halfdim, 360 - lon_halfdim, lon_cells)
    y = np.linspace(-90 + lat_halfdim, 90 - lat_halfdim, lat_cells)
    X, Y = np.meshgrid(x, y)
    return X, Y


def generate_circular_mesh(lon_cells, lat_cells, reference_lat):
    lons = []
    lats = []
    lat_halfdim = 180 / (2 * lat_cells)
    lat_centers = np.linspace(-90 + lat_halfdim, 90 - lat_halfdim, lat_cells)
    reference_cos = np.cos(np.radians(reference_lat))
    ring_lon_cells = np.int64(
        lon_cells * np.cos(np.radians(lat_centers)) / reference_cos)
    for lat, cells in zip(lat_centers, ring_lon_cells):
        ring_lons = np.linspace(0, 360, cells + 1)[:-1]
        ring_lats = [lat] * cells
        lons.extend(ring_lons)
        lats.extend(ring_lats)
    lons = np.asarray(lons)
    lats = np.asarray(lats)
    return lons, lats


def plot_winds(ax, data, alt_idx_to_plot, args, target_mesh):
    if not (args['winds'] or args['ions']):
        return
    desire_neutral_winds = not args['ions']
    neutral_vars = {
        'east': 'Zonal Wind',
        'north': 'Meridional Wind',
        'up': 'Vertical Wind'
    }
    ion_vars = {
        'east': 'BulkIon Velocity (Zonal)',
        'north': 'BulkIon Velocity (Meridional)',
        'up': 'BulkIon Velocity (Vertical)'
    }
    wind_vars = neutral_vars if desire_neutral_winds else ion_vars
    vars_to_plot = list(wind_vars.values())
    # TODO: Create adjusted wind mesh to fix density around poles
    
    target_lon, target_lat, target_winds = generate_interpolated_data(
        data, vars_to_plot, alt_idx_to_plot, target_mesh=target_mesh
    )
    # TODO: Choose x_winds and y_winds based on cut from args
    x_winds = target_winds[wind_vars['east']]
    y_winds = target_winds[wind_vars['north']]

    # Example uniform winds for testing
    # x_winds = np.ones(target_lon.shape)
    # y_winds = np.ones(target_lon.shape) * 0.5

    ax.quiver(target_lon, target_lat, x_winds, y_winds,
              transform=ccrs.PlateCarree())


def generate_interpolated_data(data, vars_to_plot, alt_idx_to_plot,
                               lon_cells=1800, lat_cells=900, target_mesh=None,
                               neighbors=1):
    # Validate vars_to_plot input type
    if not isinstance(vars_to_plot, list):
        vars_to_plot = [vars_to_plot]
    # Load input data for all variables
    source_lons = []
    source_lats = []
    source_vals = {var: [] for var in vars_to_plot}
    for i in range(data['nblocks']):
        lon = data['lon'][i, 2:-2, 2:-2, alt_idx_to_plot]
        lat = data['lat'][i, 2:-2, 2:-2, alt_idx_to_plot]
        source_lons.extend(lon.flatten())
        source_lats.extend(lat.flatten())
        for var in vars_to_plot:
            v = data[var][i, 2:-2, 2:-2, alt_idx_to_plot]
            source_vals[var].extend(v.flatten())

    # Generate KDTree of source point cloud
    x_source, y_source, z_source = lon_lat_to_cartesian(source_lons,
                                                        source_lats)
    tree = KDTree(list(zip(x_source, y_source, z_source)))

    # Generate target mesh
    if target_mesh is None:
        target_lon, target_lat = generate_spherical_mesh(lon_cells, lat_cells)
    else:
        target_lon, target_lat = target_mesh

    # Compute IDW interpolation weights
    x_target, y_target, z_target = lon_lat_to_cartesian(target_lon.flatten(),
                                                        target_lat.flatten())
    d, inds = tree.query(list(zip(x_target, y_target, z_target)), k=neighbors)
    if len(d.shape) == 1:
        d = np.expand_dims(d, axis=1)
    if len(inds.shape) == 1:
        inds = np.expand_dims(inds, axis=1)
    d += 0.00001
    w = 1.0 / (d * d)

    print(d.shape, inds.shape)

    # Apply weights to get interpolated variable data
    target_vs = {}
    for var, values in source_vals.items():
        values = np.asarray(values)
        target_v = np.sum(w * values[inds], axis=1) / np.sum(w, axis=1)
        target_v.shape = target_lon.shape
        target_vs[var] = target_v

    # Simplify output if only one variable was given
    target_output = target_v if len(vars_to_plot) == 1 else target_vs

    return target_lon, target_lat, target_output


def plot_all_blocks(data, var_to_plot, alt_idx_to_plot, plot_filename, args,
                    mini=None, maxi=None):
    print(f"  Plotting variable: {var_to_plot}")

    # Initialize figure to plot on
    fig = plt.figure(figsize=(13, 13))
    altitude = round(
        data['z'][0, 0, 0, alt_idx_to_plot] / 1000.0, 2)
    time = data['time']
    title = f"{time}; var: {var_to_plot}; alt: {altitude} km"

    # Initialize colorbar information
    if mini is None or maxi is None:
        mini, maxi = get_plotting_bounds(
            data, var_to_plot, alt_idx_to_plot)
    norm = colors.Normalize(vmin=mini, vmax=maxi)
    cmap = cm.plasma if mini >= 0 else cm.bwr
    col = 'white' if mini >= 0 else 'black'

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

    # Set subplot extents
    world_ax.set_global()

    # Limit latitudes of circle plots to >45 degrees N/S
    border_latitude = 45
    set_circle_plot_bounds([north_ax, south_ax], north_proj, border_latitude)

    # Plot interpolated var_to_plot data onto subplots
    target_lon, target_lat, target_v = generate_interpolated_data(
        data, var_to_plot, alt_idx_to_plot)

    # Apply log to data if desired
    if args['log']:
        target_v = np.log10(target_v)
        mini = np.log10(mini)
        maxi = np.log10(maxi)

    # Plot interpolated data
    ax_list = [north_ax, south_ax, world_ax]
    plot_kwargs = {
        'vmin': mini,
        'vmax': maxi,
        'cmap': cmap,
        'transform': ccrs.PlateCarree()
    }
    for ax in ax_list:
        ax.pcolormesh(target_lon, target_lat, target_v, **plot_kwargs)

    # Plot interpolated winds
    if args['winds'] or args['ions']:
        world_ax_mesh = generate_spherical_mesh(36, 18)
        circle_ax_mesh = generate_circular_mesh(36, 18, 45)
        plot_winds(world_ax, data, alt_idx_to_plot, args, world_ax_mesh)
        for ax in [north_ax, south_ax]:
            plot_winds(ax, data, alt_idx_to_plot, args, circle_ax_mesh)

    # Add elements affecting all subplots
    for ax in ax_list:
        ax.coastlines(color=col)

    # Configure colorbar
    power = int(np.log10(max(maxi, 1)))
    create_colorbar(fig, norm, cmap, ax_list, var_to_plot, power)

    # Add labels to circle plots
    north_minmax = determine_min_max_within_range(
        data, var_to_plot, alt_idx_to_plot,
        min_lat=45, max_lat=90
    )
    south_minmax = determine_min_max_within_range(
        data, var_to_plot, alt_idx_to_plot,
        min_lat=-90, max_lat=-45
    )
    label_circle_plots(north_ax, south_ax, *north_minmax, *south_minmax)

    # Save plot
    print(f"    Saving plot to: {plot_filename}.png")
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.close(fig)


def set_circle_plot_bounds(circle_ax_list, circle_proj, border_latitude):
    border_latitude = abs(border_latitude)
    r_limit = abs(circle_proj.transform_point(
        90 + circle_proj.proj4_params['lon_0'],
        border_latitude, ccrs.PlateCarree())[0]
    )
    r_extent = r_limit * 1.0001
    circle_bound = mpath.Path.unit_circle()
    circle_bound = mpath.Path(
        circle_bound.vertices.copy() * r_limit, circle_bound.codes.copy())
    for ax in circle_ax_list:
        ax.set_xlim(-r_extent, r_extent)
        ax.set_ylim(-r_extent, r_extent)
        ax.set_boundary(circle_bound)


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
                        ax=ax_list, shrink=0.5, pad=0.03)
    cbar.formatter.set_useMathText(True)
    cbar.ax.yaxis.get_offset_text().set_rotation('vertical')
    cbar_label = f"{var_to_plot} (x$10^{{{power}}} / $m$^3$)"
    cbar.set_label(cbar_label, rotation='vertical')
    cbar.formatter.set_powerlimits((0, 0))


def read_all_files(filelist, file_vars=None):
    files_data = {}
    common_vars = None
    for filename in filelist:
        # Retrieve data
        data = read_routines.read_file(filename)
        data = read_routines.standardize_data(data)
        # Save data to dict, update set of common variables
        files_data[filename] = data
        if common_vars is None:
            common_vars = set(data['vars'])
        else:
            common_vars &= set(data['vars'])
    common_vars = [var for var in common_vars if var != 'time']
    return files_data, common_vars


def plot_model_block_results():
    # Get the input arguments
    # args = get_command_line_args(inputs.process_command_line_input())
    args = argparse_command_line_args()
    if (args is False):
        return

    # Read headers for input files (assumes all files have same header)
    # header = read_routines.read_blocked_netcdf_header(args['filelist'][0])
    header = read_routines.read_file(args['filelist'][0])

    # Output help
    if (len(args['filelist']) == 0 or args['help']):
        help_str = get_help(header['vars'] if args['help'] else None)
        print(help_str)
        return

    # Determine variables to plot (currently hardcoded)
    # TODO: handle winds correctly
    desired_alt = args['alt'] * 1000
    file_vars = ['lon', 'lat', 'z', *args['var']] if args['var'] else None

    # Process all file data
    files_data, common_vars = read_all_files(args['filelist'], file_vars)

    # Calculate min and max for all common vars over all files
    var_min = {
        var: np.inf
        for var in common_vars
    }
    var_max = {
        var: -np.inf
        for var in common_vars
    }
    for filename, data in files_data.items():
        available_alts = data['z'][0, 0, 0, :]
        alt_idx_to_plot = np.argmin(np.abs(available_alts - desired_alt))
        for var in common_vars:
            data_min, data_max = determine_min_max(data, var, alt_idx_to_plot)
            var_min[var] = min(var_min[var], data_min)
            var_max[var] = max(var_max[var], data_max)

    # Generate plots for each file
    for filename, data in files_data.items():
        print(f"Currently plotting: {filename}")

        # Plot desired variable if given, plot all variables if not
        all_vars = [v for v in data['vars']
                    if v not in ['time', 'lon', 'lat', 'z']]
        plot_vars = args['var'] if args['var'] else all_vars
        available_alts = data['z'][0, 0, 0, :]
        alt_idx_to_plot = np.argmin(np.abs(available_alts - desired_alt))

        # Generate plots for each variable requested
        for var_to_plot in plot_vars:
            var_name_stripped = var_to_plot.replace(" ", "")
            plot_filename = f"{filename.split('.')[0]}_{var_name_stripped}"
            mini = var_min[var_to_plot] if var_to_plot in var_min else None
            maxi = var_max[var_to_plot] if var_to_plot in var_max else None
            plot_all_blocks(
                data, var_to_plot, alt_idx_to_plot,
                plot_filename, args, mini, maxi)


# Needed to run main script as the default executable from the command line
if __name__ == '__main__':
    plot_model_block_results()
