#!/usr/bin/env python
# Copyright 2020, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md
"""Standard plot functions for model results."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from aetherpy import logger
from aetherpy.plot import movie_routines
from aetherpy.utils import time_conversion


def plot_model_results(all_times, all_2dim_data, all_winds_x, all_winds_y,
                       plot_var, cut_var, icut, x_coord, y_coord, z_val,
                       log_scale=False, cmap=None, prefix='', ext='png',
                       movie_rate=0, save_plots=True, close_plots=True):
    """Default plot for visualizing model data.

    Parameters
    ----------
    all_times : array-like
        1D array of datetimes to plot
    all_2dim_data : array-like
        3D array of data variables (time, x, and y)
    all_winds_x : array-like or NoneType
        1D array of winds along x-axis or None if `winds` is False
    all_winds_y : array-like or NoneType
        1D array of winds along y-axis or None if `winds` is False
    plot_var : str
        Variable name of data to plot
    cut_var : str
        Variable name along which data should be sliced
    icut : int
        Cut index
    x_coord : array-like
        Array of data to include along the x-axis
    y_coord : array-like
        Array of data to include along the y-axis
    z_val : float
        Data value for cut index
    log_scale : bool
        Use a base-10 logarithmic scale for the data (default=False)
    cmap : matplotlib.colors.ListedColormap or NoneType
        Colormap for plotting, if None defaults to matplotlib's 'plasma' for
        asymmetric data and 'bwr' for symmetric data (default=mpl.cm.plasma)
    prefix : str
        Prefix for output figure filenames (default='')
    ext : str
        Extention for output figure files, do not include period (default='png')
    movie_rate : int
        Movie frame rate or zero if no movie is desired (default=0)
    save_plots : bool
        Save plots to file (default=True)
    close_plots : bool
        Close figure handles for plots (default=True)

    Returns
    -------
    figs : list
        List of open figure handles, empty list if `close_plots` is True

    See Also
    --------
    aetherpy.plots.movie_routines.save_movie, matplotlib.cm

    """

    # If desired, take the log of the data
    if log_scale:
        all_2dim_data = np.log10(all_2dim_data)

    # Define plotting limits
    mini = all_2dim_data.min() * 0.99

    if mini < 0.0:
        symmetric = True
        maxi = abs(all_2dim_data).max() * 1.05
        mini = -maxi
    else:
        symmetric = False
        maxi = all_2dim_data.max() * 1.01

    if cut_var == 'alt':
        mask_north = ((y_coord >= 40) & (y_coord <= 90.0))
        mask_south = ((y_coord <= -40) & (y_coord >= -90.0))
        plot_north = mask_north.max()
        plot_south = mask_south.max()

        if plot_north:
            maxi_north = abs(all_2dim_data[:, :, mask_north]).max() * 1.05

            if symmetric:
                mini_north = -maxi_north
            else:
                mini_north = all_2dim_data[:, :, mask_north].min() * 0.95

        if plot_south:
            maxi_south = abs(all_2dim_data[:, :, mask_south]).max() * 1.05

            if symmetric:
                mini_south = -maxi_south
            else:
                mini_south = all_2dim_data[:, :, mask_south].min() * 0.95

    # Define the color scale, if desired
    if cmap is None:
        cmap = mpl.cm.bwr if symmetric else mpl.cm.plasma

    # Define plot range
    minx = (x_coord[1] + x_coord[2]) / 2.0
    maxx = (x_coord[-2] + x_coord[-3]) / 2.0
    miny = (y_coord[1] + y_coord[2]) / 2.0
    maxy = (y_coord[-2] + y_coord[-3]) / 2.0

    # Prepare the output filename
    if save_plots:
        filename = "{:s}{:s}_{:s}{:03d}".format(
            prefix, plot_var.replace(" ", "_"), cut_var, icut)

        if movie_rate > 0:
            img_file_fmt = movie_routines.setup_movie_dir(filename)
        else:
            img_file_fmt = ''.join([filename, '_{:}.', ext])

    # Create a plot for each time
    figs = []
    for itime, utime in enumerate(all_times):
        # Initialize the figure
        figs.append(plt.figure(constrained_layout=False,
                               tight_layout=True, figsize=(10, 8.5)))

        gs1 = mpl.gridspec.GridSpec(nrows=2, ncols=2, wspace=0.0, hspace=0)
        gs = mpl.gridspec.GridSpec(nrows=2, ncols=2, wspace=-0.05, left=0.02,
                                   right=0.95, top=0.99, bottom=0.05)
        ax = figs[-1].add_subplot(gs1[1, 0:2])

        # Plot the global data set (square plot at bottom if three plots):
        dx = (x_coord[1] - x_coord[0]) / 2.0
        xp = np.append(x_coord - dx, x_coord[-1:] + dx)
        dy = (y_coord[1] - y_coord[0]) / 2.0
        yp = np.append(y_coord - dy, y_coord[-1] + dy)
        con = ax.pcolormesh(xp, yp, all_2dim_data[itime].transpose(),
                            vmin=mini, vmax=maxi, cmap=cmap, shading='auto')

        # Add the winds, if desired
        if all_winds_x is not None and all_winds_y is not None:
            ax.quiver(x_coord, y_coord, all_winds_x[itime].transpose(),
                      all_winds_y[itime].transpose())
        ax.set_ylim([miny, maxy])
        ax.set_xlim([minx, maxx])

        # Set the labels and aspect ratio
        ax.set_title("{:s}; {:s}: {:.2f} {:s}".format(
            utime.strftime("%d %b %Y %H:%M:%S UT"), cut_var, z_val,
            'km' if cut_var == 'alt' else r'$^\circ$'), fontsize='medium')
        ax.set_xlabel(r'Latitude ($^\circ$)' if cut_var == 'lon'
                      else r'Longitude ($^\circ$)')
        ax.set_ylabel(r'Latitude ($^\circ$)' if cut_var == 'alt'
                      else r'Altitude (km)')
        if cut_var == 'alt':
            ax.set_aspect(1.0)

        # Set the colorbar
        cbar = figs[-1].colorbar(con, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label(plot_var, rotation=90)

        # If this is an altitude slice, add polar dials
        if cut_var == 'alt' and (plot_north or plot_south):
            # Set the common inputs
            shift = time_conversion.calc_time_shift(utime)

            xlabels = []
            xlabelpos = []
            ylabels = [r'80$^\circ$', r'70$^\circ$', r'60$^\circ$',
                       r'50$^\circ$']

            ylabelpos = [10.0, 20.0, 30.0, 40.0]
            xticks = np.arange(0, 2 * np.pi, np.pi / 2.0)
            yticks = np.arange(10, 50, 10)

            if plot_north:
                # Top Left Graph Northern Hemisphere
                ax2 = figs[-1].add_subplot(gs[0, 0], projection='polar')
                yp = 90.0 - y_coord[mask_north]
                dy = (np.floor(100.0 * (yp[1] - yp[0])) / 100.0) / 2.0
                yp = np.append(yp - dy, yp[-1] + dy)
                xp = np.radians(x_coord + shift - 90.0)
                dx = (xp[1] - xp[0]) / 2
                xp = np.append(xp - dx, xp[-1] + dx)
                z = all_2dim_data[itime][:, mask_north].transpose()
                conn = ax2.pcolormesh(xp, yp, z, shading='auto', cmap=cmap,
                                      vmin=mini_north, vmax=maxi_north)
                ax2.set_xticks(xlabelpos)
                ax2.set_xticklabels(xlabels)
                ax2.text(-np.pi / 2, 46.0, '00 LT', verticalalignment='top',
                         horizontalalignment='center')
                ax2.text(np.pi / 2, 46.0, '12 LT', verticalalignment='bottom',
                         horizontalalignment='center')
                ax2.set_yticks(ylabelpos)
                ax2.set_yticklabels(ylabels)
                ax2.grid(linestyle=':', color='black')
                ax2.set_xticks(xticks)
                ax2.set_yticks(yticks)
                ax2.set_ylim([0, 45])
                figs[-1].colorbar(conn, ax=ax2, shrink=0.5, pad=0.01)

            if plot_south:
                # Top Right Graph Southern Hemisphere
                rad, theta = np.meshgrid(90.0 + y_coord[mask_south],
                                         np.radians(x_coord + shift - 90.0))
                ax3 = figs[-1].add_subplot(gs[0, 1], projection='polar')

                yp = 90.0 + y_coord[mask_south]
                dy = (int(100.0 * (yp[1] - yp[0])) / 100.0) / 2.0
                yp = np.append(yp - dy, yp[-1] + dy)
                xp = np.radians(x_coord + shift - 90.0)
                dx = (xp[1] - xp[0]) / 2.0
                xp = np.append(xp - dx, xp[-1] + dx)
                z = all_2dim_data[itime][:, mask_south].transpose()
                cons = ax3.pcolormesh(xp, yp, z, shading='auto', cmap=cmap,
                                      vmin=mini_south, vmax=maxi_south)
                ax3.set_xticks(xlabelpos)
                ax3.set_xticklabels(xlabels)
                ax3.text(-np.pi / 2, 46.0, '00 LT', verticalalignment='top',
                         horizontalalignment='center')
                ax3.text(np.pi / 2, 46.0, '12 LT', verticalalignment='bottom',
                         horizontalalignment='center')
                ax3.set_yticks(ylabelpos)
                ax3.set_yticklabels(ylabels)
                ax3.grid(linestyle=':', color='black')
                ax3.set_xticks(xticks)
                ax3.set_yticks(yticks)
                ax3.set_ylim([0, 45])
                figs[-1].colorbar(cons, ax=ax3, shrink=0.5, pad=0.01)

        # Format the output filename
        fmt_input = itime if movie_rate > 0 else utime.strftime(
            '%y%m%d_%H%M%S')
        outfile = img_file_fmt.format(fmt_input)

        # Save the output file
        if save_plots:
            logger.info("Writing file : ", outfile)
            figs[-1].savefig(outfile)

        if close_plots:
            plt.close(figs[-1])
            figs.pop()

    # Create a movie, if desired
    if save_plots and movie_rate > 0:
        movie_routines.save_movie(filename, ext=ext, rate=movie_rate)

    return figs
