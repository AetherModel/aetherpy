#!/usr/bin/env python
# Copyright 2020, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md
"""Standard model visualization routines."""

from glob import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import re

from aetherpy.io import read_routines
from aetherpy import plot
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
        os.path.commonpath([inputs.__file__, plot.data_prep.__file__]),
        'run_plot_model_results.py') if __name__ == '__main__' else __name__

    help_str = ''.join(['Usage:\n{:s} -[flags] [filenames]\n'.format(mname),
                        'Flags:\n',
                        '       -help : print this message, include filename ',
                        'for variable names and indices\n',
                        '       -var=string : name of variable to plot\n',
                        '       -ivar=number : index of variable to plot\n',
                        '       -cut=alt, lat, or lon : which cut you would ',
                        'like\n',
                        '       -alt=number : alt in km (closest)\n',
                        '       -lat=number : latitude in degrees (closest)\n',
                        '       -lon=number : longitude in degrees (closest)\n',
                        '       -log : plot the log of the variable\n',
                        '       -winds : overplot winds\n',
                        '       -tec : plot the TEC variable\n',
                        '       -movie=number : provide a positive frame rate',
                        ' to create a movie\n',
                        '       -ext=str : figure or movie extension\n',
                        'At end, list the files you want to plot. This code ',
                        'should work with either GITM files (*.bin) or Aether',
                        ' netCDF files (*.nc)\n',
                        'If both -var and -ivar are provided, -var is used'])

    if file_vars is not None:
        help_str += "File Variables (index, name):\n"
        for ivar, var in enumerate(file_vars):
            help_str += "               ({:d}, {:s})\n".format(ivar, var)

    return help_str


def get_command_line_args(argv):
    """Parse the arguements and set to a dictionary.

    Parameters
    ----------
    argv : list
        List of arguments fed on the command line

    Returns
    -------
    args : dict
        A dictionary containing information about arguements, including:
        filelist (list of filenames), gitm (flag that is true for GITM input,
        determined by examining filelist naming convention),
        var (variable index to plot), cut (coordinate to hold constant),
        diff (difference with other plots),
        movie (framerate for movie, which is > 0 if a movie is desired),
        ext (output extension), winds (flag to plot with winds),
        alt (to plot), lat (to plot), lon (to plot),
        log (flag to use log scale), and help (flag to display help)

    """

    # Initialize the arguments to their default values
    args = {'filelist': [], 'log': False, 'var': '', 'ivar': -1, 'alt': 400,
            'tec': False, 'lon': np.nan, 'lat': np.nan, 'cut': 'alt',
            'winds': False, 'is_gitm': False, 'has_header': False, 'movie': 0,
            'ext': 'png'}

    arg_type = {'filelist': list, 'log': bool, 'var': str, 'ivar': int,
                'alt': int, 'tec': bool, 'lon': float, 'lat': float,
                'cut': str, 'help': bool, 'winds': bool, 'is_gitm': bool,
                'has_header': bool, 'tec': bool, 'movie': int, 'ext': str}

    # If there is input, set default help to False
    args['help'] = False if len(argv) > 0 else True

    # Cycle through all arguments except the first, saving input
    for arg in argv:
        # Treat the file list and formatting seperately
        if arg.find('-') == 0:
            # This is not a filename, remove the dash to get the key
            split_arg = arg.split('=')
            akey = split_arg[0][1:]

            # Get the argument value as the desired type
            if akey not in arg_type.keys():
                raise ValueError(''.join(['unknown command line input, ',
                                          arg, ', try -help for details']))

            if len(split_arg) == 1:
                if arg_type[akey] == bool:
                    arg_val = True
                else:
                    raise ValueError('expected equality after flag {:}'.format(
                        akey))
            else:
                if arg_type[akey] == int:
                    arg_val = int(split_arg[1])
                elif arg_type[akey] == float:
                    arg_val = float(split_arg[1])
                elif arg_type[akey] == str:
                    arg_val = split_arg[1]
                else:
                    # This is boolean input
                    arg_val = inputs.bool_string(split_arg[1])

            # Assign the output
            if akey.find('tec') == 0:
                args['var'] = 34
            else:
                args[akey] = arg_val
        else:
            # Save the filenames
            args['filelist'].append(arg)

            match_bin = re.match(r'(.*)bin', arg)
            if match_bin:
                args['is_gitm'] = True
                args['has_header'] = False

                # Check for a header file:
                check_file = glob(match_bin.group(1) + "header")
                if len(check_file) > 0:
                    if len(check_file[0]) > 1:
                        args['has_header'] = True
            else:
                args['is_gitm'] = False

    # Update default movie extention for POSIX systems
    if args['movie'] > 0 and args['ext'] == 'png':
        if os.name == "posix":
            args['ext'] = "mkv"
        else:
            args['ext'] = "mp4"

    return args


# ----------------------------------------------------------------------------
# Define the main plotting routine

def main():
    """Main routine for creating standard plots.

    See Also
    --------
    aetherpy.plots.standard_plots.plot_model_results

    """

    # Get the input arguments
    args = get_command_line_args(inputs.process_command_line_input())

    if len(args['filelist']) == 0:
        help_str = get_help()
        print(help_str)
        return

    is_gitm = args['is_gitm']
    has_header = args['has_header']

    # If help is requested for a specific file, return it here
    if args['help'] or (len(args['var']) == 0 and args['ivar'] < 0):
        header = read_routines.read_headers(args['filelist'],
                                            has_header=has_header,
                                            is_gitm=is_gitm)
        help_str = get_help(header['vars'])
        print(help_str)
        return

    if len(args['var']) == 0:
        plot_var = args['ivar']
    else:
        plot_var = args['var']

    # Load the data needed for plotting
    (all_times, all_2dim_data, all_winds_x, all_winds_y, icut, x_coord,
     y_coord, z_val, var_name) = plot.data_prep.load_data_for_plotting(
         args['filelist'], plot_var, args['cut'], cut_val=args[args['cut']],
         has_header=has_header, is_gitm=is_gitm, winds=args['winds'],
         tec=args['tec'])

    # Plot the data using the specified keyword arguments from the command line
    plot.standard_plots.plot_model_results(
        all_times, all_2dim_data, all_winds_x, all_winds_y, var_name,
        args['cut'], icut, x_coord, y_coord, z_val, log_scale=args['log'],
        ext=args['ext'], movie_rate=args['movie'])

    return


# Needed to run main script as the default executable from the command line
if __name__ == '__main__':
    main()
