#!/usr/bin/env python
# Copyright 2020, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md
"""Utilities for outputing movies."""

from glob import glob
import os
import re


def setup_movie_dir(movie_dir, file_glob="image_????.png", overwrite=True):
    """Set up a directory for movie files.

    Parameters
    ----------
    movie_dir : str
        Output filename with directory, but no extention
    file_glob : str
        Base filename without directory, using wildcards to identify potential
        images that would be included in the movie (default='image_????.png')
    overwrite : bool
        Overwrite an existing movie of the same name. Windows OS frequently do
        not allow Python to remove files, so Windows users must clean up these
        files themselves. (default=True)

    Returns
    -------
    img_names : str
        Image name formatting string that may be used for `save_movie`

    Raises
    ------
    IOError
        If image files already exist and `overwrite` is False

    """

    # Test the output directory for existence and existing image files
    if os.path.isdir(movie_dir):
        oldfiles = glob(os.path.join(movie_dir, file_glob))
        if len(oldfiles) > 0:
            if overwrite:
                for ofile in oldfiles:
                    os.remove(ofile)
            else:
                raise IOError('files present in movie directory: {:}'.format(
                    movie_dir))
    else:
        os.makedirs(movie_dir)

    # Create the movie image naming string based on the `file_glob` variable
    file_base, file_ext = os.path.splitext(file_glob)
    dnum = len(file_base.split('?')) - 1 + 4 * (len(file_base.split("*")) - 1)
    file_pre = re.split(r"\W", file_base)[0]
    img_names = os.path.join(movie_dir, "".join([
        file_pre, "_%0", "{:d}".format(dnum), "d", file_ext]))

    return img_names


def save_movie(movie_dir, movie_name="movie.mp4", image_files="image_%04d.png",
               rate=30, overwrite=True):
    """Save the output as a movie.

    Parameters
    ----------
    movie_dir : str
        Output directory for the movie
    move_name : str
        Output movie name with extention (default='movie.mp4')
    image_files : str
        Full-path names for images to be used in the movie, using C-style
        descriptors for numbers that indicate the order of file inclusion
        within the movie.  For example, 'image_0001.png' and 'image_0002.png'
        would create a 2 frame movie with 'image_0001.png' appearing first
        using the default value as long as the files exist in the directory
        where the code is run (img_names='image_%04d.png')
    rate : int
        Movie frame rate (default=30)
    overwrite : bool
        Overwrite an existing movie of the same name. Windows OS frequently do
        not allow Python to remove files, so Windows users must clean up these
        files themselves. (default=True)

    Returns
    -------
    outfile : str
        Output movie name

    Raises
    ------
    IOError
        If movie file already exists and `overwrite` is False.

    Notes
    -----
    Uses ffmpeg to create the movie, this must be installed for success

    """
    # Construct the output filenames
    outfile = os.path.join(movie_dir, movie_name)

    # Test the output file
    if os.path.isfile(outfile):
        if overwrite:
            os.remove(outfile)
        else:
            raise IOError('movie file {:} already exists'.format(outfile))

    # Construct the movie commannd
    command = "ffmpeg -r {:d} -i {:s} {:s}".format(rate, image_files, outfile)
    os.system(command)

    return outfile
