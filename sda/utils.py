""" Utility functions and data. """

from datetime import datetime
import re
import time

import numpy as np

from .exceptions import BadSDAFile


# DD-MMM-YYYY HH:MM:SS
# MATLAB code uses 'datestr' to create this. HH:MM:SS is optional if all zero.
# This variant is DATE_FORMAT_SHORT.
DATE_FORMAT = "%d-%b-%Y %H:%M:%S"
DATE_FORMAT_SHORT = "%d-%b-%Y"

# Regular expression for version string
VERSION_1_RE = re.compile('1\.\d+')


def error_if_bad_attr(h5file, attr, is_valid):
    """ Raise BadSDAFile error if h5file has a bad SDA attribute. """
    name = h5file.filename
    try:
        value = h5file.attrs[attr]
    except KeyError:
        msg = "File '{}' does not contain '{}' attribute".format(name, attr)
        raise BadSDAFile(msg)
    else:
        if not is_valid(value):
            msg = "File '{}' has invalid '{}' attribute".format(name, attr)
            raise BadSDAFile(msg)


def error_if_bad_header(h5file):
    """ Raise BadSDAFile if SDA header attributes are missing or invalid. """
    # FileFormat flag
    error_if_bad_attr(h5file, 'FileFormat', is_valid_file_format)

    # FormatVersion flag
    error_if_bad_attr(h5file, 'FormatVersion', is_valid_format_version)

    # Writable flag
    error_if_bad_attr(h5file, 'Writable', is_valid_writable)

    # Created flag
    error_if_bad_attr(h5file, 'Created', is_valid_date)

    # Modified flag
    error_if_bad_attr(h5file, 'Modified', is_valid_date)


def error_if_not_writable(h5file):
    """ Raise an IOError if an SDAFile indicates 'Writable' as 'no'. """
    writable = h5file.attrs.get('Writable')
    if writable == 'no':
        msg = "File '{}' is not writable".format(h5file.filename)
        raise IOError(msg)


def get_date_str(dt=None):
    """ Get a valid date string from a datetime, or current time. """
    if dt is None:
        dt = datetime.now()
    if dt.hour == dt.minute == dt.second == 0:
        fmt = DATE_FORMAT_SHORT
    else:
        fmt = DATE_FORMAT
    date_str = dt.strftime(fmt)
    return date_str


def infer_record_type(obj):
    """ Infer record type of ``obj``.

    Supported types are 'numeric', 'bool', and 'character'.

    Parameters
    ----------
    obj :
        An object to store

    Returns
    -------
    record_type : str or None
        The inferred record type, or None if the object type is not supported.
    cast_obj :
        The object if scalar, the object cast as a numpy array if not, or None
        the type is unsupported.

    """

    if np.isscalar(obj):
        check = isinstance
        cast_obj = obj
    else:
        check = issubclass
        cast_obj = np.asarray(obj)
        obj = cast_obj.dtype.type

    if check(obj, (bool, np.bool_)):
        return 'logical', cast_obj

    if check(obj, (int, float, complex, np.number)):
        return 'numeric', cast_obj

    # Only accept strings, not arrays of strings
    if isinstance(obj, str):  # Numpy strings are also str
        return 'character', cast_obj

    return None, None


def is_valid_date(date_str):
    """ Check date str conforms to DATE_FORMAT or DATE_FORMAT_SHORT. """
    try:
        time.strptime(date_str, DATE_FORMAT)
    except ValueError:
        try:
            time.strptime(date_str, DATE_FORMAT_SHORT)
        except ValueError:
            return False
    return True


def is_valid_file_format(value):
    """ Check that file format is equivalent to 'SDA' """
    return value == 'SDA'


def is_valid_format_version(value):
    """ Check that version is '1.X' """
    return VERSION_1_RE.match(value) is not None


def is_valid_writable(value):
    """ Check that writable flag is 'yes' or 'no' """
    return value == 'yes' or value == 'no'


def write_header(attrs):
    """ Write default header values to dict-like ``attrs``. """
    attrs['FileFormat'] = 'SDA'
    attrs['FormatVersion'] = '1.0'
    attrs['Writable'] = 'yes'
    date_str = get_date_str()
    attrs['Created'] = date_str
    attrs['Modified'] = date_str
