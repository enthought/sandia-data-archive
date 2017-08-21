""" Utility functions and data. """

from datetime import datetime
import re
import time

from .exceptions import BadSDAFile


# DD-MMM-YYYY HH:MM:SS
# MATLAB code uses 'datestr' to create this. HH:MM:SS is optional if all zero.
# This variant is DATE_FORMAT_SHORT.
DATE_FORMAT = "%d-%b-%Y %H:%M:%S"
DATE_FORMAT_SHORT = "%d-%b-%Y"

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
    if dt is None:
        dt = datetime.now()
    if dt.hour == dt.minute == dt.second == 0:
        fmt = DATE_FORMAT_SHORT
    else:
        fmt = DATE_FORMAT
    date_str = dt.strftime(fmt)
    return date_str


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
    return value == 'SDA'


def is_valid_format_version(value):
    return VERSION_1_RE.match(value) is not None


def is_valid_writable(value):
    return value == 'yes' or value == 'no'


def write_header(attrs):
    attrs['FileFormat'] = 'SDA'
    attrs['FormatVersion'] = '1.0'
    attrs['Writable'] = 'yes'
    date_str = get_date_str()
    attrs['Created'] = date_str
    attrs['Modified'] = date_str
