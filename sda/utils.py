""" Utility functions and data. """

from datetime import datetime
import re
import time

import numpy as np
from scipy.sparse import coo_matrix, issparse

from .exceptions import BadSDAFile


# DD-MMM-YYYY HH:MM:SS
# MATLAB code uses 'datestr' to create this. HH:MM:SS is optional if all zero.
# This variant is DATE_FORMAT_SHORT.
DATE_FORMAT = "%d-%b-%Y %H:%M:%S"
DATE_FORMAT_SHORT = "%d-%b-%Y"

# Regular expression for version string
VERSION_1_RE = re.compile(r'1\.(?P<sub>\d+)')


# Type codes for unsupported numeric types
UNSUPPORTED_NUMERIC_TYPE_CODES = {
    'G',  # complex256
    'g',  # float128
    'e',  # float16
}


def coerce_character(data):
    """ Coerce 'character' data to uint8 stored form. """
    data = np.frombuffer(data.encode('ascii'), np.uint8)
    return data


def coerce_complex(data):
    """ Coerce complex 'numeric' data """
    data = np.asarray(data).ravel(order='F')
    return np.array([data.real, data.imag])


def coerce_logical(data):
    """ Coerce 'logical' data to uint8 stored form. """
    if np.isscalar(data) or data.shape == ():
        data = 1 if data else 0
    else:
        data = data.astype(np.uint8).clip(0, 1)
    return data


def coerce_numeric(data):
    """ Coerce 'numeric' data to stored form. """
    return data


def coerce_sparse(data):
    """ Coerce sparse 'numeric' data to stored form.

    Input is expected to coo_matrix.

    """
    # 3 x N, [row, column, value], 1-based
    return np.array([data.row + 1, data.col + 1, data.data])


def coerce_sparse_complex(data):
    """ Coerse sparse and complex 'numeric data to stored form.

    Input is expected to coo_matrix.

    """
    indices = np.ravel_multi_index((data.row, data.col), data.shape)
    coerced = coerce_complex(data.data)
    return np.vstack([indices + 1, coerced])  # 1-based


def error_if_bad_attr(h5file, attr, is_valid):
    """ Raise BadSDAFile error if h5file has a bad SDA attribute.

    This assumes that the attr is stored as bytes. The passed ``is_valid``
    function should accept the value as a string.

    """
    name = h5file.filename
    try:
        value = h5file.attrs[attr]
    except KeyError:
        msg = "File '{}' does not contain '{}' attribute".format(name, attr)
        raise BadSDAFile(msg)
    else:
        value = value.decode('ascii')
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

    # Updated flag
    error_if_bad_attr(h5file, 'Updated', is_valid_date)


def error_if_not_writable(h5file):
    """ Raise an IOError if an SDAFile indicates 'Writable' as 'no'. """
    writable = h5file.attrs.get('Writable')
    if writable == b'no':
        msg = "File '{}' is not writable".format(h5file.filename)
        raise IOError(msg)


def extract_character(data):
    """ Extract 'character' data from uint8 stored form. """
    data = data.tobytes().decode('ascii')
    return data


def extract_complex(data, shape):
    """ Extract complex 'numeric' data from stored form. """
    dtype = data.dtype
    if dtype == np.float64:
        c_dtype = np.complex128
    elif dtype == np.float32:
        c_dtype = np.complex64
    extracted = np.empty(shape, dtype=c_dtype, order='F')
    flat = extracted.ravel(order='F')
    flat.real = data[0]
    flat.imag = data[1]
    return extracted


def extract_logical(data):
    """ Extract 'logical' data from uint8 stored form. """
    if np.isscalar(data):
        data = bool(data)
    else:
        data = data.astype(bool)
    return data


def extract_numeric(data):
    """ Extract 'numeric' data from stored form. """
    return data


def extract_sparse(data):
    """ Extract sparse 'numeric' data from stored form. """
    row, col, data = data
    # Fix 1-based indexing
    row -= 1
    col -= 1
    return coo_matrix((data, (row, col)))


def extract_sparse_complex(data, shape):
    """ Extract sparse 'numeric' data from stored form. """
    index = data[0].astype(np.int64)
    # Fix 1-based indexing from MATLAB
    index -= 1
    data = extract_complex(data[1:], (data.shape[1],))
    row, col = np.unravel_index(index, shape)
    return coo_matrix((data, (row, col)))


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


def get_empty_for_type(record_type):
    """ Get the empty value for a record.

    Raises
    ------
    ValueError if ``record_type`` does not have an empty entry.

    """
    if record_type == 'numeric':
        return np.nan
    elif record_type == 'character':
        return ''
    elif record_type == 'logical':
        return np.array([], dtype=bool)
    else:
        msg = "Record type '{}' cannot be empty".format(record_type)
        raise ValueError(msg)


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
    extra :
        Extra information about the type. This may be None, 'sparse',
        'complex', or 'sparse+complex' for 'numeric' types, and will be None in
        all other cases.

    """
    if issparse(obj):
        if obj.dtype.char in UNSUPPORTED_NUMERIC_TYPE_CODES:
            return None, None, None
        extra = 'sparse'
        if np.issubdtype(obj.dtype, np.complexfloating):
            extra += '+complex'
        return 'numeric', obj.tocoo(), extra

    if np.iscomplexobj(obj):
        if np.asarray(obj).dtype.char in UNSUPPORTED_NUMERIC_TYPE_CODES:
            return None, None, None
        return 'numeric', obj, 'complex'

    if np.isscalar(obj):
        check = isinstance
        cast_obj = obj

        if np.asarray(obj).dtype.char in UNSUPPORTED_NUMERIC_TYPE_CODES:
            return None, None, None

    else:
        check = issubclass
        cast_obj = np.asarray(obj)
        if cast_obj.dtype.char in UNSUPPORTED_NUMERIC_TYPE_CODES:
            return None, None, None

        obj = cast_obj.dtype.type

    if check(obj, (bool, np.bool_)):
        return 'logical', cast_obj, None

    if check(obj, (int, np.long, float, np.number)):
        return 'numeric', cast_obj, None

    # Only accept strings, not arrays of strings
    if isinstance(obj, (str, np.unicode)):  # Numpy strings are also str
        return 'character', cast_obj, None

    return None, None, None


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
    """ Check that version is '1.X' for X <= 1 """
    m = VERSION_1_RE.match(value)
    if m is None:
        return False
    return 0 <= int(m.group('sub')) <= 1


def is_valid_writable(value):
    """ Check that writable flag is 'yes' or 'no' """
    return value == 'yes' or value == 'no'


def set_encoded(dict_like, **attrs):
    """ Encode and insert values into a dict-like object. """
    encoded = {
        attr: value.encode('ascii') if isinstance(value, str) else value
        for attr, value in attrs.items()
    }
    dict_like.update(encoded)


def get_decoded(dict_like, *attrs):
    """ Retrieve decoded values from a dict-like object if they exist.

    If no attrs are passed, all values are retrieved.

    """
    # Filter for existing
    if len(attrs) == 0:
        items = dict_like.items()
    else:
        items = [
            (attr, dict_like[attr]) for attr in attrs if attr in dict_like
        ]
    return {
        attr: value.decode('ascii') if isinstance(value, bytes) else value
        for attr, value in items
    }


def update_header(attrs):
    """ Update a header to verion 1.1. """
    set_encoded(
        attrs,
        FormatVersion='1.1',
        Updated=get_date_str(),
    )


def write_header(attrs):
    """ Write default, encoded header values to dict-like ``attrs``. """
    date_str = get_date_str()
    set_encoded(
        attrs,
        FileFormat='SDA',
        FormatVersion='1.1',
        Writable='yes',
        Created=date_str,
        Updated=date_str,
    )
