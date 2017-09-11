""" Utility functions and data.

The functions in the module work directly on data and metadata. In order to
make this easy to write and test, functionality that requires direct
interaction with HDF5 are not included here.

"""

import collections
from datetime import datetime
import re
import string
import time

import numpy as np
from scipy.sparse import coo_matrix, issparse

from .exceptions import BadSDAFile


# DD-MMM-YYYY HH:MM:SS
# MATLAB code uses 'datestr' to create this. HH:MM:SS is optional if all zero.
# This variant is DATE_FORMAT_SHORT.
DATE_FORMAT = "%d-%b-%Y %H:%M:%S"
DATE_FORMAT_SHORT = "%d-%b-%Y"

# Record groups.
# 'structures', 'object', and 'objects' are read- and replace-only
PRIMITIVE_RECORD_TYPES = ('character', 'logical', 'numeric')
SUPPORTED_RECORD_TYPES = (
    'character', 'logical', 'numeric', 'cell', 'structure', 'structures',
    'object', 'objects',
)


# Regular expression for version string
VERSION_1_RE = re.compile(r'1\.(?P<sub>\d+)')


# Type codes for unsupported numeric types
UNSUPPORTED_NUMERIC_TYPE_CODES = {
    'G',  # complex256
    'g',  # float128
    'e',  # float16
}

# Empty values for supported types
EMPTY_FOR_TYPE = {
    'numeric': np.nan,
    'character': '',
    'logical': np.array([], dtype=bool),
    'cell': [],
    'structure': {}
}


def coerce_primitive(record_type, data, extra):
    """ Coerce a primitive type based on its record type.

    Parameters
    ----------
    record_type : str or None
        The record type.
    data :
        The object if scalar, the object cast as a numpy array if not, or None
        the type is unsupported.
    extra :
        Extra information about the type returned by ``infer_record_type``.

    Returns
    -------
    coerced :
        The data coerced to storage form.
    original_shape : tuple or None
        The original shape of the data. This is only returned for complex and
        sparse-complex data.

    """
    original_shape = None
    if record_type == 'numeric':
        if extra == 'complex':
            original_shape = np.atleast_2d(data).shape
            data = coerce_complex(data)
        elif extra == 'sparse':
            data = coerce_sparse(data)
        elif extra == 'sparse+complex':
            original_shape = data.shape
            data = coerce_sparse_complex(data)
        else:
            data = coerce_numeric(data)
    elif record_type == 'logical':
        data = coerce_logical(data)
    elif record_type == 'character':
        data = coerce_character(data)
    else:
        # Should not happen
        msg = "Unrecognized record type '{}'".format(record_type)
        raise ValueError(msg)
    return data, original_shape


def coerce_character(data):
    """ Coerce 'character' data to uint8 stored form

    Parameters
    ----------
    data : str
        Input string

    Returns
    -------
    coerced : ndarray
        The string, encoded as ascii, and stored in a uint8 array

    """
    data = np.frombuffer(data.encode('ascii'), np.uint8)
    return data


def coerce_complex(data):
    """ Coerce complex 'numeric' data

    Parameters
    ----------
    data : array-like or complex
        Input complex value

    Returns
    -------
    coerced : ndarray
        2xN array containing the real and imaginary values of the input as rows
        0 and 1. This will have type float32 if the input type is complex64
        and type float64 if the input type is complex128 (or equivalent).

    """
    data = np.asarray(data).ravel(order='F')
    return np.array([data.real, data.imag])


def coerce_logical(data):
    """ Coerce 'logical' data to uint8 stored form

    Parameters
    ----------
    data : array-like or bool
        Input boolean value

    Returns
    -------
    coerced : ndarray of uint8 or uint8
        Scalar or array containing the input data coereced to uint8, clipped to
        0 or 1.

    """
    if np.isscalar(data) or data.shape == ():
        data = np.uint8(1 if data else 0)
    else:
        data = data.astype(np.uint8).clip(0, 1)
    return data


def coerce_numeric(data):
    """ Coerce complex 'numeric' data

    Parameters
    ----------
    data : array-like or scalar
        Integer or floating-point values

    Returns
    -------
    data : array-like or scalar
        This function does not modify the input data.

    """
    return data


def coerce_sparse(data):
    """ Coerce sparse 'numeric' data to stored form.

    Parameters
    ----------
    data : scipy.sparse.coo_matrix
        Input sparse matrix.

    Returns
    -------
    coerced : ndarray
        3xN array containing the rows, columns, and values of the sparse matrix
        in COO form. Note that the row and column arrays are 1-based to be
        compatible with MATLAB.

    """
    # 3 x N, [row, column, value], 1-based
    return np.array([data.row + 1, data.col + 1, data.data])


def coerce_sparse_complex(data):
    """ Coerce sparse and complex 'numeric' data to stored form.

    Parameters
    ----------
    data : scipy.sparse.coo_matrix
        Input sparse matrix.

    Returns
    -------
    coerced : ndarray
        3xN array containing the index, real, and imaginary values of the
        sparse complex data. The index is unraveled and 1-based. The original
        array shape is required to re-ravel the index and reconstitute the
        sparse, complex data.

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


def extract_primitive(record_type, data, data_attrs):
    """ Extract primitive data from its raw storage format.

    Parameters
    -----------
    data : ndarray
        Data extracted from hdf5 dataset storage
    record_type : str
        The primitive data type
    data_attrs : dict
        Attributes associated with the stored dataset

    Returns
    -------
    extracted :
        The extracted primitive data

    """
    complex_flag = data_attrs.get('Complex', 'no')
    sparse_flag = data_attrs.get('Sparse', 'no')
    shape = data_attrs.get('ArraySize', None)

    if record_type == 'numeric':
        if sparse_flag == 'yes':
            if complex_flag == 'yes':
                extracted = extract_sparse_complex(data, shape.astype(int))
            else:
                extracted = extract_sparse(data)
        elif complex_flag == 'yes':
            extracted = extract_complex(data, shape.astype(int))
        else:
            extracted = extract_numeric(data)
        # squeeze leading dimension if this is a MATLAB row array
        if extracted.ndim == 2 and extracted.shape[0] == 1:
            # if it's a scalar, go all the way
            if extracted.shape[1] == 1:
                extracted = extracted[0, 0]
            else:
                extracted = np.squeeze(extracted, axis=0)
    elif record_type == 'logical':
        extracted = extract_logical(data)
    elif record_type == 'character':
        extracted = extract_character(data)

    return extracted


def extract_character(data):
    """ Extract 'character' data from uint8 stored form.

    Parameters
    -----------
    data : ndarray
        Array of uint8 ascii encodings

    Returns
    -------
    extracted : str
        Reconstructed ascii string.

    """
    data = data.tobytes().decode('ascii')
    return data


def extract_complex(data, shape):
    """ Extract complex 'numeric' data.

    Parameters
    -----------
    data : ndarray
        2 x N array containing real and imaginary portions of the complex data.
    shape : tuple
        Shape of the extracted array.

    Returns
    -------
    extracted : ndarray
        The extracted complex array.

    """
    extracted = 1j * data[1]
    extracted.real = data[0]
    return extracted.reshape(shape, order='F')


def extract_logical(data):
    """ Extract 'logical' data from uint8 stored form.

    Parameters
    -----------
    data : ndarray or scalar
        Array or scalar of uint8 values clipped to 0 or 1

    Returns
    -------
    extracted :
        The extracted boolean or boolean array

    """
    if np.isscalar(data):
        data = bool(data)
    else:
        data = data.astype(bool)
    return data


def extract_numeric(data):
    """ Extract 'numeric' data from stored form.

    Parameters
    -----------
    data : ndarray or scalar
        Array or scalar of numeric data

    Returns
    -------
    data : ndarray or scalar
        The input data

    """
    return data


def extract_sparse(data):
    """ Extract sparse 'numeric' data from stored form.

    Parameters
    -----------
    data : 3xN ndarray
        3xN array containing the rows, columns, and values of a sparse matrix
        in COO form. Note that the row and column arrays must be 1-based to be
        compatible with MATLAB.

    Returns
    -------
    extracted : scipy.sparse.coo_matrix
        The extracted sparse matrix

    """
    row, col, data = data
    # Fix 1-based indexing
    row -= 1
    col -= 1
    return coo_matrix((data, (row, col)))


def extract_sparse_complex(data, shape):
    """ Extract sparse 'numeric' data from stored form.

    Parameters
    -----------
    data : ndarray
        3xN array containing the index, real, and imaginary values of a
        sparse complex data. The index is unraveled and 1-based.
    shape : tuple
        Shape of the extracted array

    Returns
    -------
    extracted : coo_matrix
        The extracted sparse, complex matrix

    """
    index = data[0].astype(np.int64)
    # Fix 1-based indexing
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
    try:
        return EMPTY_FOR_TYPE[record_type]
    except KeyError:
        msg = "Record type '{}' cannot be empty".format(record_type)
        raise ValueError(msg)


def infer_record_type(obj):
    """ Infer record type of ``obj``.

    Supported types are 'numeric', 'bool', 'character', and 'cell'.

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

    Notes
    -----
    The inference routines are unambiguous, and require the user to understand
    the input data in reference to these rules. The user has flexibility to
    coerce data before attempting to store it to have it be stored as a desired
    type.

    sequences :
        Lists, tuples, and anything else that identifies as a
        collections.Sequence are always inferred to be 'cell' records, no
        matter the contents.

    mappings :
        Dictionaries and anything else that identifies as
        collections.Mapping and not another type listed here are inferred to be
        'structure' records.

    numpy arrays :
        If the dtype is a supported numeric type, then the 'numeric' record
        type is inferred. Arrays of 'bool' type are inferred to be 'logical'.
        Arrays of 'object' and string type are inferred to be 'cell' arrays.

    sparse arrays (from scipy.sparse) :
        These are inferred to be 'numeric' and 'sparse', if the dtype is a type
        supported for numpy arrays.

    strings :
        These are always inferred to be 'character' type. An attempt will be
        made to convert the input to ascii encoded bytes, no matter the
        underlying encoding. This may result in an encoding exception if the
        input cannot be ascii encoded.

    non-string scalars :
        Non-string scalars are inferred to be 'numeric' if numeric, or
        'logical' if boolean.

    other :
        Arrays of characters are not supported. Convert to a string.

    Anything not listed above is not supported.

    """

    # Unwrap scalar arrays to simplify the following
    is_scalar = np.isscalar(obj)
    is_array = isinstance(obj, np.ndarray)
    while is_scalar and is_array:
        obj = obj.value()
        is_scalar = np.isscalar(obj)
        is_array = isinstance(obj, np.ndarray)

    if isinstance(obj, (str, np.unicode)):  # Numpy string type is a str
        return 'character', obj, None

    if isinstance(obj, collections.Sequence):
        return 'cell', obj, None

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

    if isinstance(obj, collections.Mapping):
        return 'structure', obj, None

    # numeric and logical scalars and arrays
    cast_obj = obj
    if is_scalar:
        check = isinstance
        if np.asarray(obj).dtype.char in UNSUPPORTED_NUMERIC_TYPE_CODES:
            return None, None, None
    elif is_array:
        check = issubclass
        if cast_obj.dtype.char in UNSUPPORTED_NUMERIC_TYPE_CODES:
            return None, None, None
        obj = cast_obj.dtype.type
    else:
        return None, None, None

    if check(obj, (bool, np.bool_)):
        return 'logical', cast_obj, None

    if check(obj, (int, np.long, float, np.number)):
        return 'numeric', cast_obj, None

    if check(obj, (np.object_, np.unicode_, np.str_)):
        return 'cell', cast_obj, None

    return None, None, None


def is_primitive(record_type):
    """ Check if record type is primitive. """
    return record_type in PRIMITIVE_RECORD_TYPES


def is_supported(record_type):
    """ Check if record type is supported. """
    return record_type in SUPPORTED_RECORD_TYPES


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


def is_valid_matlab_field_label(label):
    """ Check that passed string is a valid MATLAB field label """
    if not label.startswith(tuple(string.ascii_letters)):
        return False
    VALID_CHARS = set(string.ascii_letters + string.digits + "_")
    return set(label).issubset(VALID_CHARS)


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
    """ Update timestamp and version to 1.1 in a header. """
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
