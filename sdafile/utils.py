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
from scipy.sparse import issparse

from .exceptions import BadSDAFile


# DD-MMM-YYYY HH:MM:SS
# MATLAB code uses 'datestr' to create this. HH:MM:SS is optional if all zero.
# This variant is DATE_FORMAT_SHORT.
DATE_FORMAT = "%d-%b-%Y %H:%M:%S"
DATE_FORMAT_SHORT = "%d-%b-%Y"

# Record groups.
SIMPLE_RECORD_TYPES = ('character', 'logical', 'numeric', 'file')
SUPPORTED_RECORD_TYPES = (
    'character', 'file', 'logical', 'numeric', 'cell', 'structure',
    'structures', 'object', 'objects',
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
    'file': b'',
    'logical': np.array([], dtype=bool),
    'cell': [],
    'structure': {}
}

# Equivalent record types for reading
STRUCTURE_EQUIVALENT = {'structure', 'object'}
CELL_EQUIVALENT = {'cell', 'objects', 'structures'}


# Cell label template and generator function
CELL_LABEL_TEMPLATE = "element {}"
cell_label = CELL_LABEL_TEMPLATE.format


def are_record_types_equivalent(rt1, rt2):
    """ Determine if record types are equivalent with respect to reading """
    if rt1 == rt2:
        return True

    if rt1 in STRUCTURE_EQUIVALENT and rt2 in STRUCTURE_EQUIVALENT:
        return True

    if rt1 in CELL_EQUIVALENT and rt2 in CELL_EQUIVALENT:
        return True

    return False


def are_signatures_equivalent(sig1, sig2):
    """ Verify if data signatures are equivalent.

    Parameters
    ----------
    sig1, sig2 :
        Data or group signatures returned by unnest or unnest_record.

    """

    for item1, item2 in zip(sig1, sig2):
        key1, rt1 = item1
        key2, rt2 = item2
        if key1 != key2:
            return False

        if not are_record_types_equivalent(rt1, rt2):
            return False

    return True


def coerce_simple(record_type, data, extra):
    """ Coerce a simple record based on its record type.

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
    elif record_type == 'file':
        data = coerce_file(data)
    else:
        # Should not happen
        msg = "Unrecognized record type '{}'".format(record_type)
        raise ValueError(msg)
    return data, original_shape


def coerce_character(data):
    """ Coerce 'character' data to uint8 stored form

    Parameters
    ----------
    data : str, unicode, or ndarray of '|S1'
        Input string or array of characters

    Returns
    -------
    coerced : ndarray
        The string, encoded as ascii, and stored in a uint8 array

    """
    if isinstance(data, np.ndarray):
        data = data.view(np.uint8)
    else:
        data = np.frombuffer(data.encode('ascii'), np.uint8)
    return np.atleast_2d(data)


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
    data = np.atleast_2d(data).ravel(order='F')
    return np.array([data.real, data.imag])


def coerce_file(data):
    """ Coerce a file object.

    Parameters
    ----------
    data : file-like
        An open file to read

    Returns
    -------
    coerced : ndarray
        The contents of the file as a byte array

    """
    contents = data.read()
    return np.atleast_2d(np.frombuffer(contents, dtype=np.uint8))


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
    return np.atleast_2d(data)


def coerce_numeric(data):
    """ Coerce complex 'numeric' data

    Parameters
    ----------
    data : array-like or scalar
        Integer or floating-point values

    Returns
    -------
    coerced : array-like or scalar
        The data with at least 2 dimensions

    """
    return np.atleast_2d(data)


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
    is_empty : bool
        Flag indicating whether the data is empty.
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
        Arrays of characters (dtype 'S1') are inferred to be 'character' type.
        Arrays of 'object' and multi-character string type are inferred to be
        'cell' arrays.

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

    file-like :
        File-like objects (with a 'read' method) are inferred to be 'file'
        records.

    Anything not listed above is not supported.

    """

    UNSUPPORTED = None, None, None, None

    # Unwrap scalar arrays to simplify the following
    is_scalar = np.isscalar(obj)
    is_array = isinstance(obj, np.ndarray)
    while is_scalar and is_array:
        obj = obj.item()
        is_scalar = np.isscalar(obj)
        is_array = isinstance(obj, np.ndarray)

    if isinstance(obj, (str, np.unicode)):  # Numpy string type is a str
        is_empty = len(obj) == 0
        extra = None
        return 'character', obj, is_empty, extra

    if is_array and obj.dtype == np.dtype('S1'):
        is_empty = obj.size == 0
        extra = None
        return 'character', obj, is_empty, extra

    if isinstance(obj, collections.Sequence):
        is_empty = len(obj) == 0
        extra = None
        return 'cell', obj, is_empty, extra

    if issparse(obj):
        if obj.dtype.char in UNSUPPORTED_NUMERIC_TYPE_CODES:
            return UNSUPPORTED
        if not np.issubdtype(obj.dtype, np.number):
            return UNSUPPORTED
        is_empty = np.prod(obj.shape) == 0
        extra = 'sparse'
        if np.issubdtype(obj.dtype, np.complexfloating):
            extra += '+complex'
        return 'numeric', obj.tocoo(), is_empty, extra

    if np.iscomplexobj(obj):
        if np.asarray(obj).dtype.char in UNSUPPORTED_NUMERIC_TYPE_CODES:
            return UNSUPPORTED
        if is_scalar:
            is_empty = np.isnan(obj.real) and np.isnan(obj.complex)
        else:
            is_empty = np.isnan(obj.real).all() and np.isnan(obj.complex).all()
        extra = 'complex'
        return 'numeric', obj, is_empty, extra

    if isinstance(obj, collections.Mapping):
        is_empty = len(obj) == 0
        extra = None
        return 'structure', obj, is_empty, extra

    if hasattr(obj, 'read'):
        is_empty = False
        extra = None
        return 'file', obj, is_empty, extra

    # numeric and logical scalars and arrays
    extra = None
    is_empty = np.asarray(obj).size == 0
    cast_obj = obj
    if is_scalar:
        check = isinstance
        if np.asarray(obj).dtype.char in UNSUPPORTED_NUMERIC_TYPE_CODES:
            return UNSUPPORTED
    elif is_array:
        check = issubclass
        if cast_obj.dtype.char in UNSUPPORTED_NUMERIC_TYPE_CODES:
            return UNSUPPORTED
        obj = cast_obj.dtype.type
    else:
        return UNSUPPORTED

    if check(obj, (bool, np.bool_)):
        return 'logical', cast_obj, is_empty, extra

    if check(obj, (int, np.long, float, np.number)):
        is_empty = is_empty or np.all(np.isnan(cast_obj))
        return 'numeric', cast_obj, is_empty, extra

    if check(obj, (np.object_, np.unicode_, np.str_)):
        return 'cell', cast_obj, is_empty, extra

    return UNSUPPORTED


def is_simple(record_type):
    """ Check if record type is simple (primitive or 'file'). """
    return record_type in SIMPLE_RECORD_TYPES


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


def get_record_type(dict_like):
    """ Retrived decoded record type from a dict-like object. """
    return get_decoded(dict_like, 'RecordType').get('RecordType')


def unnest(data):
    """ Unnest data.

    Parameters
    ----------
    data :
        Data to unnest

    Returns
    -------
    record_signature :
        A list of (path, record_type) tuples where each component of the data
        is identified in the path.

    """
    record_type, _, _, _ = infer_record_type(data)
    items = [('', record_type, data)]
    for parent, record_type, obj in items:
        if record_type in SIMPLE_RECORD_TYPES:
            continue
        if are_record_types_equivalent(record_type, 'structure'):
            sub_items = sorted(obj.items())
        elif are_record_types_equivalent(record_type, 'cell'):
            sub_items = [
                (cell_label(i), sub_obj)
                for i, sub_obj in enumerate(obj)
            ]
        for key, sub_obj in sub_items:
            path = "/".join((parent, key)).lstrip("/")
            sub_record_type, _, _, _ = infer_record_type(sub_obj)
            items.append((path, sub_record_type, sub_obj))
    return [item[:2] for item in items]


def unnest_record(grp):
    """ Unnest a record group stored on file.

    Parameters
    ----------
    grp : h5py.Group
        The group to unnest.

    Returns
    -------
    record_signature :
        A list of (path, record_type) tuples where each component of the group
        is identified in the path.

    """
    record_type = get_record_type(grp.attrs)
    items = [('', record_type, grp)]
    for parent, record_type, obj in items:
        if record_type not in SIMPLE_RECORD_TYPES:
            for key in sorted(obj.keys()):
                path = "/".join((parent, key)).lstrip("/")
                sub_obj = obj[key]
                sub_record_type = get_record_type(sub_obj.attrs)
                items.append((path, sub_record_type, sub_obj))
    return [item[:2] for item in items]


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
