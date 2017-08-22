from contextlib import contextmanager
import os
import os.path as op
import tempfile

import h5py
import numpy as np


BAD_ATTRS = {
    'FileFormat': 'SDB',
    'FormatVersion': '0.5',
    'Writable': 'nope',
    'Created': '2017-01-01 01:23:45',
    'Updated': '2017-01-01 01:23:45',
}


GOOD_ATTRS = {
    'FileFormat': 'SDA',
    'FormatVersion': '1.1',
    'Writable': 'yes',
    'Created': '18-Aug-2017 01:23:45',
    'Updated': '18-Aug-2017 01:23:45',
}


FLOAT_VAL = 3.14159
INT_VAL = 3
BOOL_VAL = True
COMPLEX_VAL = 1.23 + 4.56j
STR_VAL = 'foo'
UNICODE_VAL = u'foo'

# scalars
TEST_SCALARS = [
    (FLOAT_VAL, 'numeric'),
    (np.float32(FLOAT_VAL), 'numeric'),
    (np.float64(FLOAT_VAL), 'numeric'),
    (INT_VAL, 'numeric'),
    (np.long(INT_VAL), 'numeric'),
    (np.int8(INT_VAL), 'numeric'),
    (np.int16(INT_VAL), 'numeric'),
    (np.int32(INT_VAL), 'numeric'),
    (np.int64(INT_VAL), 'numeric'),
    (np.uint8(INT_VAL), 'numeric'),
    (np.uint16(INT_VAL), 'numeric'),
    (np.uint32(INT_VAL), 'numeric'),
    (np.uint64(INT_VAL), 'numeric'),
    (COMPLEX_VAL, 'numeric'),
    (np.complex64(COMPLEX_VAL), 'numeric'),
    (np.complex128(COMPLEX_VAL), 'numeric'),
    (BOOL_VAL, 'logical'),
    (np.bool_(BOOL_VAL), 'logical'),
    (STR_VAL, 'character'),
    (np.str_(STR_VAL), 'character'),
    (np.unicode_(UNICODE_VAL), 'character'),
]

# array scalars
TEST_SCALARS += [
    (np.array(val), typ) for val, typ in TEST_SCALARS if typ != 'character'
]


# lists, tuples, and arrays
TEST_ARRAYS = []
for val, typ in TEST_SCALARS:
    if typ != 'character':
        arr = [val] * 4
        TEST_ARRAYS.append((arr, typ))
        TEST_ARRAYS.append((tuple(arr), typ))
        TEST_ARRAYS.append((np.array(arr), typ))
        TEST_ARRAYS.append((np.array(arr).reshape(2, 2), typ))


# Unsupported
TEST_UNSUPPORTED = [
    np.array(['hi', 'hello']),  # no arrays of strings
    np.array([3, 'hello'], dtype=object),
    lambda x: x**2,
    {0: 0},
    {0},
    None,
]


@contextmanager
def temporary_file(suffix='.sda'):
    pid, file_path = tempfile.mkstemp(suffix=suffix)
    os.close(pid)
    try:
        yield file_path
    finally:
        if op.isfile(file_path):
            os.remove(file_path)


@contextmanager
def temporary_h5file(suffix='.sda'):
    with temporary_file(suffix) as file_path:
        h5file = h5py.File(file_path, 'w')
        try:
            yield h5file
        finally:
            if h5file.id.valid:  # file is open
                h5file.close()
