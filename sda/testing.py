from contextlib import contextmanager
import os
import os.path as op
import tempfile

import h5py
import numpy as np
from scipy.sparse import coo_matrix

import sda


DATA_DIR = op.join(op.abspath(op.dirname(sda.__file__)), 'tests', 'data')


def data_path(fname):
    """ Get path to data in test data directory. """
    return op.join(DATA_DIR, fname)


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


# arrays
TEST_ARRAYS = []
for val, typ in TEST_SCALARS:
    if typ != 'character':
        arr = np.array([val] * 4)
        TEST_ARRAYS.append((arr, typ))
        TEST_ARRAYS.append((np.array(arr).reshape(2, 2), typ))


# Sparse matrix in all forms
TEST_SPARSE = [coo_matrix((np.arange(5), (np.arange(1, 6), np.arange(2, 7))))]
TEST_SPARSE.extend([
    TEST_SPARSE[0].tocsr(), TEST_SPARSE[0].tocsc(), TEST_SPARSE[0].tolil(),
    TEST_SPARSE[0].tobsr(), TEST_SPARSE[0].todok()
])

TEST_SPARSE_COMPLEX = [
    coo_matrix((np.arange(5) * (1 + 2j), (np.arange(1, 6), np.arange(2, 7))))
]
TEST_SPARSE_COMPLEX.extend([
    TEST_SPARSE_COMPLEX[0].tocsr(), TEST_SPARSE_COMPLEX[0].tocsc(),
    TEST_SPARSE_COMPLEX[0].tolil(), TEST_SPARSE_COMPLEX[0].tobsr(),
    TEST_SPARSE_COMPLEX[0].todok()
])


# lists, tuples
TEST_CELL = [
    ['hi', 'hello'],
    np.array(['hi', 'hello']),
    ['hello', np.arange(4)],
    ['hello', [True, np.arange(4)]],
    ['hello', (True, np.arange(4))],
    np.array(['hello', 3, [True, False, True], 3.14], dtype=object),
    np.array(
        [
            ['hello', 3],
            [[True, False, True], 3.14]
        ],
        dtype=object
    ),
    np.array(
        [
            ['hello', 3],
            [[True, False, True], 3.14]
        ],
        dtype=object,
        order='F',
    )
]

TEST_STRUCTURE = [
    {
        'foo': 'foo',
        'bar': np.arange(4),
        'baz': np.array([True, False])
    },
    {
        'foo': 'foo',
        'bar': [np.arange(4), np.array([True, False])]
    },
]


# Unsupported
TEST_UNSUPPORTED = [
    lambda x: x**2,
    {0},
    None,
]


# unsupported types, platform-specific
if hasattr(np, 'complex256'):
    TEST_UNSUPPORTED.append(np.complex256(0))
    TEST_UNSUPPORTED.append(np.arange(5, dtype=np.complex256))
if hasattr(np, 'float128'):
    TEST_UNSUPPORTED.append(np.float128(0))
    TEST_UNSUPPORTED.append(np.arange(5, dtype=np.float128))
if hasattr(np, 'float16'):
    TEST_UNSUPPORTED.append(np.float16(0))
    TEST_UNSUPPORTED.append(np.arange(5, dtype=np.float16))


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
