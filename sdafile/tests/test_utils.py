import datetime
import io
from itertools import combinations
import string
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import coo_matrix

from sdafile.exceptions import BadSDAFile
from sdafile.testing import (
    BAD_ATTRS, GOOD_ATTRS, TEST_ARRAYS, TEST_CELL, TEST_SCALARS, TEST_SPARSE,
    TEST_SPARSE_COMPLEX, TEST_STRUCTURE, TEST_UNSUPPORTED, temporary_file,
    temporary_h5file
)
from sdafile.utils import (
    CELL_EQUIVALENT, STRUCTURE_EQUIVALENT, SUPPORTED_RECORD_TYPES,
    are_record_types_equivalent, coerce_character, coerce_complex, coerce_file,
    coerce_logical, coerce_numeric, coerce_simple, coerce_sparse,
    coerce_sparse_complex, error_if_bad_attr, error_if_bad_header,
    error_if_not_writable, get_date_str, get_decoded, get_empty_for_type,
    infer_record_type, is_valid_date, is_valid_file_format,
    is_valid_format_version, is_valid_matlab_field_label, is_valid_writable,
    set_encoded, unnest, update_header, write_header
)


class TestUtils(unittest.TestCase):

    def test_are_record_types_equivalent(self):

        for rt in SUPPORTED_RECORD_TYPES:
            self.assertTrue(are_record_types_equivalent(rt, rt))

        equivalents = []
        for rt1, rt2 in combinations(SUPPORTED_RECORD_TYPES, 2):
            if are_record_types_equivalent(rt1, rt2):
                equivalents.append(sorted((rt1, rt2)))

        expected = []
        for rt1, rt2 in combinations(STRUCTURE_EQUIVALENT, 2):
            expected.append(sorted((rt1, rt2)))

        for rt1, rt2 in combinations(CELL_EQUIVALENT, 2):
            expected.append(sorted((rt1, rt2)))

        self.assertEqual(sorted(equivalents), sorted(expected))

    def test_unnest(self):
        data = dict(a=1, b=True, c='foo')
        answer = unnest(data)
        expected = [
            ('', 'structure'),
            ('a', 'numeric'),
            ('b', 'logical'),
            ('c', 'character')
        ]
        self.assertEqual(answer, expected)

        data = dict(a=1, b=True, c=dict(d='foo', e=5, f=dict(g=6)))
        answer = unnest(data)
        expected = [
            ('', 'structure'),
            ('a', 'numeric'),
            ('b', 'logical'),
            ('c', 'structure'),
            ('c/d', 'character'),
            ('c/e', 'numeric'),
            ('c/f', 'structure'),
            ('c/f/g', 'numeric'),
        ]
        self.assertEqual(answer, expected)

    def test_coerce_character(self):
        coerced = coerce_character(string.printable)
        self.assertEqual(coerced.dtype, np.dtype(np.uint8))
        expected = np.array([[ord(c) for c in string.printable]], np.uint8)
        assert_array_equal(coerced, expected)

    def test_coerce_simple(self):
        with self.assertRaises(ValueError):
            coerce_simple('foo', 0, None)

    def test_coerce_complex(self):
        data = np.arange(6, dtype=np.complex128)
        data.imag = -1

        expected = np.array([data.real, data.imag], dtype=np.float64)
        coerced = coerce_complex(data)
        self.assertEqual(expected.dtype, coerced.dtype)
        assert_array_equal(expected, coerced)

        data = data.astype(np.complex64)
        expected = expected.astype(np.float32)
        coerced = coerce_complex(data)
        self.assertEqual(expected.dtype, coerced.dtype)
        assert_array_equal(expected, coerced)

        # Flattened arrays are intrinsically column-major, because MATLAB
        f_data = data.reshape((2, 3), order='F')
        coerced = coerce_complex(f_data)
        self.assertEqual(expected.dtype, coerced.dtype)
        assert_array_equal(expected, coerced)

        c_data = np.ascontiguousarray(f_data)
        coerced = coerce_complex(c_data)
        self.assertEqual(expected.dtype, coerced.dtype)
        assert_array_equal(expected, coerced)

    def test_coerce_file(self):
        contents = b'0123456789ABCDEF'
        buf = io.BytesIO(contents)
        coerced = coerce_file(buf)
        expected = np.atleast_2d(np.frombuffer(contents, dtype=np.uint8))
        assert_array_equal(coerced, expected)

    def test_coerce_logical(self):
        self.assertEqual(coerce_logical(True), 1)
        self.assertEqual(coerce_logical(False), 0)
        self.assertEqual(coerce_logical(np.array(True)), 1)
        self.assertEqual(coerce_logical(np.array(False)), 0)
        self.assertEqual(coerce_logical(np.bool_(True)), 1)
        self.assertEqual(coerce_logical(np.bool_(False)), 0)

        x = np.array([True, False, True, True])
        coerced = coerce_logical(x)
        self.assertEqual(coerced.dtype, np.dtype(np.uint8))
        assert_array_equal(coerced, [[1, 0, 1, 1]])

    def test_coerce_numeric(self):
        for data, typ in TEST_SCALARS:
            if typ == 'numeric':
                self.assertEqual(coerce_numeric(data), data)

        for data, typ in TEST_ARRAYS:
            if typ == 'numeric' and isinstance(data, np.ndarray):
                coerced = coerce_numeric(data)
                assert_array_equal(coerced, np.atleast_2d(data))

    def test_coerce_sparse(self):
        row = np.array([3, 4, 5, 6])
        col = np.array([0, 1, 1, 4])
        data = row + col

        obj = coo_matrix((data, (row, col)))
        coerced = coerce_sparse(obj)
        expected = np.array([row + 1, col + 1, data])
        assert_array_equal(coerced, expected)

    def test_coerce_sparse_complex(self):
        row = np.array([3, 4, 5, 6])
        col = np.array([0, 1, 1, 4])
        data = row + (1 + 1j) * col

        obj = coo_matrix((data, (row, col)))
        coerced = coerce_sparse_complex(obj)
        idx = 5 * row + col + 1
        expected = np.array([idx, data.real, data.imag])
        assert_array_equal(coerced, expected)

    def test_error_if_bad_attr(self):
        with temporary_h5file() as h5file:

            # No attr -> bad
            with self.assertRaises(BadSDAFile):
                error_if_bad_attr(h5file, 'foo', lambda value: value == 'foo')

            # Wrong attr -> bad
            h5file.attrs['foo'] = b'bar'
            with self.assertRaises(BadSDAFile):
                error_if_bad_attr(h5file, 'foo', lambda value: value == 'foo')

            # Right attr -> good
            h5file.attrs['foo'] = b'foo'
            error_if_bad_attr(h5file, 'foo', lambda value: value == 'foo')

    def test_error_if_bad_header(self):
        with temporary_h5file() as h5file:

            attrs = h5file.attrs

            # Write a good header
            for attr, value in GOOD_ATTRS.items():
                attrs[attr] = value.encode('ascii')
            error_if_not_writable(h5file)

            # Check each bad value
            for attr, value in BAD_ATTRS.items():
                attrs[attr] = value.encode('ascii')

                with self.assertRaises(BadSDAFile):
                    error_if_bad_header(h5file)

    def test_error_if_not_writable(self):
        with temporary_h5file() as h5file:
            h5file.attrs['Writable'] = b'yes'
            error_if_not_writable(h5file)

            h5file.attrs['Writable'] = b'no'
            with self.assertRaises(IOError):
                error_if_not_writable(h5file)

    def test_get_date_str(self):
        dt = datetime.datetime(2017, 8, 18, 2, 22, 11)
        date_str = get_date_str(dt)
        self.assertEqual(date_str, '18-Aug-2017 02:22:11')

        dt = datetime.datetime(2017, 8, 18, 1, 1, 1)
        date_str = get_date_str(dt)
        self.assertEqual(date_str, '18-Aug-2017 01:01:01')

        dt = datetime.datetime(2017, 8, 18, 0, 0, 0)
        date_str = get_date_str(dt)
        self.assertEqual(date_str, '18-Aug-2017')

        date_str = get_date_str()  # valid without arguments

    def test_get_empty_for_type(self):
        self.assertEqual('', get_empty_for_type('character'))
        assert_array_equal(
            np.array([], dtype=bool), get_empty_for_type('logical')
        )
        self.assertEqual(get_empty_for_type('file'), b'')
        self.assertTrue(np.isnan(get_empty_for_type('numeric')))
        self.assertEqual(get_empty_for_type('cell'), [])
        self.assertEqual(get_empty_for_type('structure'), {})

    def test_is_valid_date(self):
        self.assertTrue(is_valid_date('18-Aug-2017 02:22:11'))
        self.assertTrue(is_valid_date('18-Aug-2017'))
        self.assertFalse(is_valid_date('2017-01-01 01:23:45'))

    def test_is_valid_file_format(self):
        self.assertTrue(is_valid_file_format('SDA'))
        self.assertFalse(is_valid_file_format('sda'))
        self.assertFalse(is_valid_file_format('SDB'))

    def test_is_valid_format_version(self):
        self.assertTrue(is_valid_format_version('1.0'))
        self.assertTrue(is_valid_format_version('1.1'))
        self.assertFalse(is_valid_format_version('1.2'))
        self.assertFalse(is_valid_format_version('0.2'))
        self.assertFalse(is_valid_format_version('2.0'))

    def test_is_valid_matlab_field_label(self):
        self.assertTrue(is_valid_matlab_field_label('a0n1'))
        self.assertTrue(is_valid_matlab_field_label('a0n1999999'))
        self.assertTrue(is_valid_matlab_field_label('a0_n1'))
        self.assertTrue(is_valid_matlab_field_label('a0_N1'))
        self.assertTrue(is_valid_matlab_field_label('A0_N1'))
        self.assertFalse(is_valid_matlab_field_label(''))
        self.assertFalse(is_valid_matlab_field_label(' '))
        self.assertFalse(is_valid_matlab_field_label('1n0a'))
        self.assertFalse(is_valid_matlab_field_label('A0 N1'))
        self.assertFalse(is_valid_matlab_field_label('_A0N1'))
        self.assertFalse(is_valid_matlab_field_label(' a0n1'))
        self.assertFalse(is_valid_matlab_field_label(' A0N1'))

    def test_is_valid_writable(self):
        self.assertTrue(is_valid_writable('yes'))
        self.assertTrue(is_valid_writable('no'))
        self.assertFalse(is_valid_writable('YES'))
        self.assertFalse(is_valid_writable('NO'))
        self.assertFalse(is_valid_writable(True))
        self.assertFalse(is_valid_writable(False))

    def test_get_decoded(self):
        attrs = {'a': b'foo', 'b': b'bar', 'c': 9}
        decoded = get_decoded(attrs, 'a', 'c', 'd')
        self.assertEqual(sorted(decoded.keys()), ['a', 'c'])
        self.assertEqual(decoded['a'], 'foo')
        self.assertEqual(decoded['c'], 9)

        # Get everything
        decoded = get_decoded(attrs)
        self.assertEqual(sorted(decoded.keys()), ['a', 'b', 'c'])
        self.assertEqual(decoded['a'], 'foo')
        self.assertEqual(decoded['b'], 'bar')
        self.assertEqual(decoded['c'], 9)

    def test_set_encoded(self):
        encoded = {}
        set_encoded(encoded, a='foo', b='bar', c=9)
        self.assertEqual(sorted(encoded.keys()), ['a', 'b', 'c'])
        self.assertEqual(encoded['a'], b'foo')
        self.assertEqual(encoded['b'], b'bar')
        self.assertEqual(encoded['c'], 9)

    def test_update_header(self):
        attrs = {}
        update_header(attrs)
        self.assertEqual(len(attrs), 2)
        self.assertEqual(attrs['FormatVersion'], b'1.1')
        self.assertIsNotNone(attrs['Updated'])

    def test_write_header(self):
        attrs = {}
        write_header(attrs)
        self.assertEqual(len(attrs), 5)
        self.assertEqual(attrs['FileFormat'], b'SDA')
        self.assertEqual(attrs['FormatVersion'], b'1.1')
        self.assertEqual(attrs['Writable'], b'yes')
        self.assertEqual(attrs['Created'], attrs['Updated'])
        self.assertIsNotNone(attrs['Updated'])

    def test_infer_record_type(self):

        # scalars
        for obj, typ in TEST_SCALARS:
            msg = 'record type of {!r} != {}'.format(obj, typ)
            record_type, cast_obj, is_empty, extra = infer_record_type(obj)
            self.assertEqual(record_type, typ, msg=msg)
            self.assertEqual(cast_obj, obj)
            self.assertFalse(is_empty)
            if np.iscomplexobj(obj):
                self.assertEqual(extra, 'complex')
            else:
                self.assertIsNone(extra)

        # arrays
        for obj, typ in TEST_ARRAYS:
            msg = 'record type of {!r} != {}'.format(obj, typ)
            record_type, cast_obj, is_empty, extra = infer_record_type(obj)
            self.assertEqual(record_type, typ, msg=msg)
            self.assertFalse(is_empty)
            assert_array_equal(cast_obj, np.asarray(obj))
            if np.iscomplexobj(obj):
                self.assertEqual(extra, 'complex')
            else:
                self.assertIsNone(extra)

        # sparse
        coo = TEST_SPARSE[0]
        for obj in TEST_SPARSE:
            record_type, cast_obj, is_empty, extra = infer_record_type(obj)
            self.assertEqual(record_type, 'numeric')
            self.assertFalse(is_empty)
            self.assertEqual(extra, 'sparse')
            self.assertIsInstance(cast_obj, coo_matrix)
            assert_array_equal(cast_obj.toarray(), coo.toarray())

        # sparse+complex
        coo = TEST_SPARSE_COMPLEX[0]
        for obj in TEST_SPARSE_COMPLEX:
            record_type, cast_obj, is_empty, extra = infer_record_type(obj)
            self.assertEqual(record_type, 'numeric')
            self.assertFalse(is_empty)
            self.assertEqual(extra, 'sparse+complex')
            self.assertIsInstance(cast_obj, coo_matrix)
            assert_array_equal(cast_obj.toarray(), coo.toarray())

        # lists, tuples
        for obj in TEST_CELL:
            record_type, cast_obj, is_empty, extra = infer_record_type(obj)
            self.assertEqual(record_type, 'cell')
            self.assertFalse(is_empty)
            self.assertIsNone(extra)
            self.assertIs(cast_obj, obj)

        # dicts
        for obj in TEST_STRUCTURE:
            record_type, cast_obj, is_empty, extra = infer_record_type(obj)
            self.assertEqual(record_type, 'structure')
            self.assertFalse(is_empty)
            self.assertIsNone(extra)
            self.assertIs(cast_obj, obj)

        # file
        with temporary_file() as filename:
            with open(filename, 'w') as f:
                record_type, cast_obj, is_empty, extra = infer_record_type(f)
            self.assertEqual(record_type, 'file')
            self.assertFalse(is_empty)
            self.assertIsNone(extra)
            self.assertIs(cast_obj, f)

        # empty
        record_type, _, is_empty, _ = infer_record_type('')
        self.assertEqual(record_type, 'character')
        self.assertTrue(is_empty)

        record_type, _, is_empty, _ = infer_record_type([])
        self.assertEqual(record_type, 'cell')
        self.assertTrue(is_empty)

        record_type, _, is_empty, _ = infer_record_type({})
        self.assertEqual(record_type, 'structure')
        self.assertTrue(is_empty)

        record_type, _, is_empty, _ = infer_record_type(coo_matrix([]))
        self.assertEqual(record_type, 'numeric')
        self.assertTrue(is_empty)

        record_type, _, is_empty, _ = infer_record_type(np.nan)
        self.assertEqual(record_type, 'numeric')
        self.assertTrue(is_empty)

        data = np.array([])
        record_type, _, is_empty, _ = infer_record_type(data)
        self.assertEqual(record_type, 'numeric')
        self.assertTrue(is_empty)

        data = np.array([np.nan, np.nan])
        record_type, _, is_empty, _ = infer_record_type(data)
        self.assertEqual(record_type, 'numeric')
        self.assertTrue(is_empty)

        data = np.array([], dtype=bool)
        record_type, _, is_empty, _ = infer_record_type(data)
        self.assertEqual(record_type, 'logical')
        self.assertTrue(is_empty)

        data = np.array([], dtype=object)
        record_type, _, is_empty, _ = infer_record_type(data)
        self.assertEqual(record_type, 'cell')
        self.assertTrue(is_empty)

        # Unsupported
        for obj in TEST_UNSUPPORTED:
            msg = 'record type of {!r} is not None'.format(obj)
            record_type, cast_obj, _, extra = infer_record_type(obj)
            self.assertIsNone(record_type, msg=msg)
            self.assertIsNone(cast_obj)
