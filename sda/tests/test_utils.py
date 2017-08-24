import datetime
import string
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from sda.exceptions import BadSDAFile
from sda.testing import (
    BAD_ATTRS, GOOD_ATTRS, TEST_ARRAYS, TEST_SCALARS, TEST_UNSUPPORTED,
    temporary_h5file
)
from sda.utils import (
    coerce_character, coerce_complex, coerce_logical, coerce_numeric,
    error_if_bad_attr, error_if_bad_header, error_if_not_writable,
    extract_character, extract_complex, extract_logical, extract_numeric,
    get_date_str, get_decoded, get_empty_for_type, infer_record_type,
    is_valid_date, is_valid_file_format, is_valid_format_version,
    is_valid_writable, set_encoded, update_header, write_header
)


class TestUtils(unittest.TestCase):

    def test_coerce_character(self):
        coerced = coerce_character(string.printable)
        self.assertEqual(coerced.dtype, np.dtype(np.uint8))
        expected = np.array([ord(c) for c in string.printable], np.uint8)
        assert_array_equal(coerced, expected)

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
        assert_array_equal(coerced, [1, 0, 1, 1])

    def test_coerce_numeric(self):
        for data, typ in TEST_SCALARS:
            if typ == 'numeric':
                self.assertEqual(coerce_numeric(data), data)

        for data, typ in TEST_ARRAYS:
            if typ == 'numeric' and isinstance(data, np.ndarray):
                coerced = coerce_numeric(data)
                assert_array_equal(coerced, data)
                self.assertEqual(coerced.dtype, data.dtype)

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

    def test_extract_character(self):
        expected = string.printable
        stored = np.array([ord(c) for c in expected], np.uint8)
        extracted = extract_character(stored)
        self.assertEqual(extracted, expected)

    def test_extract_complex(self):

        expected = np.arange(6, dtype=np.complex128)
        expected.imag = -1

        stored = np.array([expected.real, expected.imag], dtype=np.float64)
        extracted = extract_complex(stored, (6,))
        self.assertEqual(expected.dtype, extracted.dtype)
        assert_array_equal(expected, extracted)

        expected = expected.astype(np.complex64)
        stored = stored.astype(np.float32)
        extracted = extract_complex(stored, (6,))
        self.assertEqual(expected.dtype, extracted.dtype)
        assert_array_equal(expected, extracted)

        expected_2d = expected.reshape((2, 3), order='F')
        extracted = extract_complex(stored, (2, 3))
        self.assertEqual(expected_2d.dtype, extracted.dtype)
        assert_array_equal(expected_2d, extracted)

    def test_extract_logical(self):
        self.assertEqual(extract_logical(1), True)
        self.assertEqual(extract_logical(0), False)

        expected = np.array([True, False, True, True], dtype=bool)
        stored = np.array([1, 0, 1, 1], dtype=np.uint8)
        extracted = extract_logical(stored)
        self.assertEqual(extracted.dtype, expected.dtype)
        assert_array_equal(extracted, expected)

    def test_extract_numeric(self):
        for data, typ in TEST_SCALARS:
            if typ == 'numeric':
                self.assertEqual(extract_numeric(data), data)

        for data, typ in TEST_ARRAYS:
            if typ == 'numeric' and isinstance(data, np.ndarray):
                extracted = extract_numeric(data)
                assert_array_equal(extracted, data)
                self.assertEqual(data.dtype, extracted.dtype)

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
        self.assertTrue(np.isnan(get_empty_for_type('numeric')))

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
            record_type, cast_obj = infer_record_type(obj)
            self.assertEqual(record_type, typ, msg=msg)
            self.assertEqual(cast_obj, obj)

        # lists, tuples, and arrays
        for obj, typ in TEST_ARRAYS:
            msg = 'record type of {!r} != {}'.format(obj, typ)
            record_type, cast_obj = infer_record_type(obj)
            self.assertEqual(record_type, typ, msg=msg)
            assert_array_equal(cast_obj, np.asarray(obj))

        # Unsupported
        for obj in TEST_UNSUPPORTED:
            msg = 'record type of {!r} is not None'.format(obj)
            record_type, cast_obj = infer_record_type(obj)
            self.assertIsNone(record_type, msg=msg)
            self.assertIsNone(cast_obj)
