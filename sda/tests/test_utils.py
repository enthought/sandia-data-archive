import datetime
import unittest

import numpy as np

from sda.exceptions import BadSDAFile
from sda.testing import BAD_ATTRS, GOOD_ATTRS, temporary_h5file
from sda.utils import (
    error_if_bad_attr, error_if_bad_header, error_if_not_writable,
    get_date_str, infer_record_type, is_valid_date, is_valid_file_format,
    is_valid_format_version, is_valid_writable, write_header
)


class TestUtils(unittest.TestCase):

    def test_error_if_bad_attr(self):
        with temporary_h5file() as h5file:

            # No attr -> bad
            with self.assertRaises(BadSDAFile):
                error_if_bad_attr(h5file, 'foo', lambda value: value == 'foo')

            # Wrong attr -> bad
            h5file.attrs['foo'] = 'bar'
            with self.assertRaises(BadSDAFile):
                error_if_bad_attr(h5file, 'foo', lambda value: value == 'foo')

            # Right attr -> good
            h5file.attrs['foo'] = 'foo'
            error_if_bad_attr(h5file, 'foo', lambda value: value == 'foo')

    def test_error_if_bad_header(self):
        with temporary_h5file() as h5file:

            attrs = h5file.attrs

            # Write a good header
            attrs.update(GOOD_ATTRS)
            error_if_not_writable(h5file)

            # Check each bad value
            for attr, value in BAD_ATTRS.items():
                attrs.update(GOOD_ATTRS)
                attrs[attr] = value

                with self.assertRaises(BadSDAFile):
                    error_if_bad_header(h5file)

    def test_error_if_not_writable(self):
        with temporary_h5file() as h5file:
            h5file.attrs['Writable'] = 'yes'
            error_if_not_writable(h5file)

            h5file.attrs['Writable'] = 'no'
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
        self.assertTrue(is_valid_format_version('1.2'))
        self.assertFalse(is_valid_format_version('0.2'))
        self.assertFalse(is_valid_format_version('2.0'))

    def test_is_valid_writable(self):
        self.assertTrue(is_valid_writable('yes'))
        self.assertTrue(is_valid_writable('no'))
        self.assertFalse(is_valid_writable('YES'))
        self.assertFalse(is_valid_writable('NO'))
        self.assertFalse(is_valid_writable(True))
        self.assertFalse(is_valid_writable(False))

    def test_write_header(self):

        with temporary_h5file() as h5file:
            write_header(h5file.attrs)

            attrs = h5file.attrs
            self.assertEqual(attrs['FileFormat'], 'SDA')
            self.assertEqual(attrs['FormatVersion'], '1.0')
            self.assertEqual(attrs['Writable'], 'yes')
            self.assertEqual(attrs['Created'], attrs['Modified'])

    def test_infer_record_type(self):

        # Simulate 'long' type to simplify code below
        try:
            long(0)
        except NameError:
            long = int

        float_val = 3.14159
        int_val = 3
        bool_val = True
        complex_val = 1.23 + 4.56j
        str_val = 'foo'
        unicode_val = u'foo'

        # scalars
        scalars = [
            (float_val, 'numeric'),
            (np.float32(float_val), 'numeric'),
            (np.float64(float_val), 'numeric'),
            (int_val, 'numeric'),
            (long(int_val), 'numeric'),
            (np.int8(int_val), 'numeric'),
            (np.int16(int_val), 'numeric'),
            (np.int32(int_val), 'numeric'),
            (np.int64(int_val), 'numeric'),
            (np.uint8(int_val), 'numeric'),
            (np.uint16(int_val), 'numeric'),
            (np.uint32(int_val), 'numeric'),
            (np.uint64(int_val), 'numeric'),
            (complex_val, 'numeric'),
            (np.complex64(complex_val), 'numeric'),
            (np.complex128(complex_val), 'numeric'),
            (bool_val, 'logical'),
            (np.bool_(bool_val), 'logical'),
            (str_val, 'character'),
            (np.str_(str_val), 'character'),
            (np.unicode_(unicode_val), 'character'),
        ]

        # lists, tuples, and arrays
        arrays = []
        for val, typ in scalars:
            arr = [val] * 4
            arrays.append((arr, typ))
            arrays.append((tuple(arr), typ))
            arrays.append((np.array(arr), typ))
            arrays.append((np.array(arr).reshape(2, 2), typ))

        # Consistency with numpy upcasting
        arrays.append(([3, 'hello'], 'character'))

        # array scalars
        scalars += [(np.array(val), typ) for val, typ in scalars]

        for val, typ in scalars + arrays:
            msg = 'infer_record_type({}) != {}'.format(val, typ)
            self.assertEqual(infer_record_type(val), typ, msg=msg)

        # Unsupported
        unsupported = [
            np.array([3, 'hello'], dtype=object),
            lambda x: x**2,
            {0: 0},
            {0},
            None,
        ]

        for val in unsupported:
            msg = 'infer_record_type({}) is not None'.format(val)
            self.assertIsNone(infer_record_type(val), msg=msg)
