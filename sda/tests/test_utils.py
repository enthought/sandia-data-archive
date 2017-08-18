import datetime
import unittest

from sda.exceptions import BadSDAFile
from sda.testing import BAD_ATTRS, GOOD_ATTRS, temporary_h5file
from sda.utils import (
    error_if_bad_attr, error_if_bad_header, error_if_not_writable,
    get_date_str, is_valid_date, is_valid_file_format, is_valid_format_version,
    is_valid_writable, write_header
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
            write_header(h5file)

            attrs = h5file.attrs
            self.assertEqual(attrs['FileFormat'], 'SDA')
            self.assertEqual(attrs['FormatVersion'], '1.0')
            self.assertEqual(attrs['Writable'], 'yes')
            self.assertEqual(attrs['Created'], attrs['Modified'])
