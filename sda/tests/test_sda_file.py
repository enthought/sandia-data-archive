import unittest

from sda.exceptions import BadSDAFile
from sda.sda_file import SDAFile
from sda.testing import BAD_ATTRS, GOOD_ATTRS, temporary_file, temporary_h5file
from sda.utils import write_header


class TestSDAFile(unittest.TestCase):

    def test_init_r(self):

        # No file
        with temporary_file() as name:
            pass  # file is deleted after this point
        with self.assertRaises(IOError):
            SDAFile(name, 'r')

        # Uninitialized -> BadSDAFile
        with temporary_h5file() as h5file:
            name = h5file.filename
            h5file.close()
            with self.assertRaises(BadSDAFile):
                SDAFile(name, 'r')

        # nonexistant -> IOError
        with self.assertRaises(IOError):
            SDAFile(name, 'r')

        # Test good attrs
        with temporary_h5file() as h5file:
            name = h5file.filename
            h5file.attrs.update(GOOD_ATTRS)
            h5file.close()
            sda_file = SDAFile(name, 'r')
            self.assertHeader(sda_file, GOOD_ATTRS)

        # Test bad attrs
        with temporary_h5file() as h5file:
            name = h5file.filename
            h5file.attrs.update(BAD_ATTRS)
            h5file.close()
            with self.assertRaises(BadSDAFile):
                SDAFile(name, 'r')

    def test_init_r_plus(self):
        # No file
        with temporary_file() as name:
            pass  # file is deleted after this point
        with self.assertRaises(IOError):
            SDAFile(name, 'r+')

        # Uninitialized -> BadSDAFile
        with temporary_h5file() as h5file:
            name = h5file.filename
            h5file.close()
            with self.assertRaises(BadSDAFile):
                SDAFile(name, 'r+')

        # nonexistant -> IOError
        with self.assertRaises(IOError):
            SDAFile(name, 'r+')

        # Test good attrs
        with temporary_h5file() as h5file:
            name = h5file.filename
            h5file.attrs.update(GOOD_ATTRS)
            h5file.close()
            sda_file = SDAFile(name, 'r+')
            self.assertHeader(sda_file, GOOD_ATTRS)

        # Test bad attrs
        with temporary_h5file() as h5file:
            name = h5file.filename
            h5file.attrs.update(BAD_ATTRS)
            h5file.close()
            with self.assertRaises(BadSDAFile):
                SDAFile(name, 'r+')

    def test_init_w(self):
        default_attrs = {}
        write_header(default_attrs)

        # No file, no problem
        with temporary_file() as name:
            pass  # file is deleted after this point
        sda_file = SDAFile(name, 'w')
        self.assertHeader(sda_file, default_attrs)

        # Yes file, uninitialized, no problem
        with temporary_h5file() as h5file:
            name = h5file.filename
            h5file.close()
            sda_file = SDAFile(name, 'w')
            self.assertHeader(sda_file, default_attrs)

            # Yes file, initialized, no problem
            sda_file = SDAFile(name, 'w')
            self.assertHeader(sda_file, default_attrs)

    def test_init_x(self, mode='x'):
        default_attrs = {}
        write_header(default_attrs)

        # No file, no problem
        with temporary_file() as name:
            pass  # file is deleted after this point
        sda_file = SDAFile(name, mode)
        self.assertHeader(sda_file, default_attrs)

        # Yes file, error
        with temporary_h5file() as h5file:
            name = h5file.filename
            h5file.close()
            with self.assertRaises(IOError):
                sda_file = SDAFile(name, mode)

    def test_init_w_minus(self):
        self.test_init_x('w-')

    def test_init_a(self):
        default_attrs = {}
        write_header(default_attrs)

        # No file, no problem
        with temporary_file() as name:
            pass  # file is deleted after this point
        sda_file = SDAFile(name, 'a')
        self.assertHeader(sda_file, default_attrs)

        # Uninitialized -> BadSDAFile
        with temporary_h5file() as h5file:
            name = h5file.filename
            h5file.close()
            with self.assertRaises(BadSDAFile):
                sda_file = SDAFile(name, 'a')

        # Initialized, no problem, header and data is preserved
        with temporary_h5file() as h5file:
            name = h5file.filename
            h5file.attrs.update(GOOD_ATTRS)
            h5file.close()
            sda_file = SDAFile(name, 'a')
            self.assertHeader(sda_file, GOOD_ATTRS)

    def test_init_default(self):
        with temporary_file() as name:
            pass  # file is deleted after this point
        sda_file = SDAFile(name)
        self.assertEqual(sda_file.mode, 'a')

    def assertHeader(self, sda_file, attrs):
        for attr, expected in attrs.items():
            actual = getattr(sda_file, attr)
            self.assertEqual(actual, expected)
