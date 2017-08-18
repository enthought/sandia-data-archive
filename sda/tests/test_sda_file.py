import unittest

from sda.exceptions import BadSDAFile
from sda.testing import BAD_ATTRS, GOOD_ATTRS, temporary_file, temporary_h5file
from sda.sda_file import SDAFile


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

    def assertHeader(self, sda_file, attrs):
        for attr, value in attrs.items():
            self.assertEqual(getattr(sda_file, attr), value)
