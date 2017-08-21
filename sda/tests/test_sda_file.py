import unittest

from sda.exceptions import BadSDAFile
from sda.sda_file import SDAFile
from sda.testing import BAD_ATTRS, GOOD_ATTRS, temporary_file, temporary_h5file
from sda.utils import write_header

# TODO - test actual read and write


class TestSDAFile(unittest.TestCase):

    def test_init_r(self):
        self.assertInitNew('r', exc=IOError)
        self.assertInitExisting('r', {}, BadSDAFile)
        self.assertInitExisting('r', BAD_ATTRS, BadSDAFile)
        self.assertInitExisting('r', GOOD_ATTRS)

    def test_init_r_plus(self):
        self.assertInitNew('r+', exc=IOError)
        self.assertInitExisting('r+', exc=BadSDAFile)
        self.assertInitExisting('r+', {}, BadSDAFile)
        self.assertInitExisting('r+', BAD_ATTRS, BadSDAFile)
        self.assertInitExisting('r+', GOOD_ATTRS)

    def test_init_w(self):
        attrs = {}
        write_header(attrs)
        self.assertInitNew('w', attrs)
        self.assertInitExisting('w', attrs)
        self.assertInitExisting('w', {})

    def test_init_x(self):
        attrs = {}
        write_header(attrs)

        self.assertInitNew('x', attrs)
        self.assertInitExisting('x', exc=IOError)

    def test_init_w_minus(self):
        attrs = {}
        write_header(attrs)

        self.assertInitNew('w-', attrs)
        self.assertInitExisting('w-', exc=IOError)

    def test_init_a(self):
        attrs = {}
        write_header(attrs)

        self.assertInitNew('a', attrs)
        self.assertInitExisting('a', GOOD_ATTRS)
        self.assertInitExisting('a', BAD_ATTRS, BadSDAFile)
        self.assertInitExisting('a', {}, BadSDAFile)

    def test_init_default(self):
        with temporary_file() as name:
            pass  # file is deleted after this point
        sda_file = SDAFile(name)
        self.assertEqual(sda_file.mode, 'a')

    def assertAttrs(self, sda_file, attrs):
        for attr, expected in attrs.items():
            actual = getattr(sda_file, attr)
            self.assertEqual(actual, expected)

    def assertInitExisting(self, mode, attrs={}, exc=None):
        """ Assert attributes or error when init with existing file.

        Passed ``attrs`` are used when creating the existing file. When ``exc``
        is None, this also tests that the ``attrs`` are preserved.

        """
        with temporary_h5file() as h5file:
            name = h5file.filename
            h5file.attrs.update(attrs)
            h5file.close()

            if exc is not None:
                with self.assertRaises(exc):
                    SDAFile(name, mode)
            else:
                sda_file = SDAFile(name, mode)
                self.assertAttrs(sda_file, attrs)

    def assertInitNew(self, mode, attrs={}, exc=None):
        """ Assert attributes or error when init with non-existing file. """
        with temporary_file() as name:
            pass  # file is deleted after this point

        if exc is not None:
            with self.assertRaises(exc):
                SDAFile(name, mode)
        else:
            sda_file = SDAFile(name, mode)
            self.assertAttrs(sda_file, attrs)
