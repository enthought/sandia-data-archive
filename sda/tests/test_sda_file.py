import os
import unittest

import numpy as np
from numpy.testing import assert_equal

from sda.exceptions import BadSDAFile
from sda.sda_file import SDAFile
from sda.testing import (
    BAD_ATTRS, GOOD_ATTRS, TEST_ARRAYS, TEST_SCALARS, TEST_UNSUPPORTED,
    temporary_file, temporary_h5file
)
from sda.utils import write_header

# TODO - test actual read and write


class TestSDAFileInit(unittest.TestCase):

    def test_mode_r(self):
        self.assertInitNew('r', exc=IOError)
        self.assertInitExisting('r', {}, BadSDAFile)
        self.assertInitExisting('r', BAD_ATTRS, BadSDAFile)
        self.assertInitExisting('r', GOOD_ATTRS)

    def test_mode_r_plus(self):
        self.assertInitNew('r+', exc=IOError)
        self.assertInitExisting('r+', exc=BadSDAFile)
        self.assertInitExisting('r+', exc=BadSDAFile)
        self.assertInitExisting('r+', BAD_ATTRS, BadSDAFile)
        self.assertInitExisting('r+', GOOD_ATTRS)

    def test_mode_w(self):
        self.assertInitNew('w')
        self.assertInitExisting('w')

    def test_mode_x(self):
        self.assertInitNew('x')
        self.assertInitExisting('x', exc=IOError)

    def test_mode_w_minus(self):
        self.assertInitNew('w-')
        self.assertInitExisting('w-', exc=IOError)

    def test_mode_a(self):
        self.assertInitNew('a')
        self.assertInitExisting('a', GOOD_ATTRS)
        self.assertInitExisting('a', BAD_ATTRS, BadSDAFile)
        self.assertInitExisting('a', {}, BadSDAFile)

    def test_mode_default(self):
        with temporary_h5file() as h5file:
            name = h5file.filename
            h5file.attrs.update(GOOD_ATTRS)
            h5file.close()
            sda_file = SDAFile(name)
            self.assertEqual(sda_file.mode, 'a')

    def test_pass_kw(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w', driver='core')
            with sda_file._h5file('r') as h5file:
                self.assertEqual(h5file.driver, 'core')

    def assertAttrs(self, sda_file, attrs={}):
        """ Assert sda_file attributes are equal to passed values.

        if ``attrs`` is empty, check that ``attrs`` take on the default values.

        """
        if attrs == {}:  # treat as if new
            self.assertEqual(sda_file.Created, sda_file.Updated)
            attrs = {}
            write_header(attrs)
            del attrs['Created']
            del attrs['Updated']

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
            if attrs is not None:
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
        with temporary_file() as file_path:
            os.remove(file_path)
            if exc is not None:
                with self.assertRaises(exc):
                    SDAFile(file_path, mode)
            else:
                sda_file = SDAFile(file_path, mode)
                self.assertAttrs(sda_file)


class TestSDAFileInsert(unittest.TestCase):

    def test_read_only(self):
        with temporary_h5file() as h5file:
            name = h5file.filename
            h5file.attrs.update(GOOD_ATTRS)
            h5file.close()
            sda_file = SDAFile(name, 'r')

            with self.assertRaises(IOError):
                sda_file.insert('test', [1, 2, 3])

    def test_no_write(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            sda_file.Writable = 'no'
            with self.assertRaises(IOError):
                sda_file.insert('test', [1, 2, 3])

    def test_invalid_deflate(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            with self.assertRaises(ValueError):
                sda_file.insert('test', [1, 2, 3], deflate=-1)

            with self.assertRaises(ValueError):
                sda_file.insert('test', [1, 2, 3], deflate=10)

            with self.assertRaises(ValueError):
                sda_file.insert('test', [1, 2, 3], deflate=None)

    def test_invalid_label(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            with self.assertRaises(ValueError):
                sda_file.insert('test/', [1, 2, 3])

            with self.assertRaises(ValueError):
                sda_file.insert('test\\', [1, 2, 3])

    def test_label_exists(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            sda_file.insert('test', [1, 2, 3])
            with self.assertRaises(ValueError):
                sda_file.insert('test', [1, 2, 3])

    def test_timestamp_update(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            with sda_file._h5file('a') as h5file:
                h5file.attrs['Updated'] = 'Unmodified'

            sda_file.insert('test', [0, 1, 2])
            self.assertNotEqual(sda_file.Updated, 'Unmodified')

    def test_character_scalar(self):
        values = (obj for (obj, typ) in TEST_SCALARS if typ == 'character')

        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            for i, obj in enumerate(values):
                label = 'test' + str(i)
                deflate = i % 10
                sda_file.insert(label, obj, label, deflate)
                expected = np.frombuffer(obj.encode('ascii'), 'S1').view('u1')
                self.assertRecord(
                    sda_file, 'character', label, deflate, 'no', expected
                )

            label = 'test_empty'
            deflate = 0
            sda_file.insert(label, '', label, deflate)
            self.assertRecord(
                sda_file, 'character', label, deflate, 'yes', None
            )

    def test_logical_array(self):
        values = (obj for (obj, typ) in TEST_ARRAYS if typ == 'logical')
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')

            for i, obj in enumerate(values):
                label = 'test' + str(i)
                deflate = i % 10
                sda_file.insert(label, obj, label, deflate)
                expected = np.asarray(obj).astype(np.uint8).clip(0, 1)
                self.assertRecord(
                    sda_file, 'logical', label, deflate, 'no', expected
                )

    def test_logical_scalar(self):
        values = (obj for (obj, typ) in TEST_SCALARS if typ == 'logical')
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')

            for i, obj in enumerate(values):
                label = 'test' + str(i)
                deflate = i % 10
                sda_file.insert(label, obj, label, deflate)
                expected = 1 if obj else 0
                self.assertRecord(
                    sda_file, 'logical', label, deflate, 'no', expected
                )

    def test_numeric_array(self):
        values = (obj for (obj, typ) in TEST_ARRAYS if typ == 'numeric')
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')

            for i, obj in enumerate(values):
                label = 'test' + str(i)
                deflate = i % 10
                sda_file.insert(label, obj, label, deflate)
                expected = np.asarray(obj)
                self.assertRecord(
                    sda_file, 'numeric', label, deflate, 'no', expected
                )

            label = 'test_empty'
            deflate = 0
            sda_file.insert(label, [], label, deflate)
            self.assertRecord(sda_file, 'numeric', label, deflate, 'yes', None)

    def test_numeric_scalar(self):
        values = (obj for (obj, typ) in TEST_SCALARS if typ == 'numeric')
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')

            for i, obj in enumerate(values):
                label = 'test' + str(i)
                deflate = i % 10
                sda_file.insert(label, obj, label, deflate)
                self.assertRecord(
                    sda_file, 'numeric', label, deflate, 'no', obj
                )

            label = 'test_nan'
            deflate = 0
            sda_file.insert(label, np.nan, label, deflate)
            self.assertRecord(sda_file, 'numeric', label, deflate, 'yes', None)

    def test_unsupported(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            with sda_file._h5file('a') as h5file:
                h5file.attrs['Updated'] = 'Unmodified'

            for i, obj in enumerate(TEST_UNSUPPORTED):
                label = 'test' + str(i)
                with self.assertRaises(ValueError):
                    sda_file.insert(label, obj, label, 0)

            # Make sure the 'Updated' attr does not get updated
            self.assertEqual(sda_file.Updated, 'Unmodified')

    def assertRecord(self, sda_file, record_type, label, deflate, empty,
                     expected):
        with sda_file._h5file('r') as h5file:
            g = h5file[label]
            self.assertEqual(g.attrs['RecordType'], record_type)
            self.assertEqual(g.attrs['Deflate'], deflate)
            self.assertEqual(g.attrs['Description'], label)
            self.assertEqual(g.attrs['Empty'], empty)

            ds = g[label]
            self.assertEqual(ds.attrs['RecordType'], record_type)
            self.assertEqual(ds.attrs['Empty'], empty)
            if empty == 'no':
                assert_equal(ds[()], expected)


class TestSDAFileDescribe(unittest.TestCase):

    def test_read_only(self):
        with temporary_h5file() as h5file:
            name = h5file.filename
            h5file.attrs.update(GOOD_ATTRS)
            h5file.close()
            sda_file = SDAFile(name, 'r')

            with self.assertRaises(IOError):
                sda_file.describe('test', 'a test')

    def test_no_write(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            sda_file.Writable = 'no'
            with self.assertRaises(IOError):
                sda_file.describe('test', 'a test')

    def test_invalid_label(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            with self.assertRaises(ValueError):
                sda_file.describe('test/', 'a test')

            with self.assertRaises(ValueError):
                sda_file.describe('test\\', 'a test')

    def test_missing_label(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            with self.assertRaises(ValueError):
                sda_file.describe('test', 'a test')

    def test_happy_path(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            with sda_file._h5file('a') as h5file:
                h5file.attrs['Updated'] = 'Unmodifed'

            sda_file.insert('test', [1, 2, 3])
            sda_file.describe('test', 'second')
            with sda_file._h5file('r') as h5file:
                self.assertEqual(h5file['test'].attrs['Description'], 'second')

            # Make sure the 'Updated' attr gets updated
            self.assertNotEqual(sda_file.Updated, 'Unmodified')
