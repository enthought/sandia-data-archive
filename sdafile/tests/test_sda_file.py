import os
import random
import shutil
import string
import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from scipy.sparse import coo_matrix

from sdafile.exceptions import BadSDAFile
from sdafile.sda_file import SDAFile
from sdafile.testing import (
    BAD_ATTRS, GOOD_ATTRS, TEST_ARRAYS, TEST_CELL, TEST_SCALARS, TEST_SPARSE,
    TEST_SPARSE_COMPLEX, TEST_STRUCTURE, TEST_UNSUPPORTED, data_path,
    temporary_file, temporary_h5file
)
from sdafile.utils import (
    coerce_character, coerce_complex, coerce_logical, coerce_numeric,
    coerce_primitive, coerce_sparse, coerce_sparse_complex, get_decoded,
    infer_record_type, is_primitive, set_encoded, write_header
)


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
            set_encoded(h5file.attrs, **GOOD_ATTRS)
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
            attrs = get_decoded(attrs)

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
            if attrs is not None and len(attrs) > 0:
                set_encoded(h5file.attrs, **attrs)
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


class TestSDAFileProperties(unittest.TestCase):

    def test_file_properties(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            self.assertEqual(sda_file.mode, 'w')
            self.assertEqual(sda_file.name, file_path)

    def test_set_writable(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            self.assertEqual(sda_file.Writable, 'yes')
            sda_file.Writable = 'no'
            self.assertEqual(sda_file.Writable, 'no')

            with self.assertRaises(ValueError):
                sda_file.Writable = True

            with self.assertRaises(ValueError):
                sda_file.Writable = False

            sda_file = SDAFile(file_path, 'r')

            with self.assertRaises(ValueError):
                sda_file.Writable = 'yes'


class TestSDAFileInsert(unittest.TestCase):

    def test_read_only(self):
        with temporary_h5file() as h5file:
            name = h5file.filename
            set_encoded(h5file.attrs, **GOOD_ATTRS)
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
                set_encoded(h5file.attrs, Updated='Unmodified')

            sda_file.insert('test', [0, 1, 2])
            self.assertNotEqual(sda_file.Updated, 'Unmodified')

    def test_invalid_structure_key(self):
        record = [0, 1, 2, {' bad': np.arange(4)}]
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')

            with self.assertRaises(ValueError):
                sda_file.insert('something_bad', record)

            self.assertEqual(sda_file.labels(), [])

    def test_character(self):
        values = (obj for (obj, typ) in TEST_SCALARS if typ == 'character')

        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            for i, obj in enumerate(values):
                label = 'test' + str(i)
                deflate = i % 10
                sda_file.insert(label, obj, label, deflate)
                expected = coerce_character(obj)
                self.assertPrimitiveRecord(
                    sda_file, 'character', label, deflate, 'no', expected
                )

            label = 'test_empty'
            deflate = 0
            sda_file.insert(label, '', label, deflate)
            self.assertPrimitiveRecord(
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
                expected = coerce_logical(np.asarray(obj))
                self.assertPrimitiveRecord(
                    sda_file, 'logical', label, deflate, 'no', expected
                )

            arr = np.array([], dtype=bool)
            sda_file.insert('empty', arr, 'empty')
            self.assertPrimitiveRecord(
                sda_file, 'logical', 'empty', 0, 'yes', None
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
                self.assertPrimitiveRecord(
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
                is_complex = np.iscomplexobj(obj)
                if is_complex:
                    expected = coerce_complex(np.asarray(obj))
                    shape = np.atleast_2d(obj).shape
                    self.assertPrimitiveRecord(
                        sda_file, 'numeric', label, deflate, 'no', expected,
                        Complex='yes' if is_complex else 'no',
                        ArraySize=shape
                    )
                else:
                    expected = coerce_numeric(np.asarray(obj))
                    self.assertPrimitiveRecord(
                        sda_file, 'numeric', label, deflate, 'no', expected,
                        Complex='yes' if is_complex else 'no',
                    )

            label = 'test_empty'
            deflate = 0
            sda_file.insert(label, np.array([]), label, deflate)
            self.assertPrimitiveRecord(
                sda_file, 'numeric', label, deflate, 'yes', None,
                Complex='no'
            )

    def test_numeric_scalar(self):
        values = (obj for (obj, typ) in TEST_SCALARS if typ == 'numeric')
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')

            for i, obj in enumerate(values):
                label = 'test' + str(i)
                deflate = i % 10
                sda_file.insert(label, obj, label, deflate)
                is_complex = np.iscomplexobj(obj)
                if is_complex:
                    expected = coerce_complex(obj)
                else:
                    expected = coerce_numeric(obj)
                self.assertPrimitiveRecord(
                    sda_file, 'numeric', label, deflate, 'no', expected,
                    Complex='yes' if is_complex else 'no',
                    Sparse='no'
                )

            label = 'test_nan'
            deflate = 0
            sda_file.insert(label, np.nan, label, deflate)
            self.assertPrimitiveRecord(
                sda_file, 'numeric', label, deflate, 'yes', None
            )

    def test_sparse(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')

            for i, obj in enumerate(TEST_SPARSE):
                label = 'test' + str(i)
                deflate = i % 10
                sda_file.insert(label, obj, label, deflate)
                expected = coerce_sparse(obj.tocoo())
                self.assertPrimitiveRecord(
                    sda_file, 'numeric', label, deflate, 'no', expected,
                    Complex='no', Sparse='yes',
                )

    def test_sparse_complex(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')

            for i, obj in enumerate(TEST_SPARSE_COMPLEX):
                label = 'test' + str(i)
                deflate = i % 10
                sda_file.insert(label, obj, label, deflate)
                expected = coerce_sparse_complex(obj.tocoo())
                self.assertPrimitiveRecord(
                    sda_file, 'numeric', label, deflate, 'no', expected,
                    Complex='yes', Sparse='yes',
                )

    def test_cell(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')

            for i, objs in enumerate(TEST_CELL):
                label = 'test' + str(i)
                deflate = i % 10
                sda_file.insert(label, objs, label, deflate)
                if isinstance(objs, np.ndarray):
                    record_size = np.atleast_2d(objs).shape
                else:
                    record_size = (1, len(objs))

                self.assertCompositeRecord(
                    sda_file,
                    label,
                    objs,
                    Deflate=deflate,
                    RecordType='cell',
                    Empty='no',
                    RecordSize=record_size,
                )

    def test_structure(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')

            for i, obj in enumerate(TEST_STRUCTURE):
                label = 'test' + str(i)
                deflate = i % 10
                sda_file.insert(label, obj, label, deflate)
                self.assertCompositeRecord(
                    sda_file,
                    label,
                    obj,
                    Deflate=deflate,
                    RecordType='structure',
                    Empty='no',
                    FieldNames=' '.join(sorted(obj.keys())),
                )

    def test_unsupported(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            with sda_file._h5file('a') as h5file:
                set_encoded(h5file.attrs, Updated='Unmodified')

            for i, obj in enumerate(TEST_UNSUPPORTED):
                label = 'test' + str(i)
                with self.assertRaises(ValueError):
                    sda_file.insert(label, obj, label, 0)

            # Make sure the 'Updated' attr does not change
            self.assertEqual(sda_file.Updated, 'Unmodified')

    def assertAttrs(self, dict_like, **expected_attrs):
        attrs = get_decoded(dict_like)
        for key, value in expected_attrs.items():
            assert_equal(attrs[key], value)
        return attrs

    def assertCompositeRecord(self, sda_file, label, expected, **group_attrs):
        with sda_file._h5file('r') as h5file:
            group = h5file[label]
            self.assertCompositeGroup(group, expected, **group_attrs)

    def assertPrimitiveRecord(self, sda_file, record_type, label, deflate,
                              empty, expected, **data_attrs):
        with sda_file._h5file('r') as h5file:
            group = h5file[label]
            self.assertAttrs(
                group.attrs,
                RecordType=record_type,
                Deflate=deflate,
                Description=label,
                Empty=empty,
            )

            data_set = group[label]
            self.assertDataSet(
                data_set,
                expected,
                Empty=empty,
                RecordType=record_type,
                **data_attrs
            )

    def assertDataSet(self, data_set, expected, **data_attrs):
        attrs = self.assertAttrs(data_set.attrs, **data_attrs)
        if attrs['Empty'] == 'no':
            assert_equal(data_set[()], expected)

    def assertCompositeGroup(self, group, expected, **group_attrs):
        """ Check a composite group """
        attrs = self.assertAttrs(group.attrs, **group_attrs)

        # Check the data
        record_type = attrs['RecordType']
        if record_type == 'cell':
            expected = np.asarray(expected, dtype='object').ravel(order='F')
            labels = [
                'element {}'.format(i) for i in range(1, len(expected) + 1)
            ]
        else:
            labels, expected = zip(*expected.items())

        for label, obj in zip(labels, expected):
            sub_record_type, data, extra = infer_record_type(obj)
            if is_primitive(sub_record_type):
                data, _ = coerce_primitive(sub_record_type, data, extra)
                data_set = group[label]
                self.assertDataSet(data_set, data)
            elif sub_record_type == 'cell':
                if isinstance(obj, np.ndarray):
                    record_size = np.atleast_2d(obj).shape
                else:
                    record_size = (1, len(obj))
                sub_group = group[label]
                self.assertCompositeGroup(
                    sub_group,
                    obj,
                    RecordType=sub_record_type,
                    RecordSize=record_size,
                )
            elif sub_record_type == 'structure':
                sub_group = group[label]
                field_names = ' '.join(sorted(obj.keys()))
                self.assertCompositeGroup(
                    sub_group,
                    obj,
                    RecordType=sub_record_type,
                    FieldNames=field_names,
                )


class TestSDAFileExtract(unittest.TestCase):

    def test_invalid_label(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            with self.assertRaises(ValueError):
                sda_file.extract('test/')

            with self.assertRaises(ValueError):
                sda_file.extract('test\\')

    def test_label_not_exists(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            with self.assertRaises(ValueError):
                sda_file.extract('test')

    def test_no_timestamp_update(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            sda_file.insert('test', [0, 1, 2])
            with sda_file._h5file('a') as h5file:
                set_encoded(h5file.attrs, Updated='Unmodified')

            sda_file.extract('test')
            self.assertEqual(sda_file.Updated, 'Unmodified')

    def test_character_scalar(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            expected = string.printable
            sda_file.insert('test', expected)
            extracted = sda_file.extract('test')
            self.assertEqual(extracted, expected)

            expected = ''
            sda_file.insert('test2', '')
            extracted = sda_file.extract('test2')
            self.assertEqual(extracted, expected)

    def test_logical_scalar(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            sda_file.insert('true', True)
            sda_file.insert('false', False)
            self.assertTrue(sda_file.extract('true'))
            self.assertFalse(sda_file.extract('false'))

    def test_logical_array(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            expected = np.array([1, 0, 1, 1], dtype=bool).reshape(2, 2)
            sda_file.insert('test', expected)
            extracted = sda_file.extract('test')
            assert_array_equal(expected, extracted)
            self.assertEqual(extracted.dtype, expected.dtype)

    def test_numeric_scalar(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            sda_file.insert('test', 3.14159)
            sda_file.insert('empty', np.nan)
            self.assertEqual(sda_file.extract('test'), 3.14159)
            self.assertTrue(np.isnan(sda_file.extract('empty')))

    def test_numeric_array(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            for i, (data, typ) in enumerate(TEST_ARRAYS):
                if typ == 'numeric' and isinstance(data, np.ndarray):
                    label = 'test' + str(i)
                    sda_file.insert(label, data)
                    extracted = sda_file.extract(label)
                    self.assertEqual(data.dtype, extracted.dtype)
                    assert_array_equal(extracted, data)

            sda_file.insert('empty', np.array([], dtype=float))
            self.assertTrue(np.isnan(sda_file.extract('empty')))

    def test_sparse(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            for i, data in enumerate(TEST_SPARSE):
                label = 'test' + str(i)
                sda_file.insert(label, data)
                extracted = sda_file.extract(label)
                self.assertIsInstance(extracted, coo_matrix)
                self.assertEqual(extracted.dtype, data.dtype)
                assert_array_equal(extracted.toarray(), data.toarray())

    def test_sparse_complex(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            for i, data in enumerate(TEST_SPARSE_COMPLEX):
                label = 'test' + str(i)
                sda_file.insert(label, data)
                extracted = sda_file.extract(label)
                self.assertIsInstance(extracted, coo_matrix)
                self.assertEqual(extracted.dtype, data.dtype)
                assert_array_equal(extracted.toarray(), data.toarray())

    def test_cell(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            for i, data in enumerate(TEST_CELL):
                label = 'test' + str(i)
                sda_file.insert(label, data)
                extracted = sda_file.extract(label)
                data = np.asarray(data, dtype=object)
                extracted = np.asarray(data, dtype=object)
                assert_equal(extracted, data)

    def test_structure(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            for i, data in enumerate(TEST_STRUCTURE):
                label = 'test' + str(i)
                sda_file.insert(label, data)
                extracted = sda_file.extract(label)
                self.assertEqual(sorted(data.keys()), sorted(extracted.keys()))
                for key in data:
                    assert_equal(extracted[key], data[key])


class TestSDAFileDescribe(unittest.TestCase):

    def test_read_only(self):
        with temporary_h5file() as h5file:
            name = h5file.filename
            set_encoded(h5file.attrs, **GOOD_ATTRS)
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
                set_encoded(h5file.attrs, Updated='Unmodified')

            sda_file.insert('test', [1, 2, 3])
            sda_file.describe('test', 'second')
            with sda_file._h5file('r') as h5file:
                attrs = get_decoded(h5file['test'].attrs, 'Description')
                self.assertEqual(attrs['Description'], 'second')

            # Make sure the 'Updated' attr gets updated
            self.assertNotEqual(sda_file.Updated, 'Unmodified')


class TestSDAFileMisc(unittest.TestCase):

    def test_labels(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            sda_file.insert('l0', [0])
            sda_file.insert('l1', [1])
            self.assertEqual(sorted(sda_file.labels()), ['l0', 'l1'])

    def test_remove(self):
        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')

            labels = []

            ALL = (
                [obj for obj, _ in TEST_ARRAYS + TEST_SCALARS] + TEST_CELL +
                TEST_SPARSE + TEST_SPARSE_COMPLEX + TEST_STRUCTURE
            )

            for i, obj in enumerate(ALL):
                label = 'test' + str(i)
                labels.append(label)
                sda_file.insert(label, obj)

            with self.assertRaises(ValueError):
                sda_file.remove()

            with self.assertRaises(ValueError):
                sda_file.remove('not a label')

            random.shuffle(labels)
            removed = labels[::2]
            kept = labels[1::2]

            with sda_file._h5file('a') as h5file:
                set_encoded(h5file.attrs, Updated='Unmodified')

            sda_file.remove(*removed)
            self.assertEqual(sorted(sda_file.labels()), sorted(kept))

            # Make sure metadata is preserved and data can be extracted
            with sda_file._h5file('r') as h5file:
                for label in kept:
                    attrs = h5file[label].attrs
                    self.assertIn('Deflate', attrs)
                    self.assertIn('Description', attrs)
                    self.assertIn('RecordType', attrs)
                    self.assertIn('Empty', attrs)
                    sda_file.extract(label)

            sda_file.remove(*kept)
            self.assertEqual(sda_file.labels(), [])

            self.assertEqual(sda_file.FormatVersion, '1.1')
            self.assertNotEqual(sda_file.Updated, 'Unmodified')

    def test_probe(self):

        cols = [
            'RecordType', 'Description', 'Empty', 'Deflate', 'Complex',
            'ArraySize', 'Sparse', 'RecordSize', 'Class', 'FieldNames',
            'Command',
        ]

        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')

            labels = []
            for i, (obj, _) in enumerate(TEST_ARRAYS[:4]):
                label = 'array' + str(i)
                labels.append(label)
                sda_file.insert(label, obj, label, i)

            for i, (obj, _) in enumerate(TEST_SCALARS[:2]):
                label = 'scalar' + str(i)
                labels.append(label)
                sda_file.insert(label, obj, label, i)

            state = sda_file.probe()
            state.sort_index()
            self.assertEqual(len(state), 6)
            assert_array_equal(state.columns, cols)
            assert_array_equal(state.index, labels)
            assert_array_equal(state['Description'], labels)
            assert_array_equal(state['Deflate'], [0, 1, 2, 3, 0, 1])

            state = sda_file.probe('array.*')
            state.sort_index()
            self.assertEqual(len(state), 4)
            assert_array_equal(state.columns, cols)
            assert_array_equal(state.index, labels[:4])
            assert_array_equal(state['Description'], labels[:4])
            assert_array_equal(state['Deflate'], [0, 1, 2, 3])

            state = sda_file.probe('scalar.*')
            state.sort_index()
            self.assertEqual(len(state), 2)
            assert_array_equal(state.columns, cols)
            assert_array_equal(state.index, labels[4:])
            assert_array_equal(state['Description'], labels[4:])
            assert_array_equal(state['Deflate'], [0, 1])


class TestSDAFileReplaceUpdate(unittest.TestCase):

    def test_replace(self):

        with temporary_file() as file_path:
            sda_file = SDAFile(file_path, 'w')
            sda_file.insert('test', TEST_ARRAYS[0][0], 'test_description', 1)

            replacements = [
                data for (data, _) in TEST_ARRAYS[1:] + TEST_SCALARS
            ]
            random.shuffle(replacements)
            replacements = replacements[:10]

            with sda_file._h5file('a') as h5file:
                set_encoded(h5file.attrs, Updated='Unmodified')

            for new_data in replacements:
                sda_file.replace('test', new_data)
                assert_equal(sda_file.extract('test'), new_data)
                with sda_file._h5file('r') as h5file:
                    attrs = get_decoded(
                        h5file['test'].attrs, 'Deflate', 'Description'
                    )
                self.assertEqual(attrs['Description'], 'test_description')
                self.assertEqual(attrs['Deflate'], 1)

            self.assertNotEqual(sda_file.Updated, 'Unmodified')

    def test_replace_non_object(self):

        reference_path = data_path('SDAreference.sda')
        with temporary_file() as file_path:
            # Copy the reference, which as an object in it.
            shutil.copy(reference_path, file_path)
            sda_file = SDAFile(file_path, 'a')
            label = 'example A'
            data = sda_file.extract('example I')
            with self.assertRaises(ValueError):
                sda_file.update_object(label, data)

    def test_update_object_with_equivalent_record(self):

        reference_path = data_path('SDAreference.sda')
        with temporary_file() as file_path:
            # Copy the reference, which as an object in it.
            shutil.copy(reference_path, file_path)
            sda_file = SDAFile(file_path, 'a')
            with sda_file._h5file('a') as h5file:
                set_encoded(h5file.attrs, Updated='Unmodified')

            label = 'example I'

            # Replace some stuff with the same type
            data = sda_file.extract(label)
            data['Parameter'] = np.arange(5)
            sda_file.update_object(label, data)

            extracted = sda_file.extract(label)

            with sda_file._h5file('r') as h5file:
                attrs = get_decoded(h5file['example I'].attrs)

            self.assertNotEqual(sda_file.Updated, 'Unmodified')

        # Validate equality
        self.assertEqual(attrs['RecordType'], 'object')
        self.assertEqual(attrs['Class'], 'ExampleObject')
        self.assertIsInstance(extracted, dict)
        self.assertEqual(len(extracted), 1)
        assert_equal(extracted['Parameter'], data['Parameter'])

    def test_update_object_with_inequivalent_record(self):

        reference_path = data_path('SDAreference.sda')
        with temporary_file() as file_path:
            # Copy the reference, which as an object in it.
            shutil.copy(reference_path, file_path)
            sda_file = SDAFile(file_path, 'a')
            label = 'example I'

            # Replace some stuff with different type
            data = sda_file.extract(label)
            data['Parameter'] = 'hello world'
            with self.assertRaises(ValueError):
                sda_file.update_object(label, data)

    def test_update_object_with_non_record(self):

        reference_path = data_path('SDAreference.sda')
        with temporary_file() as file_path:
            # Copy the reference, which as an object in it.
            shutil.copy(reference_path, file_path)
            sda_file = SDAFile(file_path, 'a')
            label = 'example I'

            # Replace some stuff with a non-dictionary
            with self.assertRaises(ValueError):
                sda_file.update_object(label, 'hello')