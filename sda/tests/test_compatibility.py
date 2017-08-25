import unittest

import numpy as np
from numpy.testing import assert_array_equal

from sda.sda_file import SDAFile
from sda.testing import data_path


class TestSDAReference(unittest.TestCase):
    """ Test loading SDAreference.sda """

    def setUp(self):
        self.sda_file = SDAFile(data_path('SDAreference.sda'), 'r')

    def tearDown(self):
        del self.sda_file

    def test_header(self):
        sda_file = self.sda_file
        self.assertEqual(sda_file.FileFormat, 'SDA')
        self.assertEqual(sda_file.FormatVersion, '1.1')
        self.assertEqual(sda_file.Writable, 'yes')
        self.assertEqual(sda_file.Created, '18-Nov-2016 14:08:16')
        self.assertEqual(sda_file.Updated, '18-Nov-2016 14:08:17')

    def test_ReferenceArchivedotm(self):
        label = 'ReferenceArchive.m'
        self.assertUnsupported(label)

    def test_example_A1(self):
        """ 5x1 zeros """
        label = 'example A1'
        expected = np.zeros(5, dtype=np.float64)
        self.assertArray(label, expected)

    def test_example_A2(self):
        """ 4x3 imaginary numbers """
        label = 'example A2'
        expected = np.empty((4, 3), dtype=np.complex128)
        expected.real = 0
        expected.imag = 1
        self.assertArray(label, expected)

    def test_example_A3(self):
        """ 5x5 sparse matrix """
        label = 'example A3'
        extracted = self.sda_file.extract(label)
        assert_array_equal(extracted.toarray(), np.eye(5))

    def test_example_A4(self):
        """ Empty array """
        label = 'example A4'
        data = self.sda_file.extract(label)
        self.assertTrue(np.isnan(data))

    def test_example_B(self):
        """ Logical scalar """
        label = 'example B'
        self.assertScalar(label, True)

    def test_example_C(self):
        """ Some text """
        label = 'example C'
        expected = 'Here is some text'
        self.assertScalar(label, expected)

    def test_example_D(self):
        """ Handle to the sine function """
        label = "example D"
        self.assertUnsupported(label)

    def test_example_E(self):
        """ Cell array combining examples A1 and A2 """
        label = "example E"
        self.assertUnsupported(label)

    def test_example_F(self):
        """ Structure combining examples A1 and A2 """
        label = "example F"
        self.assertUnsupported(label)

    def test_example_G(self):
        """ Structure array combining examples A1 and A2 (repeated) """
        label = "example G"
        self.assertUnsupported(label)

    def test_example_H(self):
        """ Cell array of structures combining examples A1-A4 """
        label = "example H"
        self.assertUnsupported(label)

    def test_example_I(self):
        """ Object containing example A1 """
        label = "example I"
        self.assertUnsupported(label)

    def test_example_J(self):
        """ Object array containing examples A1 and A2 """
        label = "example J"
        self.assertUnsupported(label)

    def assertArray(self, label, expected):
        extracted = self.sda_file.extract(label)
        self.assertEqual(extracted.dtype, expected.dtype)
        assert_array_equal(extracted, expected)

    def assertScalar(self, label, expected):
        extracted = self.sda_file.extract(label)
        self.assertEqual(extracted, expected)

    def assertUnsupported(self, label):
        with self.assertRaises(ValueError):
            self.sda_file.extract(label)
