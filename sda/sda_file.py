""" Implementation of ``SDAFile`` for working with SDA files.

The SDA format was designed to be universal to facilitate data sharing across
multiple languages. It does contain constructs that are specific to MATLAB.
These constructs, the *function* and *object* data types, cannot be written to
or read from the ``SDAFile`` interface. However, entries of this type do appear
when interrogating the contents of an SDA file.

"""

from contextlib import contextmanager
import os.path as op

import h5py
import numpy as np

from .utils import (
    coerce_character, coerce_complex, coerce_logical, coerce_numeric,
    error_if_bad_header, error_if_not_writable, extract_character,
    extract_complex, extract_logical, extract_numeric, get_date_str,
    get_empty_for_type, infer_record_type, is_valid_writable, write_header,
)


WRITE_MODES = ('w', 'w-', 'x', 'a')


class SDAFile(object):
    """ Read, write, inspect, and manipulate Sandia Data Archive files.

    This supports version 1.1 of the Sandai Data Archive format.

    """

    def __init__(self, name, mode='a', **kw):
        """ Open an SDA file for reading, writing, or interrogation.

        Parameters
        ----------
        name : str
            The name of the file to be loaded or created.
        mode : str
            r         Read-only, file must exist
            r+        Read/write, file must exist
            w         Create file, truncate if exists
            w- or x   Create file, fail if exists
            a         Read/write if exists, create otherwise (default)
        kw :
            Key-word arguments that are passed to the underlying HDF5 file. See
            h5py.File for options.

        """
        file_exists = op.isfile(name)
        self._mode = mode
        self._filename = name
        self._kw = kw

        # Check existence
        if mode in ('r', 'r+') and not file_exists:
            msg = "File '{}' does not exist".format(name)
            raise IOError(msg)

        # Check the header when mode requires the file to exist
        if mode in ('r', 'r+') or (file_exists and mode == 'a'):
            with self._h5file('r') as h5file:
                error_if_bad_header(h5file)

                # Check that file is writable when mode will write to file
                if mode != 'r':
                    error_if_not_writable(h5file)

        # Create the header if this is a new file
        if mode in ('w', 'w-', 'x') or (not file_exists and mode == 'a'):
            with self._h5file(mode) as h5file:
                write_header(h5file.attrs)

    # File properties

    @property
    def name(self):
        """ File name on disk. """
        return self._filename

    @property
    def mode(self):
        """ Mode used to open file. """
        return self._mode

    # Format attrs

    @property
    def FileFormat(self):
        return self._get_attr('FileFormat')

    @property
    def FormatVersion(self):
        return self._get_attr('FormatVersion')

    @property
    def Writable(self):
        return self._get_attr('Writable')

    @Writable.setter
    def Writable(self, value):
        if self._mode not in WRITE_MODES:
            raise ValueError("File is not writable.")
        if not is_valid_writable(value):
            raise ValueError("Must be 'yes' or 'no'")
        with self._h5file('a') as h5file:
            h5file.attrs['Writable'] = value

    @property
    def Created(self):
        return self._get_attr('Created')

    @property
    def Updated(self):
        return self._get_attr('Updated')

    # Public
    def describe(self, label, description=''):
        """ Change the description of a data entry.

        Parameters
        ----------
        label : str
            The data label.
        description : str
            A description to accompany the data

        Raises
        ------
        ValueError if the label contains invalid characters
        ValueError if the label does not exist

        """
        self._validate_can_write()
        self._validate_label(label, must_exist=True)
        with self._h5file('a') as h5file:
            h5file[label].attrs['Description'] = description
        self._update_timestamp()

    def extract(self, label):
        """ Extract data from an SDA file.

        Parameters
        ----------
        label : str
            The data label.

        Raises
        ------
        ValueError if the label contains invalid characters
        ValueError if the label does not exist

        """
        self._validate_can_write()
        self._validate_label(label, must_exist=True)
        with self._h5file('r') as h5file:
            g = h5file[label]
            record_type = g.attrs['RecordType']
            # short circuit empty archives to avoid unnecessarily loading data.
            if g.attrs['Empty'] == 'yes':
                return get_empty_for_type(record_type)
            ds = g[label]
            complex_flag = ds.attrs.get('Complex', 'no')
            shape = ds.attrs.get('ArrayShape')
            data = ds[()]

        if record_type == 'numeric':
            if complex_flag == 'yes':
                extracted = extract_complex(data, shape)
                # squeeze leading dimension if this looks like a 1D array
                if extracted.ndim == 2 and extracted.shape[0] == 1:
                    # if it's a scalar, go all the way
                    if extracted.shape[1] == 1:
                        extracted = extracted[0, 0]
                    else:
                        extracted = np.squeeze(extracted, axis=0)
            else:
                extracted = extract_numeric(data)
        elif record_type == 'logical':
            extracted = extract_logical(data)
        elif record_type == 'character':
            extracted = extract_character(data)

        return extracted

    def insert(self, label, data, description='', deflate=0):
        """ Insert data into an SDA file.

        Parameters
        ----------
        label : str
            The data label.
        data :
            The data to insert. 'numeric', 'logical', and 'character' types are
            supported. Strings are accepted as 'character' type. 'numeric' and
            'logical' types can be scalar or array-like.
        description : str, optional
            A description to accompany the data
        deflate : int, optional
            An integer value from 0 to 9, specifying the compression level to
            be applied to the stored data.

        Raises
        ------
        ValueError if the data is of an unsupported type
        ValueError if the label contains invalid characters
        ValueError if the label exists

        Note
        ----
        This relies on numpy to cast inhomogeneous array-like data to a
        homogeneous type.  It is the responsibility of the caller to homogenize
        the input data if the numpy casting machinery is not sufficient for the
        input data.

        """
        self._validate_can_write()
        self._validate_label(label, can_exist=False)
        if not isinstance(deflate, int) or not 0 <= deflate <= 9:
            msg = "'deflate' must be an integer from 0 to 9"
            raise ValueError(msg)
        record_type, cast_obj = infer_record_type(data)
        if record_type is None:
            msg = "{!r} is not a supported type".format(data)
            raise ValueError(data)

        is_complex = False
        original_shape = None
        if record_type == 'numeric':
            if np.iscomplexobj(cast_obj):
                is_complex = True
                original_shape = np.atleast_2d(cast_obj).shape
                cast_obj = coerce_complex(cast_obj)
            else:
                cast_obj = coerce_numeric(cast_obj)
        elif record_type == 'logical':
            cast_obj = coerce_logical(cast_obj)
        elif record_type == 'character':
            cast_obj = coerce_character(cast_obj)
        else:
            # Should not happen
            msg = "Unrecognized record type '{}'".format(record_type)
            raise RuntimeError(msg)

        self._insert_data(
            label, cast_obj, description, deflate, record_type, is_complex,
            original_shape
        )
        self._update_timestamp()

    # Private
    def _insert_data(self, label, data, description, deflate, record_type,
                     is_complex, original_shape):
        """ Insert coerced data of a given type into the h5 file.

        """
        empty = 'no'
        if np.isscalar(data) or data.shape == ():
            maxshape = None
            compression = None
            if np.isnan(data):
                empty = 'yes'
        else:
            compression = deflate
            maxshape = (None,) * data.ndim
            if np.squeeze(data).shape == (0,):
                empty = 'yes'

        with self._h5file('a') as h5file:
            g = h5file.create_group(label)
            g.attrs['RecordType'] = record_type
            g.attrs['Deflate'] = deflate
            g.attrs['Description'] = description
            g.attrs['Empty'] = empty

            ds = g.create_dataset(
                label,
                maxshape=maxshape,
                data=data,
                compression=compression,
            )
            ds.attrs['RecordType'] = record_type
            ds.attrs['Empty'] = empty
            ds.attrs['Complex'] = 'yes' if is_complex else 'no'
            if is_complex:
                ds.attrs['ArrayShape'] = original_shape

    def _is_existing_label(self, label):
        with self._h5file('r') as h5file:
            return label in h5file

    @contextmanager
    def _h5file(self, mode):
        h5file = h5py.File(self._filename, mode, **self._kw)
        try:
            yield h5file
        finally:
            h5file.close()

    def _get_attr(self, attr):
        with self._h5file('r') as h5file:
            return h5file.attrs[attr]

    def _validate_can_write(self):
        """ Validate file mode and 'Writable' attr allow writing. """
        if self._mode not in WRITE_MODES:
            raise IOError("File is not writable")
        if self.Writable == 'no':
            raise IOError("'Writable' flag is 'no'")

    def _validate_label(self, label, can_exist=True, must_exist=False):
        if '/' in label or '\\' in label:
            msg = r"label cannot contain '/' or '\'"
            raise ValueError(msg)
        label_exists = self._is_existing_label(label)
        if not can_exist and label_exists:
            msg = "Label '{}' already exists."
            raise ValueError(msg)
        if must_exist and not label_exists:
            msg = "Label item '{}' does not exist".format(label)
            raise ValueError(msg)

    def _update_timestamp(self):
        with self._h5file('a') as h5file:
            h5file.attrs['Updated'] = get_date_str()
