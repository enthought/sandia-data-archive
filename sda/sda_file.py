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
    error_if_bad_header, error_if_not_writable, get_date_str,
    infer_record_type, is_valid_writable, write_header,
)


WRITE_MODES = ('w', 'w-', 'x', 'a')


class SDAFile(object):
    """ Read, write, inspect, and manipulate Sandia Data Archive files.

    This supports version 1.0 of the Sandai Data Archive format, defined at
    http://prod.sandia.gov/techlib/access-control.cgi/2015/151118.pdf.

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
        with self._h5file('w') as h5file:
            h5file.attrs['Writable'] = value

    @property
    def Created(self):
        return self._get_attr('Created')

    @property
    def Modified(self):
        return self._get_attr('Modified')

    # Public
    def insert(self, label, data, description='', deflate=0):
        """ Insert data into an SDA file.

        Parameters
        ----------
        label : str
            The data label.
        data :
            The data to insert. 'numeric', 'logical', and 'character' types are
            supported.
        description : str, optional
            A description to accompany the data
        deflate : int, optional
            An integer value from 0 to 9, specifying the compression level to
            be applied to the stored data.

        Raises
        ------
        ValueError if the data is of an unsupported type
        ValueError if the label is invalid
        ValueError if the label exists

        Note
        ----
        This relies on numpy to upcast inhomogeneous data to homogeneous type.
        This may result in casting to a supported type.  For example, ``[3,
        'hello']`` will be cast as 'character' type. It is the responsibility
        of the caller to homogenize the input data if the numpy casting
        machinery is not sufficient for the input data.

        """
        if self._mode not in WRITE_MODES:
            raise IOError("File is not writable")
        if self.Writable == 'no':
            raise IOError("'Writable' flag is 'no'")
        if not isinstance(deflate, int) or not 0 <= deflate <= 9:
            msg = "'deflate' must be an integer from 0 to 9"
            raise ValueError(msg)
        record_type, cast_obj = infer_record_type(data)
        if record_type is None:
            msg = "{!r} is not a supported type".format(data)
            raise ValueError(data)
        if '/' in label or '\\' in label:
            msg = r"label cannot contain '/' or '\'"
            raise ValueError(msg)
        if self._label_exists(label):
            msg = "Label '{}' already exists. Call 'replace' to replace it."
            raise ValueError(msg.format(label))

        if record_type == 'numeric':
            self._insert_numeric(label, cast_obj, description, deflate)
        elif record_type == 'logical':
            self._insert_logical(label, cast_obj, description, deflate)
        elif record_type == 'character':
            self._insert_character(label, cast_obj, description, deflate)
        else:
            # Should not happen
            msg = "Unrecognized record type '{}'".format(record_type)
            raise RuntimeError(msg)

        self._update_modified()

    # Private
    def _insert_data(self, label, data, description, deflate, record_type):
        """ Worker for the _insert methods

        This expects the data to be a scalar or cast as a numpy array.

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

    def _insert_numeric(self, label, data, description, deflate):
        self._insert_data(label, data, description, deflate, 'numeric')

    def _insert_logical(self, label, data, description, deflate):
        # Coerce the stored type to uint8
        if np.isscalar(data) or data.shape == ():
            data = 1 if data else 0
        else:
            data = data.astype(np.uint8).clip(0, 1)

        self._insert_data(label, data, description, deflate, 'logical')

    def _label_exists(self, label):
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

    def _update_modified(self):
        with self._h5file('a') as h5file:
            h5file.attrs['Modified'] = get_date_str()
