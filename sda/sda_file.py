""" Implementation of ``SDAFile`` for working with SDA files.

The SDA format was designed to be universal to facilitate data sharing across
multiple languages. It does contain constructs that are specific to MATLAB.
These constructs, the *function* and *object* data types, cannot be written to
or read from the ``SDAFile`` interface. However, entries of this type do appear
when interrogating the contents of an SDA file.

"""

from contextlib import contextmanager
import os.path as op
import re

import h5py
import numpy as np

from .utils import (
    coerce_character, coerce_complex, coerce_logical, coerce_numeric,
    coerce_sparse, error_if_bad_header, error_if_not_writable,
    extract_character, extract_complex, extract_logical, extract_numeric,
    extract_sparse, get_decoded, get_empty_for_type, infer_record_type,
    is_valid_writable, set_encoded, update_header, write_header,
)


SUPPORTED_RECORD_TYPES = ('character', 'logical', 'numeric')

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

            # Check that file is writable when mode will write to file.
            if mode != 'r':
                with self._h5file('a') as h5file:
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
            set_encoded(h5file.attrs, Writable=value)

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
        self._validate_labels(label, must_exist=True)
        with self._h5file('a') as h5file:
            set_encoded(h5file[label].attrs, Description=description)
            update_header(h5file.attrs)

    def extract(self, label):
        """ Extract data from an SDA file.

        Parameters
        ----------
        label : str
            The data label.

        Notes
        -----
        Sparse numeric data is extracted as scipy.sparse.coo_matrix. This
        format does not support all numpy operations. See the
        ``scipy.sparse.coo_matrix`` documentation for details.

        Raises
        ------
        ValueError if the label contains invalid characters
        ValueError if the label does not exist

        """
        self._validate_labels(label, must_exist=True)
        with self._h5file('r') as h5file:
            g = h5file[label]
            group_attrs = get_decoded(g.attrs, 'RecordType', 'Empty')
            record_type = group_attrs['RecordType']
            if record_type not in SUPPORTED_RECORD_TYPES:
                msg = "RecordType '{}' is not supported".format(record_type)
                raise ValueError(msg)
            # short circuit empty archives to avoid unnecessarily loading data.
            if group_attrs['Empty'] == 'yes':
                return get_empty_for_type(record_type)
            ds = g[label]
            data_attrs = get_decoded(ds.attrs, 'Complex', 'ArraySize')
            complex_flag = data_attrs.get('Complex', 'no')
            sparse_flag = data_attrs.get('Sparse', 'no')
            shape = data_attrs.get('ArraySize', None)
            data = ds[()]

        if record_type == 'numeric':
            if sparse_flag == 'yes':  # FIXME - add complex sparse
                extracted = extract_sparse(data)
            elif complex_flag == 'yes':
                extracted = extract_complex(data, shape.astype(int))
            else:
                extracted = extract_numeric(data)
            # squeeze leading dimension if this is a MATLAB row array
            if extracted.ndim == 2 and extracted.shape[0] == 1:
                # if it's a scalar, go all the way
                if extracted.shape[1] == 1:
                    extracted = extracted[0, 0]
                else:
                    extracted = np.squeeze(extracted, axis=0)
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

        Notes
        -----
        This relies on numpy to cast inhomogeneous array-like data to a
        homogeneous type.  It is the responsibility of the caller to homogenize
        the input data if the numpy casting machinery is not sufficient for the
        input data.

        Sparse matrices are converted to COO form for storing. This may be
        inefficient for some sparse matrices.

        """
        self._validate_can_write()
        self._validate_labels(label, can_exist=False)
        if not isinstance(deflate, (int, np.integer)) or not 0 <= deflate <= 9:
            msg = "'deflate' must be an integer from 0 to 9"
            raise ValueError(msg)
        record_type, cast_obj, extra = infer_record_type(data)
        if record_type is None:
            msg = "{!r} is not a supported type".format(data)
            raise ValueError(data)

        original_shape = None
        if record_type == 'numeric':
            if extra == 'complex':
                original_shape = np.atleast_2d(cast_obj).shape
                cast_obj = coerce_complex(cast_obj)
            elif extra == 'sparse':
                cast_obj = coerce_sparse(cast_obj)
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
            label, cast_obj, description, deflate, record_type, extra,
            original_shape
        )

    def labels(self):
        """ Get data labels from the archive. """
        with self._h5file('r') as h5file:
            return list(h5file.keys())

    def remove(self, *labels):
        """ Remove specified labels from the archive.

        This cannot be undone.

        """
        self._validate_can_write()
        if len(labels) == 0:
            msg = "Specify labels to remove"
            raise ValueError(msg)

        self._validate_labels(labels, must_exist=True)

        with self._h5file('a') as h5file:
            for label in labels:
                del h5file[label]
            update_header(h5file.attrs)

    def probe(self, pattern=None):
        """ Summarize the state of the archive

        This requires the pandas package.

        Parameters
        ----------
        pattern : str or None, optional
            A search pattern (python regular expression) applied to find
            archive labels of interest. If None, all labels are selected.

        Returns
        -------
        summary : DataFrame
            A table summarizing the archive.

        """
        from pandas import DataFrame
        labels = self.labels()
        if pattern is not None:
            regex = re.compile(pattern)
            labels = [
                label for label in labels if regex.match(label) is not None
            ]

        summary = []
        with self._h5file('r') as h5file:
            for label in labels:
                g = h5file[label]
                attrs = get_decoded(g.attrs)
                if label in g:
                    attrs.update(get_decoded(g[label].attrs))
                attrs['label'] = label
                summary.append(attrs)

        cols = [
            'label', 'RecordType', 'Description', 'Empty', 'Deflate',
            'Complex', 'ArraySize', 'Sparse', 'RecordSize', 'Class',
            'FieldNames', 'Command',
        ]
        return DataFrame(summary, columns=cols).set_index('label').fillna('')

    def replace(self, label, data):
        """ Replace an existing dataset.

        This is equivalent to removing the data and inserting a new entry using
        the same label, description, and deflate option.

        """
        self._validate_can_write()
        self._validate_labels(label, must_exist=True)
        with self._h5file('r') as h5file:
            attrs = get_decoded(h5file[label].attrs, 'Deflate', 'Description')
        self.remove(label)
        self.insert(label, data, attrs['Description'], attrs['Deflate'])

    # Private
    def _insert_data(self, label, data, description, deflate, record_type,
                     extra, original_shape):
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
            set_encoded(
                g.attrs,
                RecordType=record_type,
                Deflate=deflate,
                Description=description,
                Empty=empty,
            )

            ds = g.create_dataset(
                label,
                maxshape=maxshape,
                data=data,
                compression=compression,
            )
            data_attrs = {}
            is_numeric = record_type == 'numeric'
            is_complex = extra == 'complex'
            is_sparse = extra == 'sparse'
            data_attrs['RecordType'] = record_type
            data_attrs['Empty'] = empty
            if is_numeric:
                data_attrs['Complex'] = 'yes' if is_complex else 'no'
                data_attrs['Sparse'] = 'yes' if is_sparse else 'no'
            if is_complex:
                data_attrs['ArraySize'] = original_shape
            set_encoded(ds.attrs, **data_attrs)
            update_header(h5file.attrs)

    @contextmanager
    def _h5file(self, mode):
        h5file = h5py.File(self._filename, mode, **self._kw)
        try:
            yield h5file
        finally:
            h5file.close()

    def _get_attr(self, attr):
        """ Get a named atribute as a string """
        with self._h5file('r') as h5file:
            return get_decoded(h5file.attrs, attr)[attr]

    def _validate_can_write(self):
        """ Validate file mode and 'Writable' attr allow writing. """
        if self._mode not in WRITE_MODES:
            raise IOError("File is not writable")
        if self.Writable == 'no':
            raise IOError("'Writable' flag is 'no'")

    def _validate_labels(self, labels, can_exist=True, must_exist=False):
        if isinstance(labels, str):
            labels = [labels]
        for label in labels:
            if '/' in label or '\\' in label:
                msg = r"label cannot contain '/' or '\'"
                raise ValueError(msg)
        with self._h5file('r') as h5file:
            for label in labels:
                label_exists = label in h5file
            if not can_exist and label_exists:
                msg = "Label '{}' already exists.".format(label)
                raise ValueError(msg)
            if must_exist and not label_exists:
                msg = "Label item '{}' does not exist".format(label)
                raise ValueError(msg)
