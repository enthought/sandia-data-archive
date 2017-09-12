""" Implementation of ``SDAFile`` for working with SDA files.

The SDA format was designed to be universal to facilitate data sharing across
multiple languages. It supports reading and updating all record types, except
*function* records. It support writing *numeric*, *logical*, *cell*, and
*structure* records.

"""

from contextlib import contextmanager
import os
import os.path as op
import re
import shutil
import tempfile

import h5py
import numpy as np

from .utils import (
    coerce_primitive, error_if_bad_header, error_if_not_writable,
    extract_primitive, get_decoded, get_empty_for_type, infer_record_type,
    is_primitive, is_supported, is_valid_matlab_field_label, is_valid_writable,
    set_encoded, update_header, write_header,
)


WRITE_MODES = ('w', 'w-', 'x', 'a')


class SDAFile(object):
    """ Read, write, inspect, and manipulate Sandia Data Archive files.

    This supports version 1.1 of the Sandia Data Archive format.

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
            grp = h5file[label]
            attrs = get_decoded(grp.attrs)
            return self._extract_data_from_group(grp, label, attrs)

    def insert(self, label, data, description='', deflate=0):
        """ Insert data into an SDA file.

        Parameters
        ----------
        label : str
            The data label.
        data :
            The data to insert. See the notes below.
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
        This stores specific data types as described here.

        sequences :
            Lists, tuples, and anything else that identifies as a
            collections.Sequence are stored as 'cell' records, no
            matter the contents.

        mappings :
            Dictionaries and anything else that identifies as
            collections.Mapping and not another type listed here are stored as
            'structure' records.

        numpy arrays :
            If the dtype is a supported numeric type, then a numpy array is
            stored as a 'numeric' record. Arrays of 'bool' type are stored as
            'logical' records. Object arrays are stored as 'cell' records.

        sparse arrays (from scipy.sparse) :
            These are stored as 'numeric' records if the dtype is a type
            supported for numpy arrays.

        strings :
            Strings are stored as 'character' records. An attempt will be
            made to convert the input to ascii encoded bytes, no matter the
            underlying encoding. This may result in an encoding exception if
            the input cannot be ascii encoded.

        non-string scalars :
            Non-string scalars are stored as 'numeric' if numeric, or 'logical'
            if boolean.

        other :
            Arrays of characters are not supported. Convert to a string.
            Object arrays are not supported. Cast to another dtype or turn into
            a list.

        Anything not listed above is not supported.

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

        with self._h5file('a') as h5file:
            grp = h5file.create_group(label)
            # Write the known header information
            set_encoded(
                grp.attrs,
                RecordType=record_type,
                Deflate=deflate,
                Description=description,
            )

            if is_primitive(record_type):
                self._insert_primitive_data(
                    grp, label, deflate, record_type, cast_obj, extra
                )
            else:
                try:
                    self._insert_composite_data(
                        grp, deflate, record_type, cast_obj, extra
                    )
                except ValueError:
                    # If something goes wrong, don't leave the archive in an
                    # inconsistent state
                    del h5file[label]
                    raise

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

        # Create a new file so space is actually freed
        labels = set(labels)

        def _copy_visitor(path):
            """ Visitor that copies data from source to destination """

            # Skip paths corresponding to excluded labels
            if path.split('/')[0] in labels:
                return

            # Copy everything else
            source_obj = source[path]
            destination.attrs.update(source.attrs)
            if isinstance(source_obj, h5py.Group):
                destination.create_group(path)
            else:
                ds = source_obj
                destination.create_dataset(
                    path,
                    data=source_obj[()],
                    chunks=ds.chunks,
                    maxshape=ds.maxshape,
                    compression=ds.compression,
                    compression_opts=ds.compression_opts,
                    scaleoffset=ds.scaleoffset,
                    shuffle=ds.shuffle,
                    fletcher32=ds.fletcher32,
                    fillvalue=ds.fillvalue,
                )

        pid, destination_path = tempfile.mkstemp()
        os.close(pid)
        destination = h5py.File(destination_path, 'w')
        with self._h5file('r') as source:
            destination.attrs.update(source.attrs)
            source.visit(_copy_visitor)
        update_header(destination.attrs)
        destination.close()
        shutil.move(destination_path, self._filename)

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

    @contextmanager
    def _h5file(self, mode):
        h5file = h5py.File(self._filename, mode, **self._kw)
        try:
            yield h5file
        finally:
            h5file.close()

    def _extract_data_from_group(self, grp, label, attrs):
        """ Extract data from h5 group. ``label`` is the group label. """

        record_type = attrs['RecordType']
        if not is_supported(record_type):
            msg = "RecordType '{}' is not supported".format(record_type)
            raise ValueError(msg)

        empty = attrs['Empty']
        if empty == 'yes':
            return get_empty_for_type(record_type)

        if is_primitive(record_type):
            ds = grp[label]
            data_attrs = get_decoded(ds.attrs)
            return extract_primitive(record_type, ds[()], data_attrs)

        if record_type in ('cell', 'structures', 'objects'):
            record_size = attrs['RecordSize'].astype(int)
            nr = np.prod(record_size)
            labels = ['element {}'.format(i) for i in range(1, nr + 1)]
            data = self._extract_composite_data(grp, labels)
            if record_size[0] > 1 or len(record_size) > 2:
                data = np.array(
                    data, dtype=object,
                ).reshape(record_size, order='F')
        elif record_type in ('structure', 'object'):
            labels = attrs['FieldNames'].split()
            data = self._extract_composite_data(grp, labels)
            data = dict(zip(labels, data))

        return data

    def _extract_composite_data(self, grp, labels):
        """ Extract composite data from a Group object with given labels. """
        extracted = []
        for label in labels:
            sub_obj = grp[label]
            attrs = get_decoded(sub_obj.attrs)
            record_type = attrs['RecordType']
            if is_primitive(record_type):
                element = extract_primitive(record_type, sub_obj[()], attrs)
            else:  # composite type
                element = self._extract_data_from_group(sub_obj, label, attrs)
            extracted.append(element)
        return extracted

    def _insert_composite_data(self, grp, deflate, record_type, data, extra):
        attrs = {}
        if record_type == 'cell':
            if isinstance(data, np.ndarray):
                record_size = np.atleast_2d(data).shape
                data = data.ravel(order='F')
            else:
                record_size = (1, len(data))
            nr = np.prod(record_size)
            labels = ['element {}'.format(i) for i in range(1, nr + 1)]
            attrs['RecordSize'] = record_size
        elif record_type == 'structure':
            nr = len(data)
            data = sorted((str(key), value) for key, value in data.items())
            labels, data = zip(*data)
            # Check that each label is a valid MATLAB field label
            for label in labels:
                if not is_valid_matlab_field_label(label):
                    msg = "Key '{}' is not a valid MATLAB field label"
                    raise ValueError(msg.format(label))
            attrs['FieldNames'] = ' '.join(labels)
        else:
            raise ValueError(record_type)

        attrs['Empty'] = 'yes' if nr == 0 else 'no'
        set_encoded(grp.attrs, **attrs)

        for label, sub_data in zip(labels, data):
            sub_rec_type, sub_data, sub_extra = infer_record_type(sub_data)
            if is_primitive(sub_rec_type):
                self._insert_primitive_data(
                    grp, label, deflate, sub_rec_type, sub_data,
                    sub_extra
                )
            else:
                sub_grp = grp.create_group(label)
                set_encoded(
                    sub_grp.attrs,
                    RecordType=sub_rec_type
                )
                self._insert_composite_data(
                    sub_grp, deflate, sub_rec_type, sub_data, sub_extra
                )

    def _insert_primitive_data(self, grp, label, deflate, record_type, data,
                               extra):
        """ Prepare primitive data for storage and store it. """
        data, original_shape = coerce_primitive(record_type, data, extra)
        empty = 'no'
        if np.isscalar(data) or data.shape == ():
            compression = None
            maxshape = None
            if np.isnan(data):
                empty = 'yes'
        else:
            compression = deflate
            maxshape = (None,) * data.ndim
            if np.squeeze(data).shape == (0,):
                empty = 'yes'

        with self._h5file('a') as h5file:
            set_encoded(
                grp.attrs,
                Empty=empty,
            )

            ds = grp.create_dataset(
                label,
                maxshape=maxshape,
                data=data,
                compression=compression,
            )
            data_attrs = {}
            is_numeric = record_type == 'numeric'
            is_complex = extra is not None and extra.endswith('complex')
            is_sparse = extra is not None and extra.startswith('sparse')
            data_attrs['RecordType'] = record_type
            data_attrs['Empty'] = empty
            if is_numeric:
                data_attrs['Complex'] = 'yes' if is_complex else 'no'
                data_attrs['Sparse'] = 'yes' if is_sparse else 'no'
            if is_complex:
                data_attrs['ArraySize'] = original_shape
            set_encoded(ds.attrs, **data_attrs)
            update_header(h5file.attrs)

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
