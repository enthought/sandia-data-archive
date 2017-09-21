""" Inserter class for SDAFile. """

import numpy as np

from .utils import (
    cell_label, coerce_simple, infer_record_type, is_simple,
    is_valid_matlab_field_label, set_encoded, update_header,
)


class Inserter(object):
    """ Insert records into an archive.

    An Inserter serves as a context for the data that is to go into an archive.
    It manages inserting simple (single-level) data and traversing composite
    data to place it into the archive.

    """

    def __init__(self, label, data, deflate):
        """ Data insertion context for flat or hierarchical data.

        This does no validation of the inputs.

        """
        self.label = label
        self.deflate = int(deflate)
        type_info = infer_record_type(data)
        self.record_type, self.data, is_empty, self.extra = type_info
        self.is_simple = is_simple(self.record_type)
        self.empty = 'yes' if is_empty else 'no'

    def insert(self, h5file, description):
        """ Insert the data into an h5py File. """
        grp = self._create_group(
            h5file,
            self.label,
            Description=description
        )
        try:
            if self.is_simple:
                self._insert_simple_data(grp)
            else:
                self._insert_composite_data(grp)
        except ValueError:
            # If something goes wrong, don't leave the archive in an
            # inconsistent state
            del h5file[self.label]
            raise

        update_header(h5file.attrs)

    def _insert_simple_data(self, group):
        """ Store simple (non-hierarchical) data. """
        record_type = self.record_type
        data = self.data
        extra = self.extra
        data, original_shape = coerce_simple(record_type, data, extra)
        maxshape = (None,) * data.ndim
        data_attrs = {}
        is_numeric = record_type == 'numeric'
        is_complex = extra is not None and extra.endswith('complex')
        is_sparse = extra is not None and extra.startswith('sparse')
        data_attrs['RecordType'] = self.record_type
        data_attrs['Empty'] = self.empty
        if is_numeric:
            data_attrs['Complex'] = 'yes' if is_complex else 'no'
            data_attrs['Sparse'] = 'yes' if is_sparse else 'no'
        if is_complex:
            data_attrs['ArraySize'] = original_shape

        self._create_dataset(
            parent=group,
            label=self.label,
            data=data,
            maxshape=maxshape,
            **data_attrs
        )

    def _insert_composite_data(self, group):
        data = self.data
        # Specify additional group attributes
        group_attrs = {}
        if self.record_type == 'cell':
            if isinstance(data, np.ndarray):
                record_size = np.atleast_2d(data).shape
                data = data.ravel(order='F')
            else:
                record_size = (1, len(data))
            nr = np.prod(record_size)
            labels = [cell_label(i) for i in range(1, nr + 1)]
            group_attrs['RecordSize'] = record_size
        elif self.record_type == 'structure':
            nr = len(data)
            data = sorted((str(key), value) for key, value in data.items())
            labels, data = zip(*data)
            # Check that each label is a valid MATLAB field label
            for label in labels:
                if not is_valid_matlab_field_label(label):
                    msg = "Key '{}' is not a valid MATLAB field label"
                    raise ValueError(msg.format(label))
            group_attrs['FieldNames'] = ' '.join(labels)

        set_encoded(group.attrs, **group_attrs)

        # Insert the sub-data
        for label, sub_data in zip(labels, data):
            inserter = Inserter(label, sub_data, self.deflate)
            if inserter.is_simple:  # Insert into existing group
                inserter._insert_simple_data(group)
            else:                   # Create new group for composite object
                parent = inserter._create_group(group, label)
                inserter._insert_composite_data(parent)

    def _create_dataset(self, parent, label, data, maxshape, **ds_attrs):
        ds = parent.create_dataset(
            label,
            maxshape=maxshape,
            data=data,
            compression=self.deflate,
        )
        set_encoded(ds.attrs, **ds_attrs)
        return ds

    def _create_group(self, parent, label, **extra_attrs):
        """ Create a group for a new record or composite record. """
        grp = parent.create_group(label)
        set_encoded(
            grp.attrs,
            RecordType=self.record_type,
            Deflate=self.deflate,
            Empty=self.empty,
            **extra_attrs
        )
        return grp
