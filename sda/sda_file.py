""" Implementation of ``SDAFile`` for working with SDA files.

The SDA format was designed to be universal to facilitate data sharing across
multiple languages. It does contain constructs that are specific to MATLAB.
These constructs, the *function* and *object* data types cannot be written to
or read from the ``SDAFile`` interface. However, entries of this type do appear
when interrogating the contents of an SDA file.

"""

from contextlib import contextmanager
import os.path as op

import h5py

from .utils import (
    error_if_bad_header, error_if_not_writable, is_valid_writable,
    write_header,
)


class SDAFile(object):
    """ Read, write, inspect, and manipulate Sandia Data Archive files.

    This supports version 1.0 of the Sandai Data Archive format, defined at
    http://prod.sandia.gov/techlib/access-control.cgi/2015/151118.pdf.

    """

    # The version supported by this implementation. If SDA version 2 comes
    # along, a proper SDAFile interface, object hierarchy, and version support
    # should be implemented.
    supported_version = '1'

    def __init__(self, name, mode, **kw):
        """ Open an SDA file for reading, writing, or interrogation.

        Parameters
        ----------
        name : str
            The name of the file to be loaded or created.
        mode : str
            r       Read-only, file must exist
            r+      Read/write, file must exist
            w       Create file, truncate if exists
            w- or x Create file, fail if exists
            a       Read/write if exists, create otherwise (default)
        kw :
            Key-word arguments that are passed to the underlying HDF5 file. See
            h5py.File for options.

        """
        file_exists = op.isfile(name)
        self._mode = mode
        self._filename = name

        # Check existence
        if mode in ('r', 'r+') and not file_exists:
            msg = "File '{}' does not exist".format(name)
            raise IOError(msg)

        # Check the header when mode requires the file to exist
        with self._h5file() as h5file:
            if mode in ('r', 'r+') or (file_exists and mode == 'a'):
                error_if_bad_header(h5file)

            # Check that file is writable when mode will write to file
            if mode == 'r+' or (file_exists and mode == 'a'):
                error_if_not_writable(h5file)

            # Create the header if this is a new file
            if mode in ('w', 'w-', 'x') or (not file_exists and mode == 'a'):
                write_header(h5file)

    # File properties

    @property
    def name(self):
        """ File name on disk. """
        return self._filename

    @property
    def mode(self):
        """ Mode used to open file. """
        return self.mode

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
        if not is_valid_writable(value):
            raise ValueError("Must be 'yes' or 'no'")
        with self._h5file() as h5file:
            h5file.attrs['Writable'] = value

    @property
    def Created(self):
        return self._get_attr('Created')

    @property
    def Modified(self):
        return self._get_attr('Modified')

    # Private

    @contextmanager
    def _h5file(self):
        h5file = h5py.File(self._filename, self._mode)
        try:
            yield h5file
        finally:
            h5file.close()

    def _get_attr(self, attr):
        with self._h5file() as h5file:
            return h5file.attrs[attr]
