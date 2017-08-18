""" Implementation of ``SDAFile`` for working with SDA files.

The SDA format was designed to be universal to facilitate data sharing across
multiple languages. It does contain constructs that are specific to MATLAB.
These constructs, the *function* and *object* data types cannot be written to
or read from the ``SDAFile`` interface. However, entries of this type do appear
when interrogating the contents of an SDA file.

"""

import os.path as op

import h5py

from .utils import (
    error_if_bad_header, error_if_not_writable, get_date_str, is_valid_writable
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
        self._h5file = h5py.File(name, mode, **kw)

        # Check the header when mode requires the file to exist
        if mode in ('r', 'r+') or (file_exists and mode == 'a'):
            error_if_bad_header(self._h5file)

        # Check that file is writable when mode will write to file
        if mode == 'r+' or (file_exists and mode == 'a'):
            error_if_not_writable(self._h5file)

        # Create the header if this is a new file
        if mode in ('w', 'w-', 'x') or not (file_exists and mode == 'a'):
            self._write_header()

    # File properties

    @property
    def name(self):
        """ File name on disk. """
        return self._h5file.filename

    @property
    def mode(self):
        """ Mode used to open file. """
        return self._h5file.mode

    # Format attrs

    @property
    def FileFormat(self):
        return self._h5file.attrs['FileFormat']

    @property
    def FormatVersion(self):
        return self._h5file.attrs['FormatVersion']

    @property
    def Writable(self):
        return self._h5file.attrs['Writable']

    @Writable.setter
    def Writable(self, value):
        if not is_valid_writable(value):
            raise ValueError("Must be 'yes' or 'no'")
        self._h5file.attrs['Writable'] = value

    @property
    def Created(self):
        return self._h5file.attrs['Created']

    @property
    def Modified(self):
        return self._h5file.attrs['Modified']

    # Private

    def _write_header(self):
        attrs = self._h5file.attrs
        attrs['FileFormat'] = 'SDA'
        attrs['FormatVersion'] = '1.0'
        attrs['Writable'] = 'yes'
        date_str = get_date_str()
        attrs['Created'] = date_str
        attrs['Modified'] = date_str
