from contextlib import contextmanager
import os
import tempfile

import h5py


BAD_ATTRS = {
    'FileFormat': 'SDB',
    'FormatVersion': '0.5',
    'Writable': 'nope',
    'Created': '2017-01-01 01:23:45',
    'Modified': '2017-01-01 01:23:45',
}


GOOD_ATTRS = {
    'FileFormat': 'SDA',
    'FormatVersion': '1.1',
    'Writable': 'yes',
    'Created': '18-Aug-2017 01:23:45',
    'Modified': '18-Aug-2017 01:23:45',
}


@contextmanager
def temporary_file(suffix='.sda'):
    pid, file_path = tempfile.mkstemp(suffix=suffix)
    os.close(pid)
    try:
        yield file_path
    finally:
        os.remove(file_path)


@contextmanager
def temporary_h5file(suffix='.sda'):
    with temporary_file(suffix) as file_path:
        h5file = h5py.File(file_path, 'w')
        try:
            yield h5file
        finally:
            if h5file.id.valid:  # file is open
                h5file.close()
