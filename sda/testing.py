from contextlib import contextmanager
import os
import tempfile

import h5py


@contextmanager
def temporary_h5file(suffix='.sda'):
    pid, file_path = tempfile.mkstemp(suffix=suffix)
    os.close(pid)
    h5file = h5py.File(file_path, 'w')
    try:
        yield h5file
    finally:
        h5file.close()
        os.remove(file_path)
