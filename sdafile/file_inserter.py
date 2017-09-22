import numpy as np

from .record_inserter import SimpleRecordInserter, inserter


@inserter
class FileInserter(SimpleRecordInserter):
    """ Inserter for file-like objects. """

    record_type = 'file'

    @staticmethod
    def can_insert(data):
        """ Can insert file-like objects. """
        return hasattr(data, 'read')

    def prepare_data(self):
        contents = self.data.read()
        if not isinstance(contents, bytes):
            contents = contents.encode('ascii')
        self.data = np.atleast_2d(np.frombuffer(contents, dtype=np.uint8))
        self.empty = 'yes' if self.data.size == 0 else 'no'
