import argparse

import h5py

from sda.utils import write_header


def make_test_data(filename):

    with h5py.File(filename, 'w') as f:

        # Header
        write_header(f.attrs)

        # curly
        g = f.create_group('curly')
        g.attrs['RecordType'] = 'numeric'
        g.attrs['Empty'] = 'no'
        g.attrs['Deflate'] = 0
        g.attrs['Description'] = '3x4 array (double)'

        ds = g.create_dataset(
            'curly',
            shape=(3, 4),
            maxshape=(None, None),
            dtype='<f8',
            chunks=(3, 4),
            compression=0,  # gzip
            fillvalue=0.0,
        )
        ds.attrs['RecordType'] = 'numeric'
        ds.attrs['Empty'] = 'no'

        # curly 2
        g = f.create_group('curly 2')
        g.attrs['RecordType'] = 'numeric'
        g.attrs['Empty'] = 'no'
        g.attrs['Deflate'] = 0
        g.attrs['Description'] = 'Single version of curly'

        ds = g.create_dataset(
            'curly 2',
            shape=(3, 4),
            maxshape=(None, None),
            dtype='<f4',
            chunks=(3, 4),
            compression=0,
            fillvalue=0.0,
        )
        ds.attrs['RecordType'] = 'numeric'
        ds.attrs['Empty'] = 'no'

        # larry
        g = f.create_group('larry')
        g.attrs['RecordType'] = 'character'
        g.attrs['Empty'] = 'yes'
        g.attrs['Deflate'] = 0
        g.attrs['Description'] = 'An empty character array'

        ds = g.create_dataset(
            'larry',
            shape=(1, 1),
            maxshape=(None, None),
            dtype='<u1',
            chunks=(1, 1),
            compression=0,
            fillvalue=0,
        )
        ds.attrs['RecordType'] = 'character'
        ds.attrs['Empty'] = 'yes'

        # moe
        g = f.create_group('moe')
        g.attrs['RecordType'] = 'logical'
        g.attrs['Empty'] = 'no'
        g.attrs['Deflate'] = 0
        g.attrs['Description'] = 'A 1x1 logical array'

        ds = g.create_dataset(
            'moe',
            shape=(1, 1),
            maxshape=(None, None),
            dtype='<u1',
            chunks=(1, 1),
            compression=0,
            fillvalue=0,
        )
        ds.attrs['RecordType'] = 'logical'
        ds.attrs['Empty'] = 'no'

        # my function
        g = f.create_group('my function')
        g.attrs['RecordType'] = 'function'
        g.attrs['Empty'] = 'no'
        g.attrs['Deflate'] = 0
        g.attrs['Description'] = 'Sine function'

        ds = g.create_dataset(
            'my function',
            shape=(1, 10280),
            maxshape=(1, 10280),
            dtype='<u1',
            chunks=None,
            fillvalue=0,
        )
        ds.attrs['RecordType'] = 'function'
        ds.attrs['Command'] = 'sin'
        ds.attrs['Empty'] = 'no'


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help="The name of the file to create")

    args = parser.parse_args()
    make_test_data(args.filename)
