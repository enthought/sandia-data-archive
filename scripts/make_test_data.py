import argparse

import numpy as np
from scipy import sparse

from sdafile import SDAFile


EXAMPLE_A1 = np.zeros(5, dtype=np.float64)

EXAMPLE_A2 = np.empty((4, 3), dtype=np.complex128)
EXAMPLE_A2.real = 0
EXAMPLE_A2.imag = 1

EXAMPLE_A3 = sparse.eye(5).tocoo()

EXAMPLE_A4 = np.nan


def make_example_data(filename):

    sda_file = SDAFile(filename, 'w')

    sda_file.insert("example A1", EXAMPLE_A1, "5x1 zeros")

    sda_file.insert("example A2", EXAMPLE_A2, "4x3 imaginary numbers")

    sda_file.insert("example A3", EXAMPLE_A3, "5x5 sparse matrix")

    sda_file.insert("example A4", np.nan, "Empty array")

    sda_file.insert("example B", True, "Logical scalar")

    data = np.array(list('Here is some text'), 'S1').reshape(-1, 1)
    sda_file.insert("example C", data, "Some text")

    desc = "Cell array combining examples A1 and A2"
    sda_file.insert("example E", [EXAMPLE_A1, EXAMPLE_A2], desc)

    desc = "Structure combining examples A1 and A2"
    a1a2 = {"A1": EXAMPLE_A1, "A2": EXAMPLE_A2}
    sda_file.insert("example F", a1a2, desc)

    desc = "Structure array combining examples A1 and A2 (repeated)"
    cell = np.array([a1a2, a1a2], dtype=object).reshape(2, 1)
    sda_file.insert("example G", cell, desc)

    desc = "Cell array of structures combining examples A1-A4"
    a3a4 = {"A3": EXAMPLE_A3, "A4": EXAMPLE_A4}
    cell = np.array([a1a2, a3a4], dtype=object).reshape(2, 1)
    sda_file.insert("example H", cell, desc)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        help="The name of the file to create",
        nargs="?",
        default="SDAreference_py.sda",
    )

    args = parser.parse_args()
    make_example_data(args.filename)
