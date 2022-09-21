import torch
import numpy as np
from constants import *


def createIs(file_path, num_classes):
    # Matrix with indices for positive literals
    Iplus = []
    # Matrix with indeces for negative literals
    Iminus = []
    with open(file_path, 'r') as f:
        for line in f:
            split_line = line.split()
            assert split_line[2] == ':-', "Instead of :- found: %s" % split_line[2]
            iplus = np.zeros(num_classes)
            iminus = np.zeros(num_classes)
            for item in split_line[3:]:
                if 'n' in item:
                    index = int(item[1:])
                    iminus[index] = 1
                else:
                    index = int(item)
                    iplus[index] = 1
            Iplus.append(iplus)
            Iminus.append(iminus)
    Iplus = np.array(Iplus)
    Iminus = np.array(Iminus)
    return Iplus, Iminus


# createMs returns two matrices: Mplus: shape [num_labels, num_constraints] --> each column corresponds to a
# constraint and it has a one if the constraint has positive head at the column number of the label of the head
# Mminus: shape[num_labels, num_constraints] --> each column corresponds to a constraint and it has a one if the
# constraint has negative head at the column number of the label of the head
def createMs(file_path, num_classes):
    Mplus, Mminus = [], []
    with open(file_path, 'r') as f:
        for line in f:
            split_line = line.split()
            assert split_line[2] == ':-'
            mplus = np.zeros(num_classes)
            mminus = np.zeros(num_classes)
            if 'n' in split_line[1]:
                # one indentified that is negative, ignore the 'n' to get the index
                index = int(split_line[1][1:])
                mminus[index] = 1
            else:
                index = int(split_line[1])
                mplus[index] = 1
            Mplus.append(mplus)
            Mminus.append(mminus)
    Mplus = np.array(Mplus).transpose()
    Mminus = np.array(Mminus).transpose()

    return Mplus, Mminus


if __name__ == "__main__":
    Iplus, Iminus = createIs(CONSTRAINTS_PATH, NUM_LABELS)
    Mplus, Mminus = createMs(CONSTRAINTS_PATH, NUM_LABELS)

    print("I sizes", Iplus.shape, Iminus.shape)
    print("M sizes", Mplus.shape, Mminus.shape)
