import numpy as np


##################
# PUBLIC METHODS #
##################

def getAllSymmetries(dbnString, dbnDim):
    # Returns a list containing all the dbnStrings who are symmetries of the input
    og = _to_sparsed_matrix(dbnString, dbnDim)

    mats = [og]
    operations = [_doXMirroring, _doYMirroring]
    if dbnDim[0] == dbnDim[1]:
        # Only include rotations if the board is square
        operations.append(_doRotate)

    for op in operations:
        newmat = []
        for m in mats:
            newmat.append(m)
            newmat.append(op(m))
        mats = newmat

    strings = [_from_sparsed_matrix(m) for m in mats]
    return strings


def getXMirroring(dbnString, dbnDim):
    # Returns the mirrored version of the dbnSting according to the x-axis
    return _convert(_doXMirroring, dbnString, dbnDim)


def getYMirroring(dbnString, dbnDim):
    # Returns the mirrored version of the dbnSting according to the y-axis
    return _convert(_doYMirroring, dbnString, dbnDim)


def getRotation(dbnString, dbnDim):
    # Returns the rotated version of the dbnSting according to the y-axis
    return _convert(_doRotate, dbnString, dbnDim)


####################
# INTERNAL METHODS #
####################

def _convert(func, dbnString, dbnDim):
    # Converts the dbnString into another by applying the conversion-function
    return _from_sparsed_matrix(func(_to_sparsed_matrix(dbnString, dbnDim)))


def _doXMirroring(sparsedMat):
    # Returns the sparsed matrix that is mirrored by the x-axis
    return np.flip(sparsedMat, 0)  # is equivalent to flipud(m).


def _doYMirroring(sparsedMat):
    # Returns the sparsed matrix that is mirrored by the y-axis
    return np.flip(sparsedMat, 1)  # is equivalent to fliplr(m).


def _doRotate(sparsedMat):
    # Returns the sparsed matrix that is rotated 90°
    return np.rot90(sparsedMat)


##################
# SPARSED MATRIX #
##################

# The sparsed matrix is a matrix representing the dbn state in matrix form as to correspond the most with what we see
# visually. It is padded with items as to facilitate easier rotations and mirrorings. In this matrix, a None at a
# position does not represent anything (node or cell interior), a 0 represents that nothing is there, a 1 represents
# an edge is placed
# 
# Example:
# 000110011101
# ┌╴ ╶┬╴ ╶┐
#     │   │
# ├╴ ╶┼───┤
# │       │
# └───┴╴ ╶┘
# Corresponds with 
# x0x0x
# 0x1x1
# x0x1x
# 1x0x1
# x1x0x
# 
# Where x is shorthand for None


def _to_sparsed_matrix(dbnString, dbnDim):
    hparts, vparts = _splitToParts(dbnString, dbnDim)
    padH = [_insertFullPadding(i) for i in hparts]
    padV = [_insertInbetweenPadding(i) for i in vparts]
    return _interleave(padH, padV)


def _untangle(sparsedMatrix):
    # Untangles the sparsed matrix into lists consisting of the horizontal and vertical edges
    hparts, vparts = [], []
    for i, sublist in enumerate(sparsedMatrix):
        if i % 2 == 0:
            hparts.append(sublist)
        else:
            vparts.append(sublist)
    return hparts, vparts


def _removePadding(lst):
    return [i for i in lst if i is not None]


def _listToString(lst):
    # flattens the list of list of numbers into a string
    return ''.join([str(i) for li in lst for i in li])


def _from_sparsed_matrix(sparsedMatrix):
    hparts, vparts = _untangle(sparsedMatrix)
    hpure, vpure = [_removePadding(i) for i in hparts], [_removePadding(i) for i in vparts]
    res = _listToString(hpure) + _listToString(vpure)
    return res


def _splitToParts(dbnString: str, dbnDim):
    # Splits the dbnString into multiple parts.
    # This function returns a tuple containing first: all the horizontal pieces, second all the vertical pieces
    # dbnDim = a tuple containing the number of num_rows and num_cols
    # example
    # 000110011101
    # ┌╴ ╶┬╴ ╶┐
    #     │   │
    # ├╴ ╶┼───┤
    # │       │
    # └───┴╴ ╶┘
    # Will return h_parts = [[0,0], [0,1], [1,0]],
    # v_parts = [[0,1,1], [1,0,1]]

    nb_rows = dbnDim[0]
    nb_cols = dbnDim[1]
    split_pos = nb_rows * (nb_cols + 1)
    hstrings = dbnString[:split_pos]
    vstrings = dbnString[split_pos:]
    h_parts = _splitEqLen(hstrings, nb_rows)  # The list consisting of all horizontal edges
    v_parts = _splitEqLen(vstrings, nb_rows + 1)  # The list consisting of all vertical edges
    h_ints = [list(map(int, i)) for i in h_parts]  # Convert the stringlist into a list of list of ints
    v_ints = [list(map(int, i)) for i in v_parts]
    return h_ints, v_ints
    # return h_parts, v_parts


def _splitEqLen(string, length):
    # Splits a string into subparts of equal length
    return [string[i:i + length] for i in range(0, len(string), length)]


def _interleave(l1, l2):
    # Interleaves all elements from the two lists into a result list
    res = []
    len1 = len(l1)
    len2 = len(l2)
    for i in range(max(len1, len2)):
        if i < len1:
            res.append(l1[i])
        if i < len2:
            res.append(l2[i])
    return res


def _insertInbetweenPadding(lst):
    # Inserts a None in between all subsequent items in the list
    res = []
    for i in lst:
        res.append(i)
        res.append(None)
    res.pop()
    return res


def _insertFullPadding(lst):
    # Inserts a None in between all subsequent items in the list and in the head and tail
    res = [None]
    for i in lst:
        res.append(i)
        res.append(None)
    return res

# Uncomment the following to see that everything works
# rots = getAllSymmetries("000110011101", (2,2))


def test():

    # ┌───┬───┬╴ ╶┐
    #     │
    # ├╴ ╶┼╴ ╶┼───┤
    #         │
    # └╴ ╶┴───┴╴ ╶┘
    dbnStr = "11000101001000010"
    dim = (3,2)
    hp, vp = _splitToParts(dbnStr, dim)
    eh, ev = [[1,1,0], [0,0,1], [0,1,0]], [[0,1,0,0], [0,0,1,0]]
    assert hp == eh
    assert ev == vp

    # 000110011101
    # ┌╴ ╶┬╴ ╶┐
    #     │   │
    # ├╴ ╶┼───┤
    # │       │
    # └───┴╴ ╶┘
    # Will return h_parts = [[0,0], [0,1], [1,0]],
    # v_parts = [[0,1,1], [1,0,1]]
    dbnStr = "000110011101"
    dim = (2,2)
    hp, vp = _splitToParts(dbnStr, dim)
    eh, ev = [[0,0], [0,1], [1,0]], [[0,1,1], [1,0,1]]
    assert hp == eh
    assert ev == vp



test()