import math


def transpose(A):
    row = len(A)
    column = len(A[0])
    return [[A[j][i] for j in range(column)] for i in range(row)]


def add(A, B):
    row = len(A)
    column = len(A[0])
    return [[A[i][j] + B[j][i] for j in range(column)] for i in range(row)]


def multiply(A, B):
    if type(A) in [int, float]:
        row = len(B)
        column = len(B[0])
        return [[A*B[i][j] for j in range(column)] for i in range(row)]
    elif type(A) is list:
        if type(A[0]) in [int, float]:
            I = len(A)
            J = len(B[0])
            return [sum(A[i] * B[i][j] for i in range(I)) for j in range(J)]
        elif type(A[0]) is list:
            I = len(A)
            J = len(B[0])
            K = len(A[0])
            return [[sum(A[i][k] * B[k][j] for k in range(K)) for j in range(J)] for i in range(I)]


def identity(size):
    return [[1 if i == j else 0 for j in range(size)] for i in range(size)]


def inverse(A):
    # 2x2 matrix only
    determinant = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    return multiply(1./determinant, [[A[1][1], -A[0][1]], [-A[1][0], A[0][0]]])


def norm(x):
    return math.sqrt(sum(x[i] ** 2 for i in range(len(x))))


def diagonal(x):
    return [[x[i] if i == j else 0 for j in range(len(x))] for i in range(len(x))]


def eigen_composition(V, l):
    return multiply(multiply(V, diagonal(l)), inverse(V))
