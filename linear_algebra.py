
def transpose(A):
    row = len(A)
    column = len(A[0])
    return [[A[j][i] for j in range(column)] for i in range(row)]


def add(A, B):
    row = len(A)
    column = len(A[0])
    return [[A[i][j] + B[j][i] for j in range(column)] for i in range(row)]


def multiply(A, B):
    I = len(A)
    J = len(B[0])
    K = len(A[0])
    return [[sum(A[i][k] * B[k][j] for k in range(K)) for j in range(J)] for i in range(I)]


def identity(size):
    return [[1 if i == j else 0 for j in range(size)] for i in range(size)]

