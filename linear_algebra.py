import math


def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def is_scalar(x):
    return type(x) in [int, float]


def is_vector(x):
    return type(x) is list and \
           all(is_scalar(x[i]) for i in range(len(x)))


def is_matrix(x):
    return type(x) is list and  \
           all(type(x[i]) is list for i in range(len(x))) and \
           all(len(x[i]) == len(x[0]) for i in range(len(x)))


def add(A, B):
    if is_matrix(A) and is_matrix(B):
        row = len(A)
        column = len(A[0])
        return [[A[i][j] + B[j][i] for j in range(column)] for i in range(row)]
    elif is_vector(A) and is_vector(B):
        return [A[i] + B[i] for i in range(len(A))]


def minus(A):
    if is_matrix(A):
        return [[-A[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    elif is_vector(A):
        return [-A[i] for i in range(len(A))]


def multiply(A, B):
    if is_scalar(A) and is_vector(B):
        return [A * B[i] for i in range(len(B))]
    elif is_vector(A) and is_vector(B):
        # Vector product, not element-wise product
        return sum([A[i] * B[i] for i in range(len(A))])
    elif is_scalar(A) and is_matrix(B):
        I = len(B)
        J = len(B[0])
        return [[A*B[i][j] for j in range(J)] for i in range(I)]
    elif is_vector(A) and is_matrix(B):
        I = len(A)
        J = len(B[0])
        return [sum(A[i] * B[i][j] for i in range(I)) for j in range(J)]
    elif is_matrix(A) and is_vector(B):
        I = len(A)
        J = len(A[0])
        return [sum(A[i][j] * B[j] for j in range(J)) for i in range(I)]
    elif is_matrix(A) and is_matrix(B):
        I = len(A)
        J = len(B[0])
        K = len(A[0])
        return [[sum(A[i][k] * B[k][j] for k in range(K)) for j in range(J)] for i in range(I)]
    raise Exception('No match!')


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


def normalize(x):
    return [x[i]/norm(x) for i in range(len(x))]


def eigen_decomposition(A):
    # 2x2 matrix only
    l1 = (A[0][0] + A[1][1] + math.sqrt((A[0][0] - A[1][1])**2. + 4. * A[0][1] * A[1][0])) / 2.
    l2 = (A[0][0] + A[1][1] - math.sqrt((A[0][0] - A[1][1])**2. + 4. * A[0][1] * A[1][0])) / 2.
    v1 = [A[0][1], l1 - A[0][0]]
    v2 = [A[0][1], l2 - A[0][0]]
    V = transpose([normalize(v1), normalize(v2)])
    return [[l1, l2], V]


def principal_component_analysis(X):
    # X: n * 2 matrix (n: number of data points)
    return eigen_decomposition(multiply(transpose(X), X))
