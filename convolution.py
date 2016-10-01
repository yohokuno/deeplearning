def convolution1d(I, K):
    return [sum(I[i+m] * K[m] for m in range(len(K))) for i in range(len(I)-len(K)+1)]


def convolution1d_jacobian(I, K):
    return [[I[i+m] for m in range(len(K))] for i in range(len(I)-len(K)+1)]


def convolution2d(I, K):
    return [[sum(I[i+m][j+n] * K[m][n] for m in range(len(K)) for n in range(len(K[0]))) for j in range(len(I[0])-len(K[0])+1)] for i in range(len(I)-len(K)+1)]


def max_pooling1d(I, width, stride=1):
    return [max(I[i + w] for w in range(width)) for i in range(0, len(I) - width + 1, stride)]


def max_pooling2d(I, width, stride=1):
    return [[max(I[i + m][j + n] for m in range(width) for n in range(width)) for j in range(0, len(I[0]) - width + 1, stride)] for i in range(0, len(I) - width + 1, stride)]


def max_pooling1d_fix(I, size):
    width = int(len(I) / size)
    return max_pooling1d(I, width, width)


def max_pooling2d_fix(I, size):
    width = int(len(I) / size)
    return max_pooling2d(I, width, width)
