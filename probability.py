def is_probability(P):
    return sum(P) == 1.0 and all(p >= 0.0 for p in P) and all(p <= 1.0 for p in P)


def uniform(k):
    return [1.0 / k for _ in range(k)]


def marginalize(P, axis=1):
    if axis == 1:
        # marginalize P(x, y) over y to P(x)
        return [sum(P[i][j] for j in range(len(P[0]))) for i in range(len(P))]
    elif axis == 0:
        # marginalize P(x, y)) over x to P(y)
        return [sum(P[i][j] for i in range(len(P))) for j in range(len(P[0]))]
    raise Exception('Does not support axis', axis)


def condition(P):
    # condition P(x, y) on y to P(x | y)
    P_x = marginalize(P, axis=1)
    return [[P[i][j] / P_x[i] for j in range(len(P[0]))] for i in range(len(P))]


def is_independent(P):
    P_x = marginalize(P, axis=1)
    P_y = marginalize(P, axis=0)
    return all(P[i][j] == P_x[i] * P_y[j] for i in range(len(P_x)) for j in range(len(P_y)))


def expectation(P, f):
    return sum(P[x] * f(x) for x in range(len(P)))
