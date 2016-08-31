def is_probability(P):
    return sum(P) == 1.0


def marginalize(P):
    # marginalize P(i, j) over j to P(i)
    return [sum(P[i][j] for j in range(len(P[0]))) for i in range(len(P))]


def condition(P):
    # condition P(i, j) on j to P(i | j)
    return [[P[i][j] / sum(P[i][k] for k in range(len(P[0]))) for j in range(len(P[0]))] for i in range(len(P))]
