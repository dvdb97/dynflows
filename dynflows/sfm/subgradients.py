

def sg1(f, E, Y, j):
    if j in Y:
        return f(E) - f(E - set([j]))
    else:
        return f(Y | set([j])) - f(Y)
    
def sg2(f, E, Y, j):
    if j in Y:
        return f(Y) - f(Y - set([j]))
    else:
        return f(set([j]))

def sg3(f, E, Y, j):
    if j in Y:
        return f(E) - f(E - {j})
    else:
        return f({j})