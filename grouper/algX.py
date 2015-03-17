from itertools import chain, combinations, permutations
from math import factorial

def algX(X, Y, solution=[]):
    if not X:
        yield list(solution)
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution.append(r)
            cols = select(X, Y, r)
            for s in algX(X, Y, solution):
                yield s
            deselect(X, Y, r, cols)
            solution.pop()

def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols

def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)


def convert(x,y):
    """
    converts a set and a dict into the required "X" format for algX
    for instance:

    X = {1, 2, 3, 4, 5, 6, 7}

    Y = {
    'A': [1, 4, 7],
    'B': [1, 4],
    'C': [4, 5, 7],
    'D': [3, 5, 6],
    'E': [2, 3, 6, 7],
    'F': [2, 7]}

    outputs:
    {
    1: {'A', 'B'},
    2: {'E', 'F'},
    3: {'D', 'E'},
    4: {'A', 'B', 'C'},
    5: {'C', 'D'},
    6: {'D', 'E'},
    7: {'A', 'C', 'E', 'F'}}

    """

    x = {j: set() for j in x}
    for i in y:
        for j in y[i]:
            x[j].add(i)
    return x

def powergroups(studentlist, groups, dict=1):
    """ outputs all possible ~same-sized subsets given a number of students and number of groups

    if dict=1 outputs a dictionary in the format required for the "y" input in the convert function
    """
    s = studentlist
    students=len(s)
    minsize=students/groups
    even = 1 if students%groups else 0
    c = chain.from_iterable(combinations(s, r) for r in range(len(s)+1) if r==minsize or r==(minsize+even))
    if dict:
        o = {}
        i=0
        for g in c:
            o[i] = g
            i+=1
        return o
    else:
        return c


def allgroups(studentlist, groups, d=1):
    students = len(studentlist)
    X = set(studentlist)
    Y = powergroups(studentlist, groups)
    XX = convert(X, Y)
    result = algX(XX,Y)
    if d:
        return [[Y[bbb] for bbb in bb] for bb in result]
    else:
        return result


def all_possible_schedules(stations, rotations,expanded=1):
    X = set(stations)
    perms = permutations(stations,rotations)
    Y,i={},0
    for p in perms:
        Y[i] = p
        i+=1
    XX = convert(X, Y)
    solution = algX(XX,Y)
    if expanded:
        return [[Y[stat] for stat in sol ]for sol in solution]
    else:
        return solution

def dictify(l,i=1):
    Y={}
    for p in l:
        Y[i] = list(p)
        i+=1
    return Y

Y = {0:('A', 'B'),
1:('A', 'C'),
2:('A', 'D'),
3:('B', 'A'),
4:('B', 'C'),
5:('B', 'D'),
6:('C', 'A'),
7:('C', 'B'),
8:('C', 'D'),
9:('D', 'A'),
10:('D', 'B'),
11:('D', 'C') }