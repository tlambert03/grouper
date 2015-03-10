#############################################
#         List partitioning functions       #
#############################################

# from http://codereview.stackexchange.com/questions/1526/python-finding-all-k-subset-partitions
def algorithm_u(ns, m):
    """
    where ns is the set of objects (i.e. a list of student ids)
    and m is the number of partitions
    """
    def visit(n, a):
        ps = [[] for i in xrange(m)]
        for j in xrange(n):
            ps[a[j + 1]].append(ns[j])

        for i in ps:
            if (len(i)<(n/m) or len(i)>=2*(n/m)): return None        # limits sets to relatively equal sizes
        return ps


    def f(mu, nu, sigma, n, a):         # where mu is the number of partitions, nu and n are number of total objects in set
        if mu == 2:
            if visit(n, a): yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            if visit(n, a): yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if visit(n, a): yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                visit(n, a)
                a[nu] = a[nu] + 1
            visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

                                # m is the number of partitions
    n = len(ns)                 # n = number of objects in the total set
    a = [0] * (n + 1)           # a is an empty list with n + 1 items
    for j in xrange(1, m + 1):  # for the number of partitions
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)

def generate_groups(lst, n):
    """
    works better than alg_u when splitting into TWO groups
    (something wrong with algorithm_u)
    """
    if not lst:
        yield []
    else:
        for group in (((lst[0],) + xs) for xs in combinations(lst[1:], n-1)):
            for groups in generate_groups([x for x in lst if x not in group], n):
                yield [group] + groups

def partitions(set_):
    """all possible partitions?"""
    if not set_:
        yield []
        return
    for i in xrange(2**len(set_)/2):
        parts = [set(), set()]
        for item in set_:
            parts[i&1].add(item)
            i >>= 1
        for b in partitions(parts[1]):
            yield [parts[0]]+b

def partition(lst, n):
    """creates a simple partition of lst into n groups"""
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]
