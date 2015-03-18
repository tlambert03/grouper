from grouper import config
from itertools import chain, combinations
from random import shuffle

#############################################
#         List partitioning functions       #
#############################################


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


def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def powergroups(students, groups):
    s = range(students)
    minsize=students/groups
    even = 1 if students%groups else 0
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1) if r==minsize or r==(minsize+even))


# [ e for e in powerset(range(6)) if len(e)==3 or len(e)==4 ]
# returns all of the possible groups with, for instance, 3 or 4 members...
# to be used in dlx algorithm

