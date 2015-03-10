#############################################
#             Helper functions              #
#############################################

def nodupes(h):
    for i in h:
        if len(i)!=len(set(i)):
            return None
    return h

def equal_partitions(a,b):      # check whether two partitions represent the same subgroups groups
    return set(frozenset(i) for i in a) == set(frozenset(i) for i in b)


def dupebranch(test,stored):
    for i in stored:
        if equal_partitions(i,test):
            return True
    return False

def pretty_names(parts):
    for n in parts:
        print ', '.join(names[i] for i in n)