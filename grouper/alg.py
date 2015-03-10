from itertools import chain, combinations
import copy
from random import shuffle
from numpy import array, argmin
import numpy.random
from pprint import pprint
#import pyprind
#import psutil

# typical usage:
"""
to get the best current partition (not accounting for scope matching), try:
p = quickpart(n)
    where "n" is the number of groups to seperate the students into


remember if every student group is visiting every vendor: don't worry about vendors,
only worry about picking best student partition


"""


# SETUP

pairscores = {}
scopescores = {}

names = {
0:"mihai",
1:"djenet",
2:"russell",
3:"lily",
4:"tracy",
5:"saurabh",
6:"jen-yi",
7:"sarah",
8:"eileen",
9:"john",
10:"viktor",
11:"ram",
12:"carmen",
13:"samuel",
14:"shiaulou",
15:"yuxiang"
}

scale=1

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

def rand_partition(numgroups, numstudents=config.numstudents):
    lst = range(numstudents)
    shuffle(lst)
    division = len(lst) / float(numgroups)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(numgroups) ]


#############################################
#         Student Grouping functions        #
#############################################

# a combination is a pairing of two students
# a group is a set of 2 or more students (also called a subset here, I think?)
# a partition is a grouping of the all students into subsets (groups)

def groupscore(grouplist):
    """Scores a group accoring to the global pairscores dict

    accepts: a python list
            (representing the student ids of a putative group)
    returns: a number
            (lower scores mean the grouping is more novel)
    """
    global pairscores
    l = len(grouplist)
    gs = 0
    for i in combinations(grouplist,2):
        gs += pairscores[i[0]][i[1]]         # requires pairs to be a symetric array... as above
    gs = gs/float(((l**2/2) - (l/2)))       # normalize the score to the size of the group
    return gs

def partitionscore(subsetlist):
    """Scores a partition accoring to the global pairscores dict,
    by iterating the groupscore function for every subset (group) in the partition

    accepts: a python list of lists
            (a partition representing the grouping of all students into subsets)
    returns: a number
            (lower scores mean the partition is more novel)
    """
    ss = 0
    for i in subsetlist:
        ss += groupscore(i)                 # requires pairs to be a symetric array... as above
    return ss

def findbestset(setlist):
    """goes through a list of partitions (such as that returned by algorith_u)
    and scores all of them, looking for the best options

    accepts: a python list of lists of lists
            (representing )
    returns: a number
            (lower scores mean the partition is more novel)
    """
    s = float("inf")            # set high benchmark
    ind = []
    for part in setlist:            # for each partition in the list
        ss = partitionscore(part)   # get score for that partition
        if ss > s:              # if it's worse than the benchmark
            continue            # go to the next one
        elif ss < s:            # if it's better than the benchmark
            ind=[part]          # store that partition (for return)
            s = ss              # make it the new benchmark
            continue            # go on to the next
        elif ss == s:           # if it's the SAME as the benchmark
            ind.append(part)        # add it to the list of "best" partitions (allows multiple "bests")
    return ind

def shuffleandtest(lst,n,iter):
    """
    this accepts an input list (of student IDs),
    randomly partitions it into n subsets
    and then scores that partition...

    this repeats iter number of times and returns the partition
    that scored the best
    """
    z = float("inf")                    #set benchmark super high
    for i in range(iter):
        shuffle(lst)                    #shuffle the input list
        g=partition(lst,n)              #make simplest partition of list into n groups
        s=partitionscore(g)             #score the partition
        if s<z:                         #if the score is better than the benchmark
            z = s                       #make it the new benchmark
            best = g                    #and store that partition of groups...
    #print "best score: %f" % z
    return best                         #return the best partition of groups gound


def greedy(lst,iter=1):
    l = len(lst)            #count number of groups
    newsets = copy.deepcopy(lst)
    alt=[]
    for i in xrange(iter):
        besti = argmin(array([groupscore(i) for i in newsets])) #find index of group with best score
        b = newsets[besti]                      #store best group
        del newsets[besti]                      #remove the best group from the bunch
        newsets = [i for sub in newsets for i in sub]               #flatten the bunch
        f = findbestset(algorithm_u(newsets,l-1))   #find the best rearrangement of the bottom groups
        newsets = f[0]                          #pick the first option (if there are more than 1)
        newsets.append(b)                       #add back the good group
        print "new score: %f" % partitionscore(newsets)
        if len(f)>1:
            print "there were more options"     #alert if there were other paths not persued
            alt = f[1:]     #store the alternatives
            [i.append(b) for i in alt]
    return newsets                  #return the modified list of groups


def greedy2(lst,cap=20):
    """
    accepts a partition and returns a partition representing the best rearrangement
    """
    l = len(lst)            #count number of groups
    newtrunk = copy.deepcopy(lst)
    branches = []
    storedbranches = []
    tips = []
    f = 1
    while True:
        # reserve a copy of newtrunk (the current partition)
        oldtrunk = copy.deepcopy(newtrunk)
        # besti finds the index of the best scored group from the partition in newtrunk
        besti = argmin(array([groupscore(i) for i in newtrunk]))
        #store best group
        b = newtrunk[besti]
        #remove the best group from the current partition
        del newtrunk[besti]
        #flatten the partition into a single python list
        newtrunk = [i for sub in newtrunk for i in sub]
        #find the best rearrangement of that list (representing the students in the lowest scored groups)
        f = findbestset(algorithm_u(newtrunk,l-1))
        newtrunk = f[0]
        newtrunk.append(b)              # store the first "best" rearrangement
        if len(f)>1:                            # if there are multiple "best" arrangements
            print "%s branches found" % str(len(f)-1)
            for branch in f[1:]:
                q = branch
                q.append(b)
                branches.append(q)  # store the others in branches
                storedbranches.append(q)
        print newtrunk
        if equal_partitions(newtrunk,oldtrunk):
            tips.append(newtrunk)
            print "tip reached.  score: %f" % partitionscore(newtrunk)
            print "-------\n"
            break
        print "score: %f" % partitionscore(newtrunk)

    if len(branches)>0:
        branchcount=1
        while len(branches)>0 and branchcount<cap:
            print "exploring branch %d..." % branchcount
            newbranch = branches.pop()
            print newbranch
            print "starting score: %f" % partitionscore(newbranch)

            while True:
                oldbranch = copy.deepcopy(newbranch)
                besti = argmin(array([groupscore(i) for i in newbranch]))
                b = newbranch[besti]                        #store best group
                del newbranch[besti]                        #remove the best group from the bunch
                newbranch = [i for sub in newbranch for i in sub]               #flatten the bunch
                f = findbestset(algorithm_u(newbranch,l-1))     #find the best rearrangement of the bottom groups
                newbranch = f[0]
                newbranch.append(b)             # store the first "best" rearrangement
                if len(f)>1:                            # if there are multiple "best" arrangements
                    print "%s branches found" % str(len(f)-1)
                    for branch in f[1:]:
                        q = branch
                        q.append(b)
                        if not dupebranch(q,storedbranches):
                            branches.append(q)  # store in branches
                            storedbranches.append(q)
                print "%d branches left" % len(branches)
                print newbranch
                if equal_partitions(newbranch,oldbranch):
                    if not dupebranch(newbranch,tips):
                        print "novel tip reached.  score: %f" % partitionscore(newbranch)
                        tips.append(newbranch)
                        print "-------\n"
                        break
                    else:
                        print "stale tip reached.  score: %f" % partitionscore(newbranch)
                        print "-------\n"
                        break
                print "score: %f" % partitionscore(newbranch)
            branchcount+=1
            print "%d branches left" % len(branches)
    tips.reverse()
    return tips

def quickpart(n):
    """
    divide 16 students into current best partition of n groups
    given the current pairscores
    """
    p = greedy2(shuffleandtest(range(16),n,30000))
    print "\n"
    pretty_names(p[0])
    print "\npartition score: %f" % partitionscore(p[0])
    return p


#############################################
#         Vendor Matching functions         #
#############################################

def scorescopes_group(group):   # returns dict with sum of individual scope scores for all vendors for a given group
    global scopescores
    result={}
    for vendor in scopescores:
        result[vendor] = 0
    for person in group:
        for vendor in scopescores:
            result[vendor] += scopescores[vendor][person]
    return result

def scorescope(part):   # returns a list of lists scoring vendor appropriateness for each group in the partition
    return [sorted(i,key=lambda x: x[1]) for i in [[t for t in n.iteritems()] for n in [scorescopes_group(i) for i in part]]]

def poss_rotations(n,vendors):
    """
    return a list of lists of lists of possible rotations

    Inputs:
    n = number of rotations in the lab (that each student goes to)
    vendors = list of possible vendor stations

    Returns:
    list of list of lists
        each item in top list is a list of lists of rotations
        that any given student might go through
    """
    return [rotate_vendors(i,vendors) for i in unique_rotations(n,vendors)]

def unique_rotations(n, vendors):
    """
    accepts a list of vendors and a number of rotations (n)
    and return the possible combinations of station rotations

    Returns a list of tuples, where each tuple is a possible group of station rotations
    for example:
        vendors = ['olympus', 'leica', 'andor', 'api']
        n = 2

        returns:
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    this should be used to "rotate" the vendor list accordingly...
    the first tuple in the returned list (0,1) would represent the following rotations
    ['olympus', 'leica', 'andor', 'api'] and ['api', 'olympus', 'leica', 'andor']
    which, when zipped, gives the rotations that the 4 student groups would see:
    [('olympus', 'api'),('leica', 'olympus'),('andor', 'leica'),('api', 'andor')]

    """
    numvendors = len(vendors)
    return list(combinations(range(numvendors), n))

def rotate_vendors(rots,vendors):
    """ accepts a single tuple of rotations, and a list of vendors, and returns the zipped rotations

    """
    numrots = len(rots)

    def two():
        return zip(rotate_list(vendors,rots[0]),rotate_list(vendors,rots[1]))

    def three():
        return zip(rotate_list(vendors,rots[0]),rotate_list(vendors,rots[1]),rotate_list(vendors,rots[2]))

    def four():
        return zip(rotate_list(vendors,rots[0]),rotate_list(vendors,rots[1]),
            rotate_list(vendors,rots[2]),rotate_list(vendors,rots[3]))

    def five():
        return zip(rotate_list(vendors,rots[0]),rotate_list(vendors,rots[1]),
            rotate_list(vendors,rots[2]),rotate_list(vendors,rots[3]),rotate_list(vendors,rots[4]))

    def six():
        return zip(rotate_list(vendors,rots[0]),rotate_list(vendors,rots[1]),
            rotate_list(vendors,rots[2]),rotate_list(vendors,rots[3]),rotate_list(vendors,rots[4]),rotate_list(vendors,rots[5]))

    options = {
        2 : two,
        3 : three,
        4 : four,
        5 : five,
        5 : six,
    }

    return options[numrots]()




def rotate_list(lst,n):
    return lst[-n:] + lst[:-n]




def group_scope_match(part,rots):
    """
    score canditate rotations (rots) for a given student partition

    Input:
    part = list of lists (student partition)
    rots = list of lists (canditate rotations)

    Returns:
    list of scores for each group/rotation in the inputs
    """
    match=[]
    for i in xrange(len(part)):
        tot=0
        for n in rots[i]:
            tot += scorescopes_group(part[i])[n]
        match.append(tot)
    return match

def scoreallrots(part,rotslist):
    return [sum(group_scope_match(part,rot)) for rot in rotslist]

def find_best_rots(part,vendors,n):
    """
    find best vendor matches given a single partition
    or list of partitions such as that returned by greedy(2)
    part is a partition
    vendors is a list of vendors
    n is the number of stations
    """
    r = poss_rotations(n, vendors)
    if any(isinstance(el, list) for el in part[0]):
        for l in part:
            find_best_rots(l,vendors,n)
            #best = r[argmin(scoreallrots(l,r))]
            #s = group_scope_match(l,best)
            #for i in range(len(l)):
            #    print "%s: " % str(", ".join(best[i]))
            #    pretty_names(l[i:i+1])
            #    print "score: %s\n" % str(s[i])
            #print "vendor score: %d" % sum(s)
            #print "groups score: %f" % partitionscore(l)
            #combscore = sum(s)*partitionscore(l)
            #print "comboscore %f" % combscore
            #print "----------------------"
    else:
        best = r[argmin(scoreallrots(part,r))]
        s = group_scope_match(part,best)
        for i in range(len(part)):
            print "%s: " % str(", ".join(best[i]))
            pretty_names(part[i:i+1])
            print "score: %s\n" % str(s[i])
        print "vendor score: %d" % sum(s)
        print "groups score: %f" % partitionscore(part)
        combscore = sum(s)*partitionscore(part)
        print "comboscore %f" % combscore
        print "----------------------"

def shuffle_match_scopes(vendors,n,iter=1000, report=100):
    """
    generate random partition according to number of vendor stations


    inputs:
    vendors =
    n =
    *iter = number of times to randomly iterate
    *report = iteration report frequency

    returns:
    tuple (bestparts, bestvends, bestcombos) where:
    bestparts = list of partitions scoring well
    bestvends =
    bestcombos =
    """
    numgroups = len(vendors)
    bestcombo = float("inf")
    bestcombos = []
    bestpart = float("inf")
    bestparts = []
    bestvend = float("inf")
    bestvends = []

    r = poss_rotations(n, vendors)

    for i in range(iter):
        #make random partition and score for student matching
        part = shuffleandtest(range(16),n,1000)
        #part = partition(numpy.random.permutation(16).tolist(),numgroups)
        partscore = partitionscore(part)

        #score partition for scope matching
        best = r[argmin(scoreallrots(part,r))] # which of the possible rotations is best for this random partition?
        s = group_scope_match(part,best) # s is the per-group scope matching score
        vendscore = sum(s)  # vendscore is the sum group-scope match score

        # calculate combined score
        combscore = vendscore * partscore

        if combscore < bestcombo:
            print " "
            print "new best COMBO:"
            print "partition: %s " % (part)
            print "vendor match: %s " % (best)
            print "part * vendor = combo score: %f * %f = % f" % (partscore,vendscore,combscore)
            print "----------------------"
            bestcombo = combscore
            bestcombos.append([part,best])
        if partscore < bestpart:
            print " "
            print "new best PARTITION: %s " % (part)
            print "score: %f " % (partscore)
            print "----------------------"
            bestpart = partscore
            bestparts.append(part)
        if vendscore < bestvend:
            print " "
            print "new best VENDOR: %s " % (best)
            print "score: %s = %f " % (s,vendscore)
            print "partition (%f): %s " % (partscore,part)
            print "----------------------"
            bestvend = vendscore
            bestvends.append(best)
        if i%report==0:
            print "########Round %s, best so far: %f #########" % (i,bestcombo)
            #print "partition: %s " % (bestcombos[-1][0])
            #print "score: %s " % (partitionscore(bestcombos[-1][0]))
            #print "vendors: %s " % (bestcombos[-1][1])
            #print "score: %f " % (bestvend)
            #print "----------------------"

    print "best partition score: %f" % bestpart
    print "best vendor score: %d" % bestvend
    print "best combo score: %f, (%f/%d)" % (bestcombo, partitionscore(bestcombos[-1][0]), sum(group_scope_match(bestcombos[-1][0],bestcombos[-1][1])))
    return (bestparts, bestvends, bestcombos)

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


#############################################
#             Pair Score functions          #
#############################################

#pairscores
"""a dictionary that holds the score for any one student
with any other student.  each key in the dictionary is
a numeric code for a single student (decoded by the
"names" dictionary).  and each value is an ordered list
containing the score for that student with each of the
other students.  lastly, the key pairscores['i'] is a
counter for how many rounds have occured (to be used for scaling)
"""

def reset_pairscores():
    """resets the pairscores variable to 0 for every student combo."""
    global pairscores
    pairscores={}
    for i in range(16): pairscores[i]=[0]*16
    pairscores['i']=0

def random_pairscores():
    """generates random pairscores data... for testing"""
    global pairscores
    pairscores={}
    for i in range(16):
        pairscores[i]=numpy.random.randint(5, size=16).tolist()
        pairscores[i][i]=0
    pairscores['i']=0

def update_pairscores(groups,scale=scale):
    """updates the pairscores dict given the partition (groups) provided"""
    global pairscores
    pairscores['i']=pairscores['i']+1
    for i in groups:
        for n in combinations(i,2):
            pairscores[n[0]][n[1]] += scale*pairscores['i']
            pairscores[n[1]][n[0]] += scale*pairscores['i']

random_pairscores()



#############################################
#             Scope Score functions         #
#############################################


#scopescores
"""
a dictionary where each key is a vendor name,
and each value is a python list representing how frequently each student has been at that vendor
"""


def reset_scopescores():
    global scopescores
    scopescores={}
    v = ['nikon','olympus','zeiss','leica','andor','api']
    for i in v: scopescores[i]=[0]*16

def random_scopescores():
    global scopescores
    scopescores={}
    v = ['nikon','olympus','zeiss','leica','andor','api']
    for i in v: scopescores[i]=numpy.random.randint(5, size=16).tolist()

def update_scopescores(groups,scsc,scale=scale):        #updates the scope scores array given the groups provided
    pass

random_scopescores()
