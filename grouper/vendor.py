from grouper import config, scoring
from itertools import combinations
from numpy import argmin

#############################################
#         Vendor Matching functions         #
#############################################

def scorescopes_group(group, score):   # returns dict with sum of individual scope scores for all vendors for a given group
    result={}
    for vendor in score.stationscores:
        result[vendor] = 0
    for person in group:
        for vendor in score.stationscores:
            result[vendor] += score.stationscores[vendor][person]
    return result

def scorescope(part,score):   # returns a list of lists scoring vendor appropriateness for each group in the partition
    return [sorted(i,key=lambda x: x[1]) for i in [[t for t in n.iteritems()] for n in [scorescopes_group(i,score) for i in part]]]

def group_scope_match(part,rots,score):
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
            tot += scorescopes_group(part[i],score)[n]
        match.append(tot)
    return match

def scoreallrots(part,rotslist, score):
    return [sum(group_scope_match(part,rot,score)) for rot in rotslist]


########################


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

########################


def find_best_rots(part,vendors,n,score):
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
            find_best_rots(l,vendors,n,score)
            #best = r[argmin(scoreallrots(l,r))]
            #s = group_scope_match(l,best)
            #for i in range(len(l)):
            #    print "%s: " % str(", ".join(best[i]))
            #    pretty_names(l[i:i+1])
            #    print "score: %s\n" % str(s[i])
            #print "vendor score: %d" % sum(s)
            #print "groups score: %f" % scoring.partscore_pairs(l,score)
            #combscore = sum(s)*scoring.partscore_pairs(l,score)
            #print "comboscore %f" % combscore
            #print "----------------------"
    else:
        best = r[argmin(scoreallrots(part,r,score))]
        s = group_scope_match(part,best,score)
        for i in range(len(part)):
            print "%s: " % str(", ".join(best[i]))
            pretty_names(part[i:i+1])
            print "score: %s\n" % str(s[i])
        print "vendor score: %d" % sum(s)
        print "groups score: %f" % scoring.partscore_pairs(part,score)
        combscore = sum(s)*scoring.partscore_pairs(part,score)
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
        partscore = scoring.partscore_pairs(part,score)

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
            #print "score: %s " % (scoring.partscore_pairs(bestcombos[-1][0],score))
            #print "vendors: %s " % (bestcombos[-1][1])
            #print "score: %f " % (bestvend)
            #print "----------------------"

    print "best partition score: %f" % bestpart
    print "best vendor score: %d" % bestvend
    print "best combo score: %f, (%f/%d)" % (bestcombo, scoring.partscore_pairs(bestcombos[-1][0],score), sum(group_scope_match(bestcombos[-1][0],bestcombos[-1][1])))
    return (bestparts, bestvends, bestcombos)


def pretty_names(parts):
    for n in parts:
        print ', '.join(config.names[i] for i in n)