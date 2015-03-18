from itertools import combinations

def groupscore_pairs(group,score,normed=1):
    """Scores a group accoring to the provided score object

    accepts: a python list and an object
            (representing the student ids of a putative group)
    returns: a number
            (lower scores mean the grouping is more novel)
    """
    groupscore = 0
    for c in combinations(group,2):
        groupscore += score.pairscores[c[0]][c[1]]
    if normed==1:
        # normalize the score to the size of the group
        groupsize = len(group) # number of students in the group
        groupscore = groupscore/float(((groupsize**2/2) - (groupsize/2)))
    return groupscore

def partscore_pairs(part,score):
    """Scores the partition for student groupings accoring to the provided score class,
    by performing the groupscore function for every group in the partition

    returns: a number (lower scores mean the partition is more novel)
    """
    ss = 0
    for group in part:
        ss += groupscore_pairs(group,score)  # requires pairs to be a symetric array... as above
    return ss


def groupscore_stations_dict(group, stations, score):
    """returns a dict with the sum score for the given group at any given station
    """
    result={}
    for station in stations:
        result[station] = 0
    for person in group:
        for station in stations:
            result[station] += score.stationscores[station][person]
    return result

def groupscore_stations(group, stations, score):
    """
    How appropriate is the grouplist for the stationlists?

    scores a group against a list of stations for vendor matchings according to the provided score object
    by summing the vendor score for each person in the group for each station in the list of stations
    """
    gs = 0
    for person in group:
        for station in stations:
            gs += score.stationscores[station][person]
    return gs

def partscore_stations(part, stations, score):
    """returns a list of lists scoring vendor appropriateness
    for each group in the partition.
    Sorted with the best scoring station 1st
    """
    return [sorted(i,key=lambda x: x[1]) for i in [[t for t in n.iteritems()] for n in [groupscore_stations(i, stations, score) for i in part]]]



def scoreSolution(solution, score, normed=1):
    """
    takes a Solution object and scores it against a Score object
    returns a list of tuples representing the (group, vendor) score for each group in the partition
    """
    numgroups=len(solution.part)
    ss = []
    for group in range(numgroups):
        g = groupscore_pairs(solution.part[group], score, normed)
        v = groupscore_stations(solution.part[group], solution.schedule[group], score)
        ss.append((g,v))
    return ss


def rankpartitions(setlist, score):
    """goes through a list of partitions and scores all of them,
    returns an ordered list of the best ones...

    accepts: a python list of lists of lists
            (representing a list of partitions)
    returns: a number
            (lower scores mean the partition is more novel)
    """
    bench = float("inf")            # set high benchmark
    best = []
    for part in setlist:            # for each partition in the list
        ss = partscore_pairs(part,score)   # get score for that partition
        if ss > bench:              # if it's worse than the benchmark
            continue            # go to the next partition
        elif ss <= bench:            # if it's equal to or better than the benchmark
            bench = ss              # make it the new benchmark
            best.append(part)        # add it to the list of "best" partitions (allows multiple "bests")
    return best

