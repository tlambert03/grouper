from grouper import scoring, config
from random import shuffle
from numpy import array, argmin
from grouper.algX import *
import copy


#############################################
#         Student Grouping functions        #
#############################################

# a combination is a pairing of two students
# a group is a set of 2 or more students (also called a subset here, I think?)
# a partition is a grouping of the all students into subsets (groups)


def partition(lst, n):
    """creates a simple partition of lst into n groups"""
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]


def rand_partition(numgroups, numstudents=config.numstudents):
    """
    returns a random partition of students containing "numgroups" groups
    """
    lst = range(numstudents)
    shuffle(lst)
    division = len(lst) / float(numgroups)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(numgroups) ]


def shuffleandtest(numgroups, score, results=1, iter=1000):
    """
    this accepts a number, randomly partitions it into n subsets
    and then scores that partition...

    this repeats iter number of times and returns the partition
    that scored the best
    """
    bench = float("inf")                    #set benchmark super high
    best = []
    for i in range(iter):
        p = rand_partition(numgroups)              #make random partition of students into n groups
        s = scoring.partscore_pairs(p,score)             #score the partition
        if s < bench:                         #if the score is better than the benchmark
            bench = s                       #make it the new benchmark
            best.append(p)                    #and store that partition of groups...
    if results == 1:
        print "top score: ",
        print scoring.partscore_pairs(best[-1], score)
        return best[-1]
    elif results == 0:
        print "top scores: ",
        print ", ".join([str(round(scoring.partscore_pairs(p,score),2)) for p in best])
        return best
    else:
        print "top scores: ",
        print ", ".join([str(round(scoring.partscore_pairs(p,score),2)) for p in best[(results * -1):]])
        return best[(results * -1):]


def greedypairs(numgroups, score, iter=1, lst=None):
    lst=lst or shuffleandtest(numgroups,score,1,1000)
    newsets = copy.deepcopy(lst)
    #alt=[]
    for i in xrange(iter):
        besti = argmin(array([scoring.groupscore_pairs(i,score) for i in newsets])) #find index of group with best score
        b = newsets[besti]                      #store best group
        del newsets[besti]                      #remove the best group from the bunch
        newsets = [i for sub in newsets for i in sub]               #flatten the bunch
        #f = findbestpart(algorithm_u(newsets,numgroups-1))   #find the best rearrangement of the bottom groups
        f = findbestpart(allgroups(newsets, numgroups-1), score)
        newsets = f[0]                          #pick the first option (if there are more than 1)
        newsets.append(b)                       #add back the good group
        print "new score: %f" % scoring.partscore_pairs(newsets, score)
        #if len(f)>1:
            #print "there were more options"     #alert if there were other paths not persued
            #alt = f[1:]     #store the alternatives
            #[i.append(b) for i in alt]
    return newsets                  #return the modified list of groups


def findbestpart(partlist, score):
    """goes through a list of partitions and scores all of them,
    returning a list of the best options

    accepts: a python list of lists of lists
            (representing )
    returns: a list of partitions
            (later partitions are higher scoring)
    """
    bench = float("inf")            # set high benchmark
    ind = []
    for part in partlist:            # for each partition in the list
        ss = scoring.partscore_pairs(part,score)   # get score for that partition
        if ss > bench:              # if it's worse than the benchmark
            continue            # go to the next one
        elif ss < bench:            # if it's better than the benchmark
            ind=[part]          # store that partition (for return)
            bench = ss              # make it the new benchmark
            continue            # go on to the next
        elif ss == bench:           # if it's the SAME as the benchmark
            ind.append(part)        # add it to the list of "best" partitions (allows multiple "bests")
    return ind
