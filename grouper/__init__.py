import numpy.random
from numpy import argmin, array
import copy
import cPickle as pickle
import os
import sys
from itertools import combinations
import multiprocessing

from grouper import config
from grouper import params
from random import shuffle



#############################################
#              Score Class                  #
#############################################


class Score:
    """
    A Score object must be instantiated to use the program.
    It stores the pair-scores dict (the dict that holds how many times
    each student has been with any another student)
    It also stores the station-scores dict (the dict that stores how many
    times any given station has seen any given student)

    it should be updated after each "round"  with a Solution object using the update() method
    """

    def __init__(self):
        self.numstudents = config.numstudents
        self.names = config.names
        self.stations = config.stations
        self.rounds = 0
        self.history = []
        self.future = []

        self.reset_pairs()
        self.reset_stationscores()

    def reset_pairs(self):
        """resets the pairscores variable to 0 for every student combo."""
        self.pairscores={}
        for i in range(self.numstudents): self.pairscores[i]=[0]*self.numstudents

    def reset_stationscores(self):
        self.stationscores = {}
        for s in self.stations: self.stationscores[s]=[0]*self.numstudents

    def reset_rounds(self):
        self.rounds=0

    def reset(self):
        self.history.append(copy.deepcopy(self))
        self.reset_pairs()
        self.reset_stationscores()
        self.reset_rounds()

    def random(self):
        self.rand_pairscores()
        self.rand_stationscores()

    def rand_pairscores(self):
        """generates random pairscores data... for testing"""
        maxscore = 12
        self.pairscores={}
        b = numpy.random.random_integers(0,maxscore,size=(self.numstudents,self.numstudents))
        b_symm = (b + b.T)/2
        for i in range(len(b_symm)):
            b_symm[i][i]=0
        self.pairscores = dict(zip(range(self.numstudents), b_symm.T.tolist()))
        self.rounds=numpy.random.randint(10)
        self.print_pairscores()

    def rand_stationscores(self):
        maxscore = 5
        self.stationscores = {}
        for s in self.stations: self.stationscores[s]=numpy.random.randint(maxscore, size=self.numstudents).tolist()
        self.print_stationscores()

    def update(self,solution,scale=config.scale,inc=config.increment):
        """updates the score given the partition provided"""
        self.history.append(copy.deepcopy(self))
        self.solutions.append(solution)
        self.rounds+=1
        i = 0
        for group in solution.part: # for every group in the partition
            #update the pairscores for each student combo in the group...
            for n in combinations(group,2):
                if inc:
                    self.pairscores[n[0]][n[1]] += scale * self.rounds
                    self.pairscores[n[1]][n[0]] += scale * self.rounds
                else:
                    self.pairscores[n[0]][n[1]] += scale
                    self.pairscores[n[1]][n[0]] += scale
            for student in group:
                for station in solution.schedule[i]:
                    if inc:
                        self.stationscores[station][student] += scale * self.rounds
                    else:
                        self.stationscores[station][student] += scale
            i += 1

    def print_pairscores(self):
        print "%8s" % "",
        for name in self.names:
            print "%8s" % name,
        print "\n"
        for key in self.pairscores:
            print "%8s" % self.names[key],
            for i in self.pairscores[key]:
                print "%8s" % i,
            print "\n"

    def print_stationscores(self):
        print "%14s" % "",
        for name in self.names:
            print "%8s" % name,
        print "\n"
        for key in self.stationscores:
            print "%14s" % key,
            for i in self.stationscores[key]:
                print "%8s" % i,
            print "\n"


    def rollback(self,steps=1):
        for i in range(steps):
            self.future.append(copy.deepcopy(self))
            temp = self.history.pop(-1)
            self.numstudents = temp.numstudents
            self.names = temp.names
            self.stations = temp.stations
            self.rounds = temp.rounds
            self.pairscores = temp.pairscores
            self.stationscores = temp.stationscores


    def rollforward(self,steps=1):
        for i in range(steps):
            self.history.append(copy.deepcopy(self))
            temp = self.future.pop(-1)
            self.numstudents = temp.numstudents
            self.names = temp.names
            self.stations = temp.stations
            self.rounds = temp.rounds
            self.pairscores = temp.pairscores
            self.stationscores = temp.stationscores

    def save(self, savedir=config.savedir, name='save'):
        name = name + '.p'
        if os.path.isfile(os.path.join(savedir,name)):
            sys.stdout.write("Save file already exists, overwrite? ('y' or 'n')\n")
            choice = raw_input().lower()
            if choice=='n':
                sys.stdout.write("Input a new filename:\n")
                name = raw_input().lower() + ".p"
                if not os.path.isfile(os.path.join(savedir,name)):
                    pickle.dump(self, open( os.path.join(savedir,name), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    print "file exists, please try saving again."
            elif choice=='y':
                pickle.dump(self, open( os.path.join(savedir,name), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print "didn't understand answer, please try saving again"
                return
        else:
            pickle.dump(self, open( os.path.join(savedir,name), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, f=None):
        f = f or os.path.join(config.savedir,'save.p')
        temp = pickle.load( open(f, "rb" ) )
        self.numstudents = temp.numstudents
        self.names = temp.names
        self.stations = temp.stations
        self.rounds = temp.rounds
        self.history = temp.history
        self.future = temp.future
        self.pairscores = temp.pairscores
        self.stationscores = temp.stationscores

class Solution():
    """ A Solution object represents a parition of students and
    the rotation schedule being used """

    def __init__(self, part, schedule):
        self.part, self.schedule = part,schedule
        #self.numgroups = len(self.part)
        #self.numrotations = len(self.schedule)

    def makedict(self):
        self.dict = {}
        g = 1
        for group in self.part:
            self.dict[g] = {}
            self.dict[g]['students'] = group
            self.dict[g]['schedule'] = self.schedule[g-1]
            # could put additional group scores here...
            g += 1

    def printSol(self):
        i = 1
        for group in self.part:
            print "Group %d: (%.2f)" % (i,round(groupscore_pairs(group),2))
            print ", ".join([config.names[g] for g in group])
            print "Schedule: (%.2f)" % (round(groupscore_stations(group,self.schedule[i-1]),2))
            print ", ".join([station for station in self.schedule[i-1]] )
            print
            i += 1
        s = scoreSolution(self)
        gs = sum([g[0] for g in s])
        rs = sum([g[1] for g in s])
        cs = gs + rs
        print self.part
        print self.schedule
        print "Partition Score: ", gs
        print "Rotation Score: ", rs
        print "Combo Score: ", cs
        print "--------------"

    def printSchedule(self, withscores=0):
        pass


def loadScore(f=None):
    f = f or os.path.join(config.savedir,'save.p')
    return pickle.load( open(f, "rb" ) )


S = Score()
#S.rand_pairscores()
#S.rand_stationscores()
S.load()

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


def shuffleandtest(numgroups, it=1000, results=1, score=S):
    """
    this randomly shuffles the students into a partition of numgroups
    and then scores that partition...

    this repeats it number of times and returns the partition
    that scored the best
    """
    bench = float("inf")                    #set benchmark super high
    best = []
    for i in range(it):
        p = rand_partition(numgroups)              #make random partition of students into n groups
        s = partscore_pairs(p,score)             #score the partition
        if s < bench:                         #if the score is better than the benchmark
            bench = s                       #make it the new benchmark
            best.append(p)                    #and store that partition of groups...
    if results == 1:
        #print "top score: ",
        #print partscore_pairs(best[-1], score)
        return best[-1]
    elif results == 0:
        #print "top scores: ",
        #print ", ".join([str(round(partscore_pairs(p,score),2)) for p in best])
        return best
    else:
        #print "top scores: ",
        #print ", ".join([str(round(partscore_pairs(p,score),2)) for p in best[(results * -1):]])
        return best[(results * -1):]


def greedypairs(numgroups, it=1, lst=None, score=S):
    lst=lst or shuffleandtest(numgroups,1000)
    newsets = copy.deepcopy(lst)
    #alt=[]
    for i in xrange(it):
        besti = argmin(array([groupscore_pairs(g) for g in newsets])) #find index of group with best score
        b = newsets[besti]                      #store best group
        del newsets[besti]                      #remove the best group from the bunch
        newsets = [i for sub in newsets for i in sub]               #flatten the bunch
        #f = findbestpart(algorithm_u(newsets,numgroups-1))   #find the best rearrangement of the bottom groups
        f = findbestpart(allgroups(newsets, numgroups-1))
        newsets = f[0]                          #pick the first option (if there are more than 1)
        newsets.append(b)                       #add back the good group
        print "new score: %f" % partscore_pairs(newsets)
        #if len(f)>1:
            #print "there were more options"     #alert if there were other paths not persued
            #alt = f[1:]     #store the alternatives
            #[i.append(b) for i in alt]
    return newsets                  #return the modified list of groups


def findbestpart(partlist, score=S):
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
        ss = partscore_pairs(part,score)   # get score for that partition
        if ss > bench:              # if it's worse than the benchmark
            continue            # go to the next one
        elif ss < bench:            # if it's better than the benchmark
            ind=[part]          # store that partition (for return)
            bench = ss              # make it the new benchmark
            continue            # go on to the next
        elif ss == bench:           # if it's the SAME as the benchmark
            ind.append(part)        # add it to the list of "best" partitions (allows multiple "bests")
    return ind

def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def powergroups(students, groups):
    s = range(students)
    minsize=students/groups
    even = 1 if students%groups else 0
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1) if r==minsize or r==(minsize+even))


#############################################
#         Scoring functions                 #
#############################################



def groupscore_pairs(group,normed=1,score=S):
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

def partscore_pairs(part,score=S):
    """Scores the partition for student groupings accoring to the provided score class,
    by performing the groupscore function for every group in the partition

    returns: a number (lower scores mean the partition is more novel)
    """
    ss = 0
    for group in part:
        ss += groupscore_pairs(group)  # requires pairs to be a symetric array... as above
    return ss


def groupscore_stations_dict(group, stations, score=S):
    """returns a dict with the sum score for the given group at any given station
    """
    result={}
    for station in stations:
        result[station] = 0
    for person in group:
        for station in stations:
            result[station] += score.stationscores[station][person]
    return result

def groupscore_stations(group, stations, score=S):
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

def partscore_stations(part, stations, score=S):
    """returns a list of lists scoring vendor appropriateness
    for each group in the partition.
    Sorted with the best scoring station 1st
    """
    return [sorted(i,key=lambda x: x[1]) for i in [[t for t in n.iteritems()] for n in [groupscore_stations(i, stations) for i in part]]]



def scoreSolution(solution, normed=1, score=S,):
    """
    takes a Solution object and scores it against a Score object
    returns a list of tuples representing the (group, vendor) score for each group in the partition
    """
    numgroups=len(solution.part)
    ss = []
    for group in range(numgroups):
        g = groupscore_pairs(solution.part[group], normed)
        v = groupscore_stations(solution.part[group], solution.schedule[group])
        ss.append((round(g,2),v))
    return ss


def rankpartitions(setlist, score=S):
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
        ss = partscore_pairs(part)   # get score for that partition
        if ss > bench:              # if it's worse than the benchmark
            continue            # go to the next partition
        elif ss <= bench:            # if it's equal to or better than the benchmark
            bench = ss              # make it the new benchmark
            best.append(part)        # add it to the list of "best" partitions (allows multiple "bests")
    return best



#############################################
#         Vendor Matching functions         #
#############################################

def scorescopes_group(group, score=S):   # returns dict with sum of individual scope scores for all vendors for a given group
    result={}
    for vendor in score.stationscores:
        result[vendor] = 0
    for person in group:
        for vendor in score.stationscores:
            result[vendor] += score.stationscores[vendor][person]
    return result

def scorescope(part,score):   # returns a list of lists scoring vendor appropriateness for each group in the partition
    return [sorted(i,key=lambda x: x[1]) for i in [[t for t in n.iteritems()] for n in [scorescopes_group(i) for i in part]]]

def group_scope_match(part,rots,score=S):
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

def scoreallrots(part,rotslist, score=S):
    return [sum(group_scope_match(part,rot)) for rot in rotslist]


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
            rotate_list(vendors,rots[2]),rotate_list(vendors,rots[3]),rotate_list(vendors,rots[4]),
            rotate_list(vendors,rots[5]))

    def seven():
        return zip(rotate_list(vendors,rots[0]),rotate_list(vendors,rots[1]),
            rotate_list(vendors,rots[2]),rotate_list(vendors,rots[3]),rotate_list(vendors,rots[4]),
            rotate_list(vendors,rots[5]),rotate_list(vendors,rots[6]))

    def eight():
        return zip(rotate_list(vendors,rots[0]),rotate_list(vendors,rots[1]),
            rotate_list(vendors,rots[2]),rotate_list(vendors,rots[3]),rotate_list(vendors,rots[4]),
            rotate_list(vendors,rots[5]),rotate_list(vendors,rots[6]),rotate_list(vendors,rots[7]))

    def nine():
        return zip(rotate_list(vendors,rots[0]),rotate_list(vendors,rots[1]),
            rotate_list(vendors,rots[2]),rotate_list(vendors,rots[3]),rotate_list(vendors,rots[4]),
            rotate_list(vendors,rots[5]),rotate_list(vendors,rots[6]),rotate_list(vendors,rots[7]),
            rotate_list(vendors,rots[8]))


    options = {
        2 : two,
        3 : three,
        4 : four,
        5 : five,
        6 : six,
        7 : seven,
        8 : eight,
        9 : nine,
    }

    return options[numrots]()

def rotate_list(lst,n):
    return lst[-n:] + lst[:-n]

########################


def find_best_rots(part,vendors,n,score=S):
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
            #print "groups score: %f" % partscore_pairs(l,score)
            #comboscore = sum(s)*partscore_pairs(l,score)
            #print "comboscore %f" % comboscore
            #print "----------------------"
    else:
        best = r[argmin(scoreallrots(part,r))]
        s = group_scope_match(part,best)
        for i in range(len(part)):
            print "%s: " % str(", ".join(best[i]))
            pretty_names(part[i:i+1])
            print "score: %s\n" % str(s[i])
        print "vendor score: %d" % sum(s)
        print "groups score: %f" % partscore_pairs(part)
        comboscore = sum(s)*partscore_pairs(part)
        print "comboscore %f" % comboscore
        print "----------------------"



def shuffle_match_scopes(vendors,n,it=1000,report=100, headstart=config.headstart):
    """
    generate random partition according to number of vendor stations

    inputs:
    vendors =
    n =
    *it = number of times to randomly iterate
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

    for i in range(it):
        #make random partition and score for student matching
        part = shuffleandtest(numgroups,headstart)
        #part = partition(numpy.random.permutation(16).tolist(),numgroups)
        partscore = partscore_pairs(part)

        #score partition for scope matching
        best = r[argmin(scoreallrots(part,r))] # which of the possible rotations is best for this random partition?
        s = group_scope_match(part,best) # s is the per-group scope matching score
        vendscore = sum(s)  # vendscore is the sum group-scope match score

        # calculate combined score
        comboscore = vendscore + partscore

        if comboscore < bestcombo:
            if i > it/4: #avoid dumping out the early stuff...
                print "--------COMBO---------"
                print "COMBO: %.2f " % (round(comboscore,2))
                print "Partition: %s " % (part)
                print "score: %.2f " % (round(partscore,2))
                print "Schedule: %s " % (best)
                print "score: %.2f " % (round(vendscore,2))
                print "--------/COMBO--------"
            bestcombo = comboscore
            bestcombos.append([part,best])
        #if partscore < bestpart:
            #print " "
            #print "PARTITION: %s " % (part)
            #print "score: %f " % (round(partscore,2))
            #print "----------------------"
            #bestpart = partscore
            #bestparts.append(part)
        if vendscore < bestvend:
            if i > it/4:
                print "-------VENDOR--------"
                print "combo: %.2f " % (round(comboscore,2))
                print "Partition: %s " % (part)
                print "score: %.2f " % (round(partscore,2))
                print "Schedule: %s " % (best)
                print "score: %.2f " % (round(vendscore,2))
                print "-------/VENDOR--------"
            bestvend = vendscore
            bestvends.append(best)
        if i%report==0:
            print "########Round %s, best so far: %.2f #########" % (i,round(bestcombo,2))
            #print "partition: %s " % (bestcombos[-1][0])
            #print "score: %s " % (partscore_pairs(bestcombos[-1][0]))
            #print "vendors: %s " % (bestcombos[-1][1])
            #print "score: %f " % (bestvend)
            #print "----------------------"

    #print "best partition score: %f" % bestpart
    #print "best vendor score: %d" % bestvend
    #print "best combo score: %f, (%f/%d)" % (bestcombo, partscore_pairs(bestcombos[-1][0]), sum(group_scope_match(bestcombos[-1][0],bestcombos[-1][1])))
    #return (bestparts, bestvends, bestcombos)
    return Solution(bestcombos[-1][0],bestcombos[-1][1])


def parallel_shuffle(args):
    stations,numrotations,iterations,report = args
    return shuffle_match_scopes(stations,numrotations,iterations,report)

def solve(stations=params.stations, numrotations=params.numrotations, cores=multiprocessing.cpu_count(), iterations=1000, report=100):
    p = multiprocessing.Pool(processes=cores)
    results = p.map(parallel_shuffle, [(stations,numrotations,iterations,report)]*cores)
    print
    print
    print "--------------"
    print 'BEST RESULT:'
    print
    bench = float("inf")
    for sol in results:
        s = scoreSolution(sol)
        gs = sum([g[0] for g in s])
        rs = sum([g[1] for g in s])
        cs = gs * rs
        if cs < bench:
            bench = cs
            best = sol
    print best.printSol()
    return best



def greedy_match_scopes(vendors,n,it=100, report=10):
    """
    generate random partition according to number of vendor stations

    inputs:
    vendors =
    n =
    *it = number of times to randomly iterate
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

    for i in range(it):
        #make random partition and score for student matching
        part = greedypairs(numgroups)
        #part = partition(numpy.random.permutation(16).tolist(),numgroups)
        partscore = partscore_pairs(part)

        #score partition for scope matching
        best = r[argmin(scoreallrots(part,r))] # which of the possible rotations is best for this random partition?
        s = group_scope_match(part,best) # s is the per-group scope matching score
        vendscore = sum(s)  # vendscore is the sum group-scope match score

        # calculate combined score
        comboscore = vendscore + partscore

        if comboscore < bestcombo:
            print " "
            print "new best COMBO:"
            print "partition: %s " % (part)
            print "vendor match: %s " % (best)
            print "part * vendor = combo score: %f + %f = % f" % (partscore,vendscore,comboscore)
            print "----------------------"
            bestcombo = comboscore
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
            #print "score: %s " % (partscore_pairs(bestcombos[-1][0]))
            #print "vendors: %s " % (bestcombos[-1][1])
            #print "score: %f " % (bestvend)
            #print "----------------------"

    print "best partition score: %f" % bestpart
    print "best vendor score: %d" % bestvend
    print "best combo score: %f, (%f/%d)" % (bestcombo, partscore_pairs(bestcombos[-1][0]), sum(group_scope_match(bestcombos[-1][0],bestcombos[-1][1])))
    return (bestparts, bestvends, bestcombos)


#############################################
#         Algorithm x                       #
#############################################



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


#############################################
#             Helper functions              #
#############################################

def equal_partitions(a,b):
    """ check whether two partitions represent the same grouping of students"""
    return set(frozenset(i) for i in a) == set(frozenset(i) for i in b)


def dupebranch(test,stored):
    for i in stored:
        if equal_partitions(i,test):
            return True
    return False


def pretty_names(parts):
    for n in parts:
        print ', '.join(config.names[i] for i in n)