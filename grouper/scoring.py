from grouper import config
import numpy.random
from itertools import combinations
import copy
import cPickle as pickle

#############################################
#             Pair Score functions          #
#############################################

class Score:

    def __init__(self, count=config.numstudents, names=config.names):
        self.score = {}
        self.count = count or 16
        self.names = names or []
        self.rounds = 0
        self.history = []
        self.future = []

    def reset(self):
        """resets the pairscores variable to 0 for every student combo."""
        self.score={}
        for i in range(self.count): self.score[i]=[0]*self.count
        self.rounds=0  # this currently ruins the proper count on the rollbacks

    def random(self):
        """generates random pairscores data... for testing"""
        self.score={}
        b = numpy.random.random_integers(0,12,size=(self.count,self.count))
        b_symm = (b + b.T)/2
        for i in range(len(b_symm)):
            b_symm[i][i]=0
        self.score = dict(zip(range(self.count), b_symm.T.tolist()))
        self.rounds=numpy.random.randint(10)

    def update(self,partition,scale=config.scale,inc=config.increment):
        """updates the score given the partition provided"""
        self.history.append(copy.deepcopy(self.score))
        self.rounds+=1
        for i in partition:
            for n in combinations(i,2):
                if inc:
                    self.score[n[0]][n[1]] += scale * self.rounds
                    self.score[n[1]][n[0]] += scale * self.rounds
                else:
                    self.score[n[0]][n[1]] += scale
                    self.score[n[1]][n[0]] += scale


    def printscore(self):
        print "%8s" % "",
        for name in config.names:
            print "%8s" % name,
        print "\n"
        for key in self.score:
            print "%8s" % config.names[key],
            for i in self.score[key]:
                print "%8s" % i,
            print "\n"
        pass

    def rollback(self,steps=1):
        for i in range(steps):
            self.future.append(copy.deepcopy(self.score))
            self.score = self.history.pop(-1)
            self.rounds -= 1

    def rollforward(self,steps=1):
        for i in range(steps):
            self.history.append(copy.deepcopy(self.score))
            self.score = self.future.pop(-1)
            self.rounds += 1


class Partition:

    def __init__(self, part):
        self.part = part
        self.numgroups = len(self.part)

    def score(self,score):
        """Scores the partition accoring to the provided score class,
        by performing the groupscore function for every group in the partition

        returns: a number (lower scores mean the partition is more novel)
        """
        ss = 0
        for i in self.part:
            ss += groupscore(i,score)                 # requires pairs to be a symetric array... as above
        return ss


def groupscore(grouplist,score):
    """Scores a group accoring to the global pairscores dict

    accepts: a python list
            (representing the student ids of a putative group)
    returns: a number
            (lower scores mean the grouping is more novel)
    """
    groupsize = len(grouplist) # number of students in the group
    groupscore = 0
    for i in combinations(grouplist,2):
        groupscore += score.score[i[0]][i[1]]         # requires pairs to be a symetric array... as above
    groupscore = groupscore/float(((groupsize**2/2) - (groupsize/2)))       # normalize the score to the size of the group
    return groupscore

#############################################
#             Scope Score functions         #
#############################################


#scopescores
"""
a dictionary where each key is a vendor name,
and each value is a python list representing how frequently each student has been at that vendor
"""


#def reset_scopescores():
#    global scopescores
#    scopescores={}
#    v = ['nikon','olympus','zeiss','leica','andor','api']
#    for i in v: scopescores[i]=[0]*16
#
#def random_scopescores():
#    global scopescores
#    scopescores={}
#    v = ['nikon','olympus','zeiss','leica','andor','api']
#    for i in v: scopescores[i]=numpy.random.randint(5, size=16).tolist()
#
#def update_scopescores(groups,scsc,scale=config.scale):        #updates the scope scores array given the groups provided
#    pass
#