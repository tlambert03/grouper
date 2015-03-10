from grouper import cfg
import numpy.random
from itertools import combinations

#############################################
#             Pair Score functions          #
#############################################

class Score:

    def __init__(self, count=16, names=cfg.names):
        self.score = {}
        self.count = len(names) or count
        self.names = names or []
        self.rounds = 0

    def reset(self):
        """resets the pairscores variable to 0 for every student combo."""
        self.score={}
        for i in range(self.count): self.score[i]=[0]*self.count
        self.rounds=0
        self.score

    def random(self):
        """generates random pairscores data... for testing"""
        self.score={}
        b = numpy.random.random_integers(0,12,size=(self.count,self.count))
        b_symm = (b + b.T)/2
        for i in range(len(b_symm)):
            b_symm[i][i]=0
        self.score = dict(zip(range(self.count), b_symm.T.tolist()))
        self.rounds=numpy.random.randint(10)

    def update(self,partition,scale=cfg.scale):
        """updates the pairscores dict given the partition provided"""
        self.score['i']=self.score['i']+1
        for i in partition:
            for n in combinations(i,2):
                self.score[n[0]][n[1]] += scale*self.score['i']
                self.score[n[1]][n[0]] += scale*self.score['i']

    def printscore(self):
        print "%8s" % "",
        for name in cfg.names:
            print "%8s" % name,
        print "\n"
        for key in self.score:
            print "%8s" % cfg.names[key],
            for i in self.score[key]:
                print "%8s" % i,
            print "\n"
        pass


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
#def update_scopescores(groups,scsc,scale=cfg.scale):        #updates the scope scores array given the groups provided
#    pass
#