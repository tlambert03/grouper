from itertools import chain, combinations
import copy
from random import shuffle
from numpy import array, argmin
from pprint import pprint
import cPickle as pickle
import os

from grouper import config, partition, grouping, vendor, helpers, scoring

State = pickle.load( open( os.getcwd()+"/save.p", "rb" ) )