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
        gs += pairscores[i[0]][i[1]]                        # requires pairs to be a symetric array... as above
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
