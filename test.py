import grouper as G
from multiprocessing import Pool
import sys

cores = int(float(sys.argv[1]))
n = int(float(sys.argv[2]))
iterations_per_thread = 1000 if len(sys.argv) < 4 else int(float(sys.argv[3]))



if __name__ == '__main__':
    p = Pool(processes=cores)
    results = p.map(G.parallel_shuffle, [(G.config.stations[5:9],n,iterations_per_thread,iterations_per_thread)]*cores)
    print
    print
    print "--------------"
    print 'RESULTS:'
    print
    bench = float("inf")
    for sol in results:
        s = G.scoreSolution(sol)
        gs = sum([g[0] for g in s])
        rs = sum([g[1] for g in s])
        cs = gs * rs
        if cs < bench:
            bench = cs
            best = sol
    print best.printSol() 
    s = G.scoreSolution(best)
    gs = sum([g[0] for g in s])
    rs = sum([g[1] for g in s])
    cs = gs * rs
    print "Grouping Score: ", gs
    print "Rotation Score: ", rs
    print "Combo Score: ", cs
    print "--------------"