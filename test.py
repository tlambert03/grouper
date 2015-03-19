import grouper as G
from multiprocessing import Pool
import sys


cores = 8 if len(sys.argv) < 2 else int(float(sys.argv[1]))
iterations_per_thread = 1000 if len(sys.argv) < 3 else int(float(sys.argv[2]))

n = G.params.numrotations
stations = G.params.stations

if __name__ == '__main__':
    p = Pool(processes=cores)
    results = p.map(G.parallel_shuffle, [(stations,n,iterations_per_thread,iterations_per_thread/10)]*cores)
    print
    print
    print "--------------"
    print 'BEST RESULT:'
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
    print sol.part
    print "Partition Score: ", gs
    print sol.schedule
    print "Rotation Score: ", rs
    print "Combo Score: ", cs
    print "--------------"