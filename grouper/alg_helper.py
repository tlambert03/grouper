def scope_score_student(student):
	global scopescores
	score = {}
	for key in scopescores:
		score[key]=scopescores[key][student]
	return score


def best_students_for_vendors(vendors, rots):	# output groups of students best paired to vendors
	pass

def shuffle_match_scopes(vendors,n,iterations=1000, report=100):
    # randomly simultaneously check scope and student matches
	numgroups = len(vendors)
	bestcombo = float("inf")
	bestcombos = []
	bestpart = float("inf")
	bestparts = []
	bestvend = float("inf")
	bestvends = []

	for i in range(iterations):
		#make random partition and score for student matching
		part = partition(numpy.random.permutation(16).tolist(),numgroups)
		partscore = partitionscore(part)

		#score partition for scope matching
		r = poss_rotations(n, vendors)
		best = r[argmin(scoreallrots(part,r))]
		s = group_scope_match(part,best)
		vendscore = sum(s)

		# calculate combined score
		combscore = vendscore * partscore

		if combscore < bestcombo:
			bestcombo = combscore
			bestcombos.append([part,best])
		if partscore < bestpart:
			bestpart = partscore
			bestparts.append(part)
		if vendscore < bestvend:
			bestvend = vendscore
			bestvends.append(best)
		if i%report==0:
			print "Round %s, best so far: %f " % (i,bestcombo)

	print "best partition score: %f" % bestpart
	print "best vendor score: %d" % bestvend
	print "best combo score: %f, (%f/%d)" % (bestcombo, partitionscore(bestcombos[-1][0]), sum(group_scope_match(bestcombos[-1][0],bestcombos[-1][1])))
	return (bestparts, bestvends, bestcombos)
