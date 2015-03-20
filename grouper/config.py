import os

# Config

#student names
names = (
"mihai",
"djenet",
"russell",
"lily",
"tracy",
"saurabh",
"jen-yi",
"sarah",
"eileen",
"john",
"viktor",
"ram",
"carmen",
"samuel",
"shiaulou",
"yuxiang"
)

stations = [
'Nikon-1',
'Nikon-2',
'Olympus-WF1',
'Olympus-WF2',
'Olympus-TIRF',
'Olympus-FV1200',
'Leica-SP8',
'Leica-WF',
'Andor-WD',
'Biovision',
'Bruker-Opterra',
'DeltaVision']

numstudents=len(names)

# Scale for scaling the scoring mechanism
scale = 1

# whether or not to increment the scoring step with each round
# (once everyone has been together at least once,
# this tried to pair students with more distant partners)
increment = True

# NOTE: the final scoring increment (i.e. the amount added to either the station score or the student score
#    is the combination of scale * #rounds if increment is true, otherwise, just scale...)


# where to save the scoring object
savedir = os.curdir

# how many iterations the ititial shuffle and match groups goes through during the solve routine
# (before then matching that group to a station rotation)
headstart = 2000