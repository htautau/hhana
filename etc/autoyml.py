import yaml
import ROOT
import itertools

with open('hhdb/hhdb/samples/hadhad/2015/backgrounds.yml') as infile:
    backgrounds = yaml.load(infile)

ntuple = ROOT.TFile('/tmpfs/ntuples_hh_run2/v7/hhskim/hhskim.root', 'r')

# IN YAML BUT NOT NTUPLES
yaml_no_ntup = []
for c in backgrounds.keys():
    for s in backgrounds[c]['samples']:
        if not ntuple.GetListOfKeys().Contains(s):
            yaml_no_ntup.append((c, s))


# IN NTUPLES BUT NOT YAML
ntup_no_yaml = []
for a in ntuple.GetListOfKeys():
    if '_daod' in a.GetName() or 'data' in a.GetName() or 'gg' in a.GetName() or 'VBF' in a.GetName():
        continue
    for c in backgrounds.keys():
        if a.GetName() in backgrounds[c]['samples']:
            break
    else:
        # DETERMINE CATEGORY
        if "tautau" in a.GetName():
            c = 'pythia_ztautau'
        else:
            c = 'others'
        ntup_no_yaml.append((c, a.GetName()))

print "IN YAML BUT NOT NTUP", yaml_no_ntup
print "IN NTUP BUT NOT YAML", ntup_no_yaml

print "DELETING STUFF FROM YAML"
for c, s in yaml_no_ntup:
    print s, " from ", c
    backgrounds[c]['samples'].remove(s)

print "APPENDING STUFF TO YAML"
for c, s in ntup_no_yaml:
    print s, " from ", c
    backgrounds[c]['samples'].append(s)

with open('backgrounds.yml', 'w') as outfile:
    outfile.write( yaml.dump(backgrounds) )
