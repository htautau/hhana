from rootpy.tree import Cut
import math
from .. import MMC_MASS

# All basic cut definitions are here

TAU1_MEDIUM = Cut('tau1_JetBDTSigMedium==1')
TAU2_MEDIUM = Cut('tau2_JetBDTSigMedium==1')
TAU1_TIGHT = Cut('tau1_JetBDTSigTight==1')
TAU2_TIGHT = Cut('tau2_JetBDTSigTight==1')

ID_MEDIUM = TAU1_MEDIUM & TAU2_MEDIUM
ID_TIGHT = TAU1_TIGHT & TAU2_TIGHT
ID_MEDIUM_TIGHT = (TAU1_MEDIUM & TAU2_TIGHT) | (TAU1_TIGHT & TAU2_MEDIUM)
# ID cuts for control region where both taus are medium but not tight
ID_MEDIUM_NOT_TIGHT = (TAU1_MEDIUM & -TAU1_TIGHT) & (TAU2_MEDIUM & -TAU2_TIGHT)

TAU_SAME_VERTEX = Cut('tau_same_vertex')

LEAD_TAU_35 = Cut('tau1_pt > 35000')
SUBLEAD_TAU_25 = Cut('tau2_pt > 25000')

LEAD_JET_50 = Cut('jet1_pt > 50000')
SUBLEAD_JET_30 = Cut('jet2_pt > 30000')
AT_LEAST_1JET = Cut('jet1_pt > 30000')

CUTS_2J = LEAD_JET_50 & SUBLEAD_JET_30
CUTS_1J = LEAD_JET_50 & (- SUBLEAD_JET_30)
CUTS_0J = (- LEAD_JET_50)

MET = Cut('MET_et > 20000')

# common preselection cuts
PRESELECTION = (
    LEAD_TAU_35 & SUBLEAD_TAU_25
    & MET
    & Cut('%s > 0' % MMC_MASS)
    & Cut('0.8 < dR_tau1_tau2 < 2.6')
    & TAU_SAME_VERTEX
    # looser MET centrality
    & Cut('MET_bisecting || (dPhi_min_tau_MET < %f)' % (0.2 * math.pi))
    )

CUTS_VBF = CUTS_2J

CUTS_BOOSTED = (
    # tighter MET centrality
    Cut('MET_bisecting || (dPhi_min_tau_MET < %f)' % (0.1 * math.pi))
    & Cut('resonance_pt > 80000'))
