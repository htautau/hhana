

def treesplit(tree):
    """
    Split input tree in half and return two output trees
    Used by root2hdf5
    """
    tree1 = tree.CopyTree('Entry$%2==0')
    tree1.SetName('%s_0' % tree.GetName())
    tree2 = tree.CopyTree('Entry$%2==1')
    tree2.SetName('%s_1' % tree.GetName())
    return [tree1, tree2]
