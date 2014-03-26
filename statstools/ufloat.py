import math

class ufloat(object):
    def __init__(self, value, stat, syst=None):
        self.value = value
        self.stat = stat
        self.syst = syst

    def __str__(self):
        return repr(self)

    def __repr__(self):
        if self.stat == 0:
            pos = 1
        else:
            pos = int(math.floor(math.log10(abs(self.stat))))
            pos = abs(pos) if pos < 0 else 1
        if self.syst is None:
            fmt = '${0:.'+str(pos)+'f} \pm {1:.'+str(pos)+'f}$'
            return fmt.format(self.value, self.stat)
        if self.syst[0] == 0:
            spos_up = 1
        else:
            spos_up = int(math.floor(math.log10(abs(self.syst[0]))))
            spos_up = abs(spos_up) if spos_up < 0 else 1
        if self.syst[1] == 0:
            spos_dn = 1
        else:
            spos_dn = int(math.floor(math.log10(abs(self.syst[1]))))
            spos_dn = abs(spos_dn) if spos_dn < 0 else 1
        pos = max(pos, spos_up, spos_dn)
        fmt = ('${0:.'+str(pos)+'f} \pm {1:.'+str(pos)+'f}$ '
                '${{}}^{{+{2:.'+str(pos)+'f}}}_{{-{3:.'+str(pos)+'f}}}$')
        return fmt.format(self.value, self.stat, self.syst[0], self.syst[1])

    def __iadd__(self,other):
        self.value += other.value
        self.stat = math.sqrt( self.stat+self.stat + other.stat*other.stat )
        if self.syst is not None and other.syst is not None:
            self.syst = math.sqrt( self.stat+self.stat + other.stat*other.stat )
        return self

    def __add__(self,other):
        summed_ufloat = ufloat(self.value,self.stat,self.syst)
        summed_ufloat += other
        return summed_ufloat
