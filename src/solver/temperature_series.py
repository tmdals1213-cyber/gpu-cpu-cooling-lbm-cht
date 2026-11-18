# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Interpolate saved physical-temperature snapshots; never manufacture a warm-up."""
import bisect
from pathlib import Path
import numpy as np


class TemperatureSeries:
    def __init__(self,folders):
        records={}
        for folder in folders:
            for path in sorted(Path(folder).glob('frame_*.npz')):
                with np.load(path) as data:records[float(data['time_s'])]=path
        self.times=sorted(records);self.paths=[records[t] for t in self.times];self.cache={}
        assert self.times
    def sample(self,t):
        t=float(np.clip(t,self.times[0],self.times[-1]))
        j=min(bisect.bisect_right(self.times,t),len(self.times)-1);i=max(0,j-1)
        for k in (i,j):
            if k not in self.cache:
                with np.load(self.paths[k]) as data:self.cache[k]=data['T_half'].copy()
        self.cache={k:v for k,v in self.cache.items() if k in (i,j)}
        fraction=0. if i==j else (t-self.times[i])/(self.times[j]-self.times[i])
        return self.cache[i]*(1-fraction)+self.cache[j]*fraction
