"""Voxelize the provided STL surrogate and export a small setup report (no solve)."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src/solver'))
from runtime_paths import OUTPUT


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['natural','ventilation','cpu_fan','cold_plate'],required=True);a=p.parse_args()
    if a.case=='cold_plate':
        from cold_plate import configuration,geometry
        ids,offset=geometry();meta=configuration()
    else:
        from case_setup import configuration,geometry
        ids,offset=geometry(a.case);meta=configuration(a.case)
    mirror=ids[::-1].copy();m2=mirror==2;m3=mirror==3;mirror[m2]=3;mirror[m3]=2
    assert np.array_equal(ids,mirror),'X symmetry broken'
    assert all(np.any(ids==m) for m in [2,3,4,5]),'Missing CPU material'
    meta.update(status='setup_checked_no_flow_solve',material_counts={str(m):int(np.sum(ids==m)) for m in np.unique(ids)},
                offset_mm=list(map(float,offset)),x_symmetry=True)
    (OUTPUT/'setup.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(dict(status=meta['status'],case=a.case,grid=list(ids.shape),output=str(OUTPUT/'setup.json'))))


if __name__=='__main__':main()
