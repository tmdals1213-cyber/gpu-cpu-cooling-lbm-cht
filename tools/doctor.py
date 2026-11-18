"""Print dependency/GPU readiness without exposing credentials or host paths."""
import importlib.metadata as metadata
import json,os,sys
from pathlib import Path


def main():
    cache=Path(__file__).resolve().parents[1]/'.cache'
    for variable,folder in [('CUPY_CACHE_DIR','cupy'),('TEMP','tmp'),('TMP','tmp')]:
        path=cache/folder;path.mkdir(parents=True,exist_ok=True);os.environ[variable]=str(path)
    packages={}
    for name in ['numpy','scipy','sympy','lbmpy','pystencils','cupy-cuda12x','trimesh','rtree']:
        try:packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError:packages[name]='MISSING'
    report=dict(python=sys.version.split()[0],packages=packages)
    try:
        import cupy as cp
        devices=[]
        for index in range(cp.cuda.runtime.getDeviceCount()):
            props=cp.cuda.runtime.getDeviceProperties(index);name=props['name']
            devices.append(dict(index=index,name=name.decode() if isinstance(name,bytes) else name,
                memory_GiB=round(props['totalGlobalMem']/1024**3,2)))
        report.update(devices=devices,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion())
        assert devices,'No CUDA GPU found'
        assert all(v!='MISSING' for v in packages.values()),'Missing dependencies'
        report['status']='ready_for_smoke_test'
    except Exception as exc:
        report.update(status='not_ready',error=str(exc));print(json.dumps(report,indent=2));raise SystemExit(1)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
