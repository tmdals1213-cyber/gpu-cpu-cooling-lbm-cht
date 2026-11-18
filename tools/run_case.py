"""One explicit entry point; planning/setup never starts a long calculation."""
from pathlib import Path
import argparse,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];ENGINE=ROOT/'src/solver'
CASES=json.loads((ROOT/'cases.json').read_text())


def commands(key,cfg,out,with_air):
    r=cfg['run'];py=sys.executable
    def cmd(name,*args):return [py,str(ENGINE/name),*map(str,args)]
    if key=='natural':
        return [('heat',cmd('run_transient_buffered.py','--case',key,'--name','heat','--dt',r['flow_dt_s'],
            '--max-time',r['max_heat_time_s'],'--sample-interval',.2,'--save-interval',r['save_interval_s'],'--checkpoint'))]
    if key=='cold_plate':
        stages=[('flow',cmd('cold_plate.py','--solve')),
                ('heat',cmd('fixed_flow_heat.py','--case',key,'--flow',out/'cold_plate_run/flow_fields.npz',
                 '--name','cold_plate_heat','--lb-dt',r['flow_dt_s'],'--heat-dt',r['heat_dt_s'],
                 '--max-time',r['max_heat_time_s'],'--save-interval',r['save_interval_s']))]
        if with_air:
            stages.extend([('case-air-flow',cmd('water_case_air.py','--name','water_case_air_steady','--pr-screening')),
                           ('case-air-heat',cmd('water_air_heat_history.py','--name','water_case_air_transient'))])
        return stages
    return [('flow',cmd('prepare_flow.py','--case',key,'--name','flow','--dt',r['flow_dt_s'],
                '--duration',r['preparation_s'],'--average-window',r['average_window_s'])),
            ('heat',cmd('fixed_flow_heat.py','--case',key,'--flow',out/'flow/flow_fields.npz','--name','heat',
                '--lb-dt',r['flow_dt_s'],'--heat-dt',r['heat_dt_s'],'--max-time',r['max_heat_time_s'],
                '--save-interval',r['save_interval_s']))]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=CASES,required=True)
    p.add_argument('--stage',choices=['plan','setup','smoke','flow','heat','case-air-flow','case-air-heat','run'],default='plan')
    p.add_argument('--output-root',type=Path);p.add_argument('--with-case-air',action='store_true')
    a=p.parse_args();cfg=json.loads((ROOT/CASES[a.case]/'case.json').read_text())
    if cfg['model']['power_W']!=200:raise ValueError('These drivers use fixed 200 W. Modify source consistently before changing this model summary.')
    out=(a.output_root or ROOT/'runs'/a.case).resolve()
    env=os.environ.copy();env['COOLING_OUTPUT_ROOT']=str(out);env['COOLING_CACHE_ROOT']=str(ROOT/'.cache')
    temporary=ROOT/'.cache/tmp';temporary.mkdir(parents=True,exist_ok=True)
    env['TMP']=str(temporary);env['TEMP']=str(temporary)
    env['CUPY_CACHE_DIR']=str(ROOT/'.cache/cupy')
    env['PYSTENCILS_CACHE_DIR']=str(ROOT/'.cache/pystencils')
    stages=commands(a.case,cfg,out,a.with_case_air)
    if a.stage=='plan':
        print(json.dumps(dict(case=a.case,output_root=str(out),stages=[dict(stage=s,command=c) for s,c in stages],
            note='Long GPU runs are explicit. Reference raw fields are intentionally excluded.'),indent=2));return
    if a.stage=='smoke':
        subprocess.run([sys.executable,str(ROOT/'tools/smoke_test.py')],env=env,check=True);return
    if a.stage=='setup':
        subprocess.run([sys.executable,str(ROOT/'tools/export_setup.py'),'--case',a.case],env=env,check=True);return
    selected=stages if a.stage=='run' else [(s,c) for s,c in stages if s==a.stage]
    if not selected:p.error('This case has no requested stage; use --with-case-air for optional water/case-air stages')
    # Refuse stale/failed prepared flows rather than silently proceeding to heat.
    for stage,command in selected:
        if stage=='flow' and a.case=='cold_plate' and (out/'cold_plate_run/result.json').exists():
            raise FileExistsError('Cold-plate result already exists; use a new --output-root')
        if stage=='heat' and a.case in ['ventilation','cpu_fan']:
            meta=out/'flow/metadata.json'
            if not meta.exists() or json.loads(meta.read_text())['status']!='flow_prepared_for_heat_history':
                raise RuntimeError('Run and pass the flow stage first')
        subprocess.run(command,env=env,cwd=ROOT,check=True)


if __name__=='__main__':main()
