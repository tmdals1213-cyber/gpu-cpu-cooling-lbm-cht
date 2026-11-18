"""Dependency-free source/setup/media audit; does not execute GPU code."""
from pathlib import Path
import argparse,ast,hashlib,json,os,re
ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','.cache','.venv','runs','__pycache__'}
EXTENSIONS={'.py','.md','.txt','.json','.csv','.png','.mp4','.stl'}
SPECIAL={'LICENSE','NOTICE.txt','.gitignore','.gitattributes'}


def payload_files():
    for folder,dirs,names in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in SKIP]
        for name in names:
            path=Path(folder)/name
            if path.name=='PUBLICATION_MANIFEST.json':continue
            if path.is_symlink():raise ValueError('Symlinks are not allowed in this public payload')
            yield path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write-manifest',action='store_true');a=parser.parse_args()
    files=list(payload_files());errors=[];entries=[]
    for p in files:
        rel=p.relative_to(ROOT).as_posix();size=p.stat().st_size
        if p.suffix not in EXTENSIONS and p.name not in SPECIAL:errors.append('Unexpected payload file: '+rel)
        if size>=50*1024**2:errors.append('File exceeds package 50 MiB cap: '+rel)
        if p.suffix in {'.py','.md','.txt','.json','.csv'}:
            content=p.read_text(encoding='utf-8-sig')
            if re.search(r'[A-Za-z]:[\\/]+Users[\\/]+[^\s"\']+',content):errors.append('Absolute user path: '+rel)
            if re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}',content):errors.append('Potential credential: '+rel)
            if re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',content):errors.append('Private key: '+rel)
            if p.suffix=='.py':
                try:ast.parse(content,filename=rel)
                except SyntaxError as exc:errors.append(str(exc))
        entries.append(dict(path=rel,bytes=size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    cases=json.loads((ROOT/'cases.json').read_text())
    if set(cases)!={'natural','ventilation','cpu_fan','cold_plate'}:errors.append('Unexpected case registry')
    for key,folder in cases.items():
        c=json.loads((ROOT/folder/'case.json').read_text())
        if c['id']!=key or c['method']['IBM'] or c['model']['power_W']!=200:errors.append('Case contract mismatch: '+key)
        for name in ['README.md','run.py','case.json','endpoint.png','summary.mp4']:
            if not (ROOT/folder/name).is_file():errors.append('Missing '+folder+'/'+name)
    if not (ROOT/'LICENSE').read_text().startswith('                    GNU GENERAL PUBLIC LICENSE'):errors.append('Original GPL license missing or changed')
    for p in (ROOT/'assets/geometry/am5_surrogate').glob('*.json'):
        for part in json.loads(p.read_text())['parts']:
            if not (p.parent/part['file']).is_file():errors.append('Missing STL '+part['file'])
    report=dict(status='failed' if errors else 'passed',file_count=len(entries),total_bytes=sum(e['bytes'] for e in entries),
        max_file_bytes=max(e['bytes'] for e in entries),cases=list(cases),errors=errors)
    if a.write_manifest and not errors:
        (ROOT/'PUBLICATION_MANIFEST.json').write_text(json.dumps(report|dict(files=sorted(entries,key=lambda e:e['path'])),indent=2)+'\n',newline='\n')
    print(json.dumps(report,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
