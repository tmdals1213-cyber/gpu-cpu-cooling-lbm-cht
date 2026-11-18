"""Build a portable allowlisted ZIP outside the repository; no raw run data."""
from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
from verify_package import ROOT,payload_files


def main():
    subprocess.run([sys.executable,str(ROOT/'tools/verify_package.py'),'--write-manifest'],check=True)
    out=ROOT.parent/'dist';out.mkdir(exist_ok=True)
    target=out/'gpu-cpu-cooling-lbm-cht-portable.zip'
    files=list(payload_files())+[ROOT/'PUBLICATION_MANIFEST.json']
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,Path(ROOT.name)/p.relative_to(ROOT))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for p in files:
            member=(Path(ROOT.name)/p.relative_to(ROOT)).as_posix()
            assert hashlib.sha256(z.read(member)).digest()==hashlib.sha256(p.read_bytes()).digest()
    report=dict(file=target.name,bytes=target.stat().st_size,files=len(files),
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),crc_and_member_hashes='passed')
    (out/'portable_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
