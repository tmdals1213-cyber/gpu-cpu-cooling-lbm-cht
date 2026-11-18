"""Case entry point. Default: print a plan without starting a long run."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[1]
raise SystemExit(subprocess.call([sys.executable,str(root/'tools/run_case.py'),'--case','cpu_fan',*sys.argv[1:]]))
