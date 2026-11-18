"""Plot a newly calculated scalar temperature history without editing its data."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('history',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=json.loads(a.history.read_text());fig,ax=plt.subplots(figsize=(10,5),facecolor='black');ax.set_facecolor('black')
    for field,color in [('CPU_C','#ffcc69'),('IHS_C','#ec929c'),('fins_C','#66dbe7'),('coolant_outlet_bulk_C','#66aaff')]:
        points=[(r['time_s'],r[field]) for r in rows if field in r]
        if points:ax.plot(*zip(*points),label=field.replace('_C',''),color=color)
    ax.axhline(95,color='#a77a7a',linestyle='--',label='95°C event');ax.tick_params(colors='white')
    ax.set_xlabel('Physical heat time [s]',color='white');ax.set_ylabel('Temperature [°C]',color='white')
    ax.legend(facecolor='black',labelcolor='white');fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=160);plt.close(fig)


if __name__=='__main__':main()
