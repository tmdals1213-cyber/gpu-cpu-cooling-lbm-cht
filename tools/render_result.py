"""Inspection PNG from an actual endpoint field; no synthetic temperature data."""
from pathlib import Path
import argparse
import numpy as np
import pyvista as pv


def main():
    p=argparse.ArgumentParser();p.add_argument('field',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with np.load(a.field) as data:
        ids=data['materials'];half=data['T_half'];dx=float(data['dx'])*1000
    temperature=np.concatenate([half,half[::-1]],axis=0)
    if temperature.shape!=ids.shape:raise ValueError('Expected half-domain temperature and full material grid')
    grid=pv.ImageData(dimensions=np.array(ids.shape)+1,spacing=(dx,dx,dx))
    grid.cell_data['Temperature']=temperature.ravel(order='F')
    view=pv.Plotter(off_screen=True,window_size=(1600,900));view.set_background('black')
    active=np.flatnonzero(np.isin(ids,[1,2,3,4,5,7,8,10]).ravel(order='F'))
    body=grid.extract_cells(active).extract_surface()
    view.add_mesh(body,scalars='Temperature',cmap='turbo',clim=(25,100),show_edges=False,lighting=False,
                  scalar_bar_args={'title':'Solid temperature [C]','color':'white'})
    view.view_isometric();view.camera.zoom(1.1)
    cpu=float(temperature[np.isin(ids,[2,3,4])].max())
    view.add_text(f'CPU {cpu:.1f} C',position='upper_left',color='#ffcc69',font_size=22)
    a.output.parent.mkdir(parents=True,exist_ok=True);view.show(screenshot=str(a.output));view.close()


if __name__=='__main__':main()
