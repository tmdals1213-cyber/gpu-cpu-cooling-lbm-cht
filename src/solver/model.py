# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path


import sys


import numpy as np


PROPERTIES={0:(1.184,1007.,[.0263]*3),1:(1850.,1000.,[10.,10.,.8]),
 2:(2330.,700.,[130.]*3),3:(2330.,700.,[130.]*3),4:(2330.,700.,[130.]*3),
 5:(8960.,385.,[390.]*3),6:(1.184,1007.,[.0263]*3),7:(2700.,900.,[200.]*3)}


CONTACTS={(2,5):2e-6,(3,5):2e-6,(4,5):2e-6,(5,7):2e-5}


ALL_FACES=tuple((axis,side) for axis in range(3) for side in range(2))


AIR=dict(name='Air',rho=1.184,cp=1007.,k=.0263,nu=1.562e-5,beta=1/298.15)


OIL=dict(name='Castrol ON DC20',rho=797.,cp=2080.,k=.135,nu=5.1e-6,beta=.00075,
    reference='rho at 15.6 C; cp,k,nu at 40 C; beta assumed; constant-property screening',
    source='https://www.castrol.com/content/dam/castrol/business-sites-new/en/global/corporate/documents/industries/data-centre-cooling/Castrol-ON-immersion-cooling-fluid-DC-20-product-data.pdf')


CPU=dict(reference='AMD Ryzen 9 9950X3D2 Dual Edition',reference_TDP_W=200.,Tjmax_C=95.,
    geometry='AM5-inspired equivalent solid model, not manufacturer internal CAD',
    power='Prescribed constant 200 W design scenario referenced to published TDP; not measured package power',
    source='https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x3d2-dual-edition.html')


FAN=dict(name='Noctua NF-A12x25 G2 PWM',diameter_mm=120.,rated_rpm=1800.,
    free_flow_m3_s=107.3/3600,shutoff_Pa=3.14*9.80665,
    curve='Parabolic interpolation of published endpoints; not a measured manufacturer curve',
    source='https://cdn.noctua.at/media/noctua_nf_a12x25_g2_pwm_infosheet_en_web.pdf')

