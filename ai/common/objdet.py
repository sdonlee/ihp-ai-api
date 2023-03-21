import numpy as np

def get_colors(labels, nskip:int = 4):
    
    ngrid_color = int(np.ceil(len(labels)**(1/3))+1)
    colorgap = int(np.floor(256/ngrid_color))
    
    colors = [[colorgap*ir,colorgap*ig,colorgap*ib] for ir in range(ngrid_color) for ig in range(ngrid_color) for ib in range(ngrid_color)]
    colors = [colors[i] for i in sorted(range(len(colors)), key=lambda k: sum(colors[k]))]
    
    colors = colors[nskip:nskip+len(labels)]
    
    return colors