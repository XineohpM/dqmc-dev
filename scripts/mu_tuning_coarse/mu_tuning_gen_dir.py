import numpy as np
import os
import argparse
import re
import ast
from shutil import copy2

def parse_num_list(s, typ=float):
    """
    Parse the string s into a list of numbers.
    The format of s should be as follows:
    1) separated by comma/space ("1,2,3" or "1 2 3")
    2) json list ("[1, 2, 3]")
    3) interval "start:stop:step" ("1:3:1"="1,2,3")
    """
    if s is None:
        raise ValueError("Expected a non-empty string (e.g., --betas 1,2,3). Got: None")
    s = str(s).strip()

    # Try json list
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)): return [typ(x) for x in v]
    except Exception: pass

    #Try interval
    if ":" in s:
        parts = [float(x) for x in s.split(":")]
        if len(parts) != 3: raise ValueError("The format of interval should be \"start:stop:step\"")
        start, stop, step = parts
        if step == 0:
            raise ValueError("step cannot be 0")
        n = int(round((stop - start) / step)) + 1
        return [typ(start + i * step) for i in range(n)]
    
    #Default: separated by comma/space ("1,2,3" or "1 2 3")
    tokens = re.split(r"[,\s]+", s)
    tokens = [t for t in tokens if t]
    return [typ(x) for x in tokens]

def get_L_dt(betas_list):
    barr = np.array(list(map(float,betas_list)))
    dts_list = []
    Ls_list = []
    
    for (i,b) in enumerate(barr):
        if float(b) < 0.7:
            dt = str(float(b)/10)
        elif float(b) == 0.7:
            dt = str(0.035)
        else:
            dt = str(0.05)
        #dt = dts_list[i]
        dts_list.append(dt)
        Ls_list.append(f"{b/float(dt):.3g}")
    L_list = list(map(float,Ls_list))
    dt_list = list(map(float,dts_list))
    dtaumax = f"{np.max(dt_list):.3g}"
    return Ls_list, dts_list, L_list, dt_list, dtaumax

def generate_directories(args, betas_list, Us_list, mu_list, Ls_list, dts_list):
    # generate directories and sim files based on user input and parameter lists
    #User input
    geometry=args.geometry; Nx = args.Nx; Ny = args.Ny; 
    nflux = args.nflux
    tp = args.tp
    nt = args.nt

    #------------------------------------------------------
    #command for generating sim files, need substitutions
    cmd=""
    with open("gen_sim_files.sh","r") as f:
        c = f.readlines()
        cmd = c[0]

    cmd = cmd.replace("VALSETGEOMETRY",geometry)
    cmd = cmd.replace("VALSETNX",Nx)
    cmd = cmd.replace("VALSETNY",Ny)
    cmd = cmd.replace("VALSETNFLUX",nflux)
    cmd = cmd.replace("VALSETTP",tp)

    #------------------------------------------------------
    #generate directories:
    for i in range(len(betas_list)):
        bs = betas_list[i]
        Ls = Ls_list[i]
        dts = dts_list[i]
        for Us in Us_list:
            for m in mu_list:
                s = cmd.replace("VALSETL",Ls)
                s = s.replace("VALSETDT",dts)
                s = s.replace("VALSETU",str(Us))
                s = s.replace("VALSETMU",f"{m}")
                os.makedirs(f"{Nx}x{Ny}_tp{tp}_nflux{nflux}/n{nt}/beta{bs}_U{Us}/mu{m:.5f}",exist_ok=False)
                with open(f"{Nx}x{Ny}_tp{tp}_nflux{nflux}/n{nt}/beta{bs}_U{Us}/mu{m:.5f}/gen_sim_files.sh","w") as f:
                    f.write(s)

def copy_scripts(args, dtaumax):
    #User input
    geometry=args.geometry
    Nx = args.Nx; Ny = args.Ny; 
    nflux = args.nflux; tp = args.tp; nt = args.nt
    #------------------------------------------------------
    #command for getting and outputting optimal mu, need substitutions
    mucmd = ""
    with open("get_mu.sh","r") as f:
        ll = f.readlines()
        for i in range(len(ll)):
            ll[i] = ll[i].replace("VALSETGEOMETRY",geometry)
            ll[i] = ll[i].replace("VALSETNX",Nx)
            ll[i] = ll[i].replace("VALSETNY",Ny)
            ll[i] = ll[i].replace("VALSETNT",nt)
            ll[i] = ll[i].replace("VALSETNFLUX",nflux)
            ll[i] = ll[i].replace("VALSETTP",tp)
            ll[i] = ll[i].replace("DTAUMAX",dtaumax)
        mucmd = ll

    with open( f"{Nx}x{Ny}_tp{tp}_nflux{nflux}/n{nt}/get_mu.sh","w") as f:
        for l in mucmd:
            f.write(l)      

    # copy files without modification
    copy2('sweep_state.sh',f"{Nx}x{Ny}_tp{tp}_nflux{nflux}/n{nt}/")

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('geometry', choices=['square', 'triangular', 'honeycomb', 'kagome'])
    parser.add_argument("Nx", help = "integer, lattice x dimension")
    parser.add_argument("Ny", help = "integer, lattice y dimension")
    parser.add_argument("nflux", help = "integer, number of flux quantum through lattice")
    parser.add_argument("nt", help = "float in [0,2], target filling level")
    parser.add_argument("tp", help = "float, tp = next nearest neighbor hopping strength")
    parser.add_argument("--betas", "--betas_list", dest="betas", type=str, required=True,
                        help="beta list, should be formatted as one of the following: \"1,2,3\", \"1 2 3\", \"[1, 2, 3]\" or \"start:stop:step\"")
    parser.add_argument("--Us", "--Us_list", dest="Us", type=str, required=True,
                        help="U list, should be formatted as beta list")
    parser.add_argument("--mus", "--mus_list", dest="mus", type=str, required=True,
                        help="mu list, should be formatted as beta list")
    return parser

def main(): 
    parser = _parse_args()
    args = parser.parse_args()
    betas_list = parse_num_list(args.betas, float)
    Us_list = parse_num_list(args.Us, float)
    mu_list = parse_num_list(args.mus, float) # mu values for coarse grid search
    Ls_list, dts_list, L_list, dt_list,dtaumax= get_L_dt(betas_list)

    print(f"set geometry={args.geometry} Nx={args.Nx} Ny={args.Ny} nflux={args.nflux} target n={args.nt} tp={args.tp}")
    
    #SANITY CHECK
    print(f"betas list = {betas_list}")
    print(f"L*dt list =  {np.array(L_list)*np.array(dt_list)}")
    print(f"Ls list =   {Ls_list}")
    print(f"dts list =  {dts_list}")
    print(f"U list = {Us_list}")
    print(f"mu_list = {mu_list}")
    assert len(dts_list) == len(betas_list) == len(Ls_list)
    
    #generate directories and helper scripts
    generate_directories(args, betas_list, Us_list, mu_list, Ls_list, dts_list)
    copy_scripts(args, dtaumax)

if __name__ == "__main__":
    main()
