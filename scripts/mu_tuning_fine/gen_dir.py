import numpy as np
import os
import argparse
from shutil import copy2

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

    print(f"L*dt list = {np.array(dt_list)*L_list}")
    print(f"L list =  {Ls_list}")
    print(f"dt list = {dts_list}")
    return Ls_list, dts_list,dtaumax


def get_lists_from_file(args):
    Nx = args.Nx; Ny = args.Ny; 
    nflux = args.nflux; nt = args.nt; tp = args.tp
    
    betas_list = []
    Us_list = []
    mu_dict = {}

    #read master document that gives starting estimate mu
    fb = f"best_mu_{Nx}x{Ny}_tp{tp}_nflux{nflux}_n{nt}.txt"

    print(f"baseline file = {fb}")
    with open(fb,"r") as f:
        c = f.readlines();
        for i in range(len(c)):
            if "target n = " in c[i]:
                ntr = c[i][11:-1] #read from file, match spec
                assert ntr == nt
            if "beta" in c[i]:
                bi = c[i].find("beta");ui = c[i].find("U")
                bs = c[i][bi+4:ui-1]
                #if float(bs) != 6.0:
                    #continue
                Us = c[i][ui+1:-1]
                if bs not in betas_list:
                    betas_list.append(bs); 
                if Us not in Us_list:
                    Us_list.append(Us)

                mu = float(c[i-1][1:-2])
                mu_dict[(bs,Us)] = mu #full precision, float
    print(f"beta list = {betas_list}")
    print(f"U list = {Us_list}")
    return betas_list, Us_list, mu_dict

def generate_directories(args, betas_list, Us_list, mu_dict, Ls_list, dts_list):

    dmu = float(args.dmu)

    #---------------------------------------------------------
    #sim file master generation command, before replacements  
    cmd = ""
    with open("gen_sim_files.sh","r") as f:
        c = f.readlines()
        cmd = c[0]

    cmd = cmd.replace("VALSETGEOMETRY",args.geometry)
    cmd = cmd.replace("VALSETNX",args.Nx)
    cmd = cmd.replace("VALSETNY",args.Ny)
    cmd = cmd.replace("VALSETNFLUX",args.nflux)
    cmd = cmd.replace("VALSETTP",args.tp)

    #generate directories:
    for i in range(len(betas_list)):
        bs = betas_list[i]
        Ls = Ls_list[i]
        dts = dts_list[i]
        for Us in Us_list:
            mu = mu_dict[(bs,Us)] #full precision
            mu_list = [mu-2*dmu, mu-dmu, mu, mu+dmu, mu+2*dmu ] #full precision
            print(f"mu list = {mu_list}")
            #adjecent values of mu
            for m in mu_list:
                s = cmd.replace("VALSETL",Ls)
                s = s.replace("VALSETDT",dts)
                s = s.replace("VALSETU",Us)
                s = s.replace("VALSETMU",f"{m}")
                os.makedirs(f"{Nx}x{Ny}_tp{tp}_nflux{nflux}/n{nt}/beta{bs}_U{Us}/mu{m:.5f}",exist_ok=False)
                with open(  f"{Nx}x{Ny}_tp{tp}_nflux{nflux}/n{nt}/beta{bs}_U{Us}/mu{m:.5f}/gen_sim_files.sh","w") as f:
                    f.write(s)

def copy_scripts(args, dtaumax):
    #command for getting and outputting optimal mu, need substitutions
    geometry = args.geometry; Nx = args.Nx; Ny = args.Ny; 
    nflux = args.nflux; nt = args.nt; tp = args.tp
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
            ll[i] = ll[i].replace("VALSETDMU",args.dmu)
        mucmd = ll

    with open( f"{Nx}x{Ny}_tp{tp}_nflux{nflux}/n{nt}/get_mu.sh","w") as f:
        for l in mucmd:
            f.write(l)  

    copy2('sweep_state.sh',f"{Nx}x{Ny}_tp{tp}_nflux{nflux}/n{nt}/")
    
def main(args):
    #get betas and Us lists from file
    betas_list, Us_list, mu_dict = get_lists_from_file(args)
    Ls_list, dts_list, dtaumax =  get_L_dt(betas_list)
    generate_directories(args, betas_list, Us_list, mu_dict, Ls_list, dts_list)
    copy_scripts(args, dtaumax)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('geometry', choices=['square', 'triangular', 'honeycomb', 'kagome']);
    parser.add_argument("Nx", help = "integer, lattice x dimension")
    parser.add_argument("Ny", help = "integer, lattice y dimension")
    parser.add_argument("nflux", help = "integer, number of flux quantum through lattice")
    parser.add_argument("nt", help = "float in [0,2], target filling level")
    parser.add_argument("tp", help = "float, tp = next nearest neighbor hopping strength")
    parser.add_argument("dmu", help = "float, max deviation from prev iter mu values")
    args = parser.parse_args()

    geometry = args.geometry; Nx = args.Nx; Ny = args.Ny; 
    nflux = args.nflux; nt = args.nt; tp = args.tp
    dmu = float(args.dmu)
    print(f"set geometry={geometry} Nx = {Nx} Ny = {Ny} nflux = {nflux} target n = {nt} tp = {tp} dmu = {dmu}")
    #print(args)
    main(args)

