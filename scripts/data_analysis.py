import numpy as np
np.set_printoptions(precision=4)
import matplotlib.pyplot as plt
import sys
import glob
import os
# src = os.environ['DEV'] 
# if not src+"util/" in sys.path:
#     sys.path.insert(0,src+"util/")
from pathlib import Path
utilpath = Path(__file__).resolve().parents[1]/"util"
sys.path.insert(0, str(utilpath))
import util
from itertools import product


plaq_per_cell_dict = {}
plaq_per_cell_dict['square'] = 1;
plaq_per_cell_dict['triangular'] = 2;
plaq_per_cell_dict['honeycomb'] = 1;
plaq_per_cell_dict['kagome'] = 2;

def real_to_momentum(geometry,dat):
    """
    Discrete Fourier Transforms real space numpy array dat 
    of shape (nbin,[L,]Nx,Ny) into its momentum space counterpart
    
    Args:
        geometry (string): [description]
        dat (numpy.ndarray): DQMC measurement. dtype complex
    
    Raises:
        NotImplementedError: [description]
    """

    if geometry == "square":
        #square lattice unit vectors
        a1 = np.array((1,0))
        a2 = np.array((0,1))
        b1 = np.array((2*np.pi,0))
        b2 = np.array((0, 2*np.pi))
    elif geometry == "triangular" or geometry == "honeycomb" or geometry == 'kagome':
        #triangular lattice unit vectors
        a1 = np.array((1,0))
        a2 = np.array((0.5,np.sqrt(3)/2))     #/_ space cell
        b1 = np.array((2*np.pi, -2*np.pi/np.sqrt(3)))   #|  Momentum cell
        b2 = np.array((0, 4*np.pi/np.sqrt(3)))          # \ 
    else:
        raise NotImplementedError('Invalid geometry')

    assert dat.ndim == 3 or dat.ndim==4

    Nx = dat.shape[-2]
    Ny = dat.shape[-1] 

    #print("dat shape",dat.shape)

    #all allowed values of momentum
    dat_q_all = np.full(dat.shape,np.nan,dtype=complex)
    for (kix,kiy) in product(range(Nx),range(Ny)):
        k1 = kix / Nx * b1
        k2 = kiy / Ny * b2
        k = k1+k2 #wave vector
        phasemat = np.zeros((Nx,Ny),dtype=float)

        for (dx,dy) in product(range(Nx),range(Ny)):
            loc = dx * a1 + dy * a2
            phase = np.dot(k,np.array(loc))
            phasemat[dx,dy] = phase

        dat_q = np.sum(np.exp(-1j*phasemat)*dat,axis = (-2,-1))

        if dat.ndim == 3:
            dat_q_all[:,kix,kiy] = dat_q
        elif dat.ndim == 4:
            dat_q_all[:,:,kix,kiy] = dat_q

    return dat_q_all

def cv(beta_arr,E_arr,E_err_arr):
    assert beta_arr.shape == E_arr.shape
    #check: imaginary part is small, or no data
    assert np.nanmax(np.abs(E_arr.imag/E_arr.real)) < 1e-4 \
        or np.all(np.isnan(E_arr))

    #print(np.nanmax(np.abs(E_arr.imag/E_arr.real)) or np.all(np.isnan(E_arr)))

    temp_mid = 1/2*(1/beta_arr[1:] + 1/beta_arr[:-1])
    temp_diff = 1/beta_arr[1:] - 1/beta_arr[:-1]

    dE = E_arr[1:].real - E_arr[:-1].real

    dE_err = np.sqrt(E_err_arr[1:].real**2 + E_err_arr[:-1].real**2)

    return temp_mid, dE/temp_diff, -dE_err/temp_diff

def jackknife(asgn,sgn):
    """this jackknife function keeps track of imaginary parts,
    for real measurements, gives same estimator as edwin's jackknife,
    except includes imaginary parts. deosn't allow arbitrary function input
    like edwin's jackknife function"""
    #print(asgn.shape,sgn.shape)
    assert asgn.shape[0] == sgn.shape[0], f"{asgn.shape},{sgn.shape}"
    #0 axis is nbin
    nbin = sgn.shape[0]

    if nbin == 1:
        print("only one bin, thus zero error")
        return asgn[0]/sgn[0] , np.zeros(asgn.shape[1:])

    jk_resample = np.full(asgn.shape,np.nan,dtype=complex)

    for j in range(nbin):
        jk_resample[j] = (np.sum(asgn,axis=0)-asgn[j]) / (np.sum(sgn)-sgn[j])
    jk_mean = np.mean(jk_resample,axis=0)
    jk_variance = np.mean(np.abs(jk_resample - jk_mean)**2,axis=0)
    jk_std = np.sqrt((nbin-1)*jk_variance)

    return jk_mean, jk_std

def info(path,uneqlt=False,show=False,imagtol=1e-2):
    #TODO: allow working with Wen's code, where there is no nflux parameter
    '''return integer based on if the runs are complete
    
    If return 1, then directory doesn't have any completed MC runs
    If return 2, then imaginary parts of sign or density estimator too large
    If return -1, then not enough bins for maxent
    If return 0, then this directory has OK data, can proceed.
    '''
    
    #if no *h5 files, likely pathname is wrong
    nfiles = len(glob.glob(path+'*.h5'))
    #print("nfiles = ", nfiles)
    if nfiles == 0:
        print(f"\033[4mNo *h5 files in {path}\033[0m")
        return 1

    U, beta, mu, Nx, Ny, bps,nflux, tp, N, L, dt, n_sweep_warm, n_sweep_meas, \
    period_uneqlt, period_eqlt, meas_energy_corr, meas_bond_corr = \
        util.load_firstfile(path,"metadata/U","metadata/beta","metadata/mu",\
            "metadata/Nx","metadata/Ny","metadata/bps","metadata/nflux","metadata/t'",\
            "params/N","params/L","params/dt","params/n_sweep_warm","params/n_sweep_meas",
            "params/period_uneqlt","params/period_eqlt","params/meas_energy_corr","params/meas_bond_corr")


    if show:
        print(path)
        print(f"{Nx}x{Ny}    tp={tp}    U={U}   nflux={nflux}    beta={beta:.3g}    dt={dt:.3g}\n"+\
            f"mu={mu:.3f}\tn_sweep_warm={n_sweep_warm}\tn_sweep_meas={n_sweep_meas}\t"+\
            f"period_eqlt={period_eqlt}\tnbins target={nfiles}\n"+\
            f"Uneqlt period? {period_uneqlt}\tEnergy corr? {bool(meas_energy_corr)}\t"+\
            f"Bond corr? {bool(meas_bond_corr)}")
        try: 
            meas_2bond, meas_thermal = \
                util.load_firstfile(path,"params/meas_2bond_corr","params/meas_thermal")
            print(f"2 bond corr? {bool(meas_2bond)}\t"+\
                f"thermal corr? {bool(meas_thermal)}")
        except:
            print("At least one of 2 bond or thermal measurements toggled to False")

    #calculate expected n_sample 
    if uneqlt:
        ns, s = util.load(path, "meas_uneqlt/n_sample", "meas_uneqlt/sign")
        ns_expect = n_sweep_meas//period_uneqlt
    else:
        ns, s, d = util.load(path, "meas_eqlt/n_sample", "meas_eqlt/sign","meas_eqlt/density")
        ns_expect = n_sweep_meas*L//period_eqlt

    #no data or incomplete
    if ns.max() < ns_expect:
        print(f"\033[4mshould have {ns_expect} data, actual: {ns.max()}\033[0m")
        return 1

    if ns.max() > ns_expect:
        print(f"\033[91mshould have {ns_expect} data, actual: {ns.max()}, need check\033[91m")
        raise ValueError

    #at least one bin is full, print info
    nbin_all = ns.shape[0]
    mask = ns == ns.max(); nbin = mask.sum();

    #use only filled bins by applying mask
    ns,s = ns[mask],s[mask]
    avgsgn, sgn_stderr = jackknife(s,ns)
    #print sign (and density) real part and its error
    if uneqlt:
        if show:
            colorize = "\033[93m " if nbin < nbin_all else "\033[0m "
            print(f"uneqlt complete: {colorize} {nbin}/{nbin_all} \033[0m\ts={avgsgn:.3g} SE(s)={sgn_stderr:.3g}")
    else:
        d=d[mask]
        avgd, d_stderr = jackknife(d,s)
        avgd = np.mean(avgd)
        d_stderr = np.mean(d_stderr)
        
        if show:
            colorize = "\033[93m " if nbin < nbin_all else "\033[0m "
            print(f"eqlt complete: {colorize} {nbin}/{nbin_all}\033[0m",
                f"\t<sign>={avgsgn:.3g} SE(<sign>)={sgn_stderr:.3g}",
                f"\x1b[1m  n={avgd:.3g} SE(n)={d_stderr:.3g} \x1b[0m")
        # imaginary part of density mean is too large
        if np.abs(avgd.imag)/np.abs(avgd.real) > imagtol: 
            print(f"\033[91md imag/real norm = {np.abs(avgd.imag)/np.abs(avgd.real)} > {imagtol}\033[0m")
            return 2
        # standard error of density mean estimator is too large
        if d_stderr/np.abs(avgd) > imagtol*5:
            print(f'\033[91mSE(d)/abs(d) = {d_stderr/np.abs(avgd):.3g} > {imagtol*5}\033[0m')
            return 2

    # imaginary part of sign estimator is too large
    if np.abs(avgsgn.imag)/np.abs(avgsgn.real) > imagtol: 
        print(f"\033[91ms imag/real norm = {np.abs(avgsgn.imag)/np.abs(avgsgn.real)} > {imagtol}\033[0m")
        return 2

    # standard error of sign mean estimator is too large
    if sgn_stderr/np.abs(avgsgn) > imagtol*5:
        print(f'\033[91mSE(s)/abs(s) = {sgn_stderr/np.abs(avgsgn):.3g} > {imagtol*5}\033[0m')
        return 2

    #Check for maxent
    if uneqlt and nbin <= 2*L :
        print(f"\033[93m{nbin}/{nbin_all} bins not sufficient for maxent \033[0m")
        return -1

    return 0

def infer_metadata(path):
    """For backwards compatibility with older DQMC versions.
    Infer Norb, plaq_per_cell, trans_sym, geometry. Issue warnings
    if no inference can be made
    
    [description]
    
    Args:
        path ([type]): [description]
    
    Returns:
        (int,int) : Norb, trans_sym
    """

    Nx, Ny, num_i= util.load_firstfile(path,\
        "metadata/Nx","metadata/Ny","params/num_i")

    #Issue warnings when processing old data
    try:
        Norb = util.load_firstfile(path,"metadata/Norb")[0]
    except KeyError as e:
        Norb = num_i if num_i < Nx * Ny else num_i/(Nx*Ny)
        print(f"\033[93mWarning\033[0m: No Norb info saved in metadata, inferring from num_i: Norb={Norb}")
    
    try:
        plaq_per_cell = util.load_firstfile(path,"metadata/plaq_per_cell")[0]
    except KeyError as e:
        plaq_per_cell = num_i if num_i < Nx * Ny else num_i/(Nx*Ny)
        print("\033[93mWarning\033[0m: No plaq_per_cell info saved in metadata, plaq_per_cell unknown")

    try:
        trans_sym = util.load_firstfile(path,"metadata/trans_sym")[0]
    except KeyError as e:
        trans_sym = not (num_i >= Nx * Ny)
        print(f"\033[93mWarning\033[0m: No trans_sym toggle saved in metadata, inferring from num_i: {trans_sym}")
    
    try:
        geometry = util.load_firstfile(path,"metadata/geometry")[0]
    except KeyError as e:
        print("\033[93mWarning\033[0m: No geometry info saved in metadata, geometry unknown")
    
    return Norb,trans_sym

#site-site correlator
def eqlt_meas_Nx_Ny(path, meas_list,geometry="square"):

    if info(path,uneqlt=False,show=False) == 1 : 
        raise OSError(f"No completed MC bin")

    Norb, trans_sym = infer_metadata(path)

    Nx, Ny, U, tp, beta = util.load_firstfile(path,\
        'metadata/Nx','metadata/Ny','metadata/U','metadata/t\'',"metadata/beta")
    ns, s =  util.load(path, "meas_eqlt/n_sample", "meas_eqlt/sign")
    #at least one bin is full, print info
    #nbin_all = ns.shape[0]
    mask = ns == ns.max(); nbin = mask.sum();
    dm_dict = {}; de_dict = {};
    for meas_name in meas_list:
        #      y
        # Ny-1  |
        #     . |
        #     . |
        #     4 |________ (pi,pi)
        #     3 |        |
        #     2 |        |
        #     1 |        |
        #     0 |________|_______ x
        #       0 1 2 3 4 5 ... Nx-1
        #    (0,0)

        if meas_name.endswith("_q"):
            assert Nx == Ny and trans_sym and Norb == 1
            s, dat = util.load(path, "meas_eqlt/sign",f"meas_eqlt/{meas_name[:-2]}")
            s, dat = s[mask],np.squeeze(dat[mask])
            dat = np.reshape(dat,(-1,Nx,Ny),order='F')

            # sus_val = np.full((Nx,Ny),np.nan,dtype=complex)
            # sus_err = np.full((Nx,Ny),np.nan,dtype=float)

            dat_q_all = real_to_momentum(geometry,dat)
            sus_val,sus_err = jackknife(dat_q_all,s)

            datm,date = sus_val,sus_err
        else:
            s, dat= util.load(path, "meas_eqlt/sign",f"meas_eqlt/{meas_name}")
            #use only filled bins by applying mask
            s, dat = s[mask],dat[mask]
            datm,date = jackknife(dat,s) 
            if trans_sym:
                if Norb == 1:
                    datm = np.reshape(datm,(Nx,Ny),order='F')
                    date = np.reshape(date,(Nx,Ny),order='F')
                else:
                    datm = np.reshape(datm,(Nx,Ny,Norb,Norb),order='F')
                    date = np.reshape(date,(Nx,Ny,Norb,Norb),order='F')
            else:
                if Norb == 1:
                    datm = np.reshape(datm,(Nx*Ny,Nx*Ny),order='F')
                    date = np.reshape(date,(Nx*Ny,Nx*Ny),order='F')
                else:
                    datm = np.reshape(datm,(Nx*Ny,Norb,Nx*Ny,Norb),order='F')
                    date = np.reshape(date,(Nx*Ny,Norb,Nx*Ny,Norb),order='F')
        
        dm_dict[meas_name] = datm
        de_dict[meas_name] = date
        
    return dm_dict,de_dict

#per plaquette measurement
def eqlt_meas_plaq(path,meas_list,geometry='square'):
    if info(path,uneqlt=False,show=False) == 1 : 
        raise OSError(f"No completed MC bin")

    Norb, trans_sym = infer_metadata(path)
    
    Nx, Ny, U, tp, beta = util.load_firstfile(path,\
        'metadata/Nx','metadata/Ny',\
        'metadata/U','metadata/t\'',"metadata/beta")

    try:
        plaq_per_cell = util.load_firstfile(path,"metadata/plaq_per_cell")[0]
        print("loaded plaq_per_cell = ",plaq_per_cell)
    except KeyError:
        plaq_per_cell = plaq_per_cell_dict[geometry]
        print(f"manually set plaq_per_cell based on {geometry} geometry = ",plaq_per_cell)

    ns, s =  util.load(path, "meas_eqlt/n_sample", "meas_eqlt/sign")

    #at least one bin is full, print info
    #nbin_all = ns.shape[0]
    mask = ns == ns.max(); nbin = mask.sum();
    dm_dict = {}; de_dict = {};

    ns,s = ns[mask],s[mask]
    for meas_name in meas_list:
        if meas_name == "chi":
            #if not trans_sym: raise NotImplementedError(meas_name)
            dat= util.load(path,f"meas_eqlt/{meas_name}")[0]
            dat= dat[mask]
            #dat = dat if trans_sym else np.reshape(dat,(-1,Nx,Ny),order='F') 
            print(dat.shape)
            datm,date = jackknife(dat,s)
        else:
            print(meas_name)
            raise NotImplementedError
        dm_dict[meas_name] = datm
        de_dict[meas_name] = date
        
    return dm_dict,de_dict

#per Norb measurement
def eqlt_meas_1(path,meas_list,geometry = "square"):
    """Given a list of measurement names and data path directory, returns: 
    A tuple of dictionaries (dm,de). dm contains the measurement values, 
    de contains the measurement errors"""
    
    if info(path,uneqlt=False,show=False) == 1 : 
        raise OSError(f"No completed MC bin")

    Norb, trans_sym = infer_metadata(path)

    Nx, Ny, U, tp, beta = util.load_firstfile(path,'metadata/Nx','metadata/Ny',\
        'metadata/U','metadata/t\'',"metadata/beta")

    ns, s =  util.load(path, "meas_eqlt/n_sample", "meas_eqlt/sign")
    #at least one bin is full, print info
    #nbin_all = ns.shape[0]
    mask = ns == ns.max(); nbin = mask.sum();
    dm_dict = {}; de_dict = {};

    ns,s = ns[mask],s[mask]
    for meas_name in meas_list:
        
        if meas_name == "mz":
            d, double_occ= util.load(path, "meas_eqlt/density","meas_eqlt/double_occ")
            #use only filled bins by applying mask
            d, double_occ = (np.squeeze(d[mask],axis=1),np.squeeze(double_occ[mask],axis=1)) \
                if (trans_sym and Norb ==1) else (d[mask],double_occ[mask])
            datm,date  = jackknife(d-2*double_occ,s)

        elif "density" in meas_name:
            dat = util.load(path, f"meas_eqlt/{meas_name}")[0]
            #use only filled bins by applying mask
            dat = np.squeeze(dat[mask],axis=1) if (trans_sym and Norb ==1) else dat[mask]
            datm,date  = jackknife(dat,s)

        elif meas_name == "sign":
            datm, date = jackknife(s,ns)

        elif meas_name == "energy":
            if not (trans_sym and Norb == 1): raise NotImplementedError(meas_name)
            d, g00, double_occ  = \
            util.load(path,'meas_eqlt/density',\
                      "meas_eqlt/g00", "meas_eqlt/double_occ")
            g00 = np.reshape(g00,(-1,Nx,Ny),order='F')
            #use only filled bins by applying mask
            d,double_occ,g00 = np.squeeze(d[mask],axis=1),\
                np.squeeze(double_occ[mask],axis=1),g00[mask]
            #kinetic terms = t1 + t2
            if geometry == "square":
                #factor of 2 appears because g00 = 0.5*(gup + gdn), I think
                t1 = 2*(g00[:,0,1]+g00[:,1,0]+g00[:,0,Ny-1]+g00[:,Nx-1,0]) 
                #TODO: check if tp terms are correct
                t2 = 2*tp*(g00[:,1,1] + g00[:,1,Ny-1] + g00[:,Nx-1,1]+ g00[:,Nx-1,Ny-1])
            elif geometry == "triangular":
                #factor of 2 appears because g00 = 0.5*(gup + gdn), I think
                t1 = 2*(g00[:,0,1]+g00[:,1,0]+g00[:,0,Ny-1]+g00[:,Nx-1,0] + g00[:,Nx-1,1] + g00[:,1,Ny-1]) 
                #TODO: add tp terms
                t2 = 0
            else:
                raise NotImplementedError(meas_name)
            #potential energy = t3
            t3 = U*(double_occ)
            datm,date = jackknife(t1+t2+t3,s)
            #datm += 1/4*U
            
        elif meas_name =="kinetic":
            if not (trans_sym and Norb==1): raise NotImplementedError(meas_name)
            g00 = util.load(path,"meas_eqlt/g00")[0]
            g00 = np.reshape(g00,(-1,Nx,Ny),order='F')
            #use only filled bins by applying mask
            g00 = g00[mask]
            #kinetic terms = t1 + t2
            if geometry == "square":
                #factor of 2 appears because g00 = 0.5*(gup + gdn), I think
                t1 = 2*(g00[:,0,1]+g00[:,1,0]+g00[:,0,Ny-1]+g00[:,Nx-1,0]) 
                #TODO: check if tp terms are correct
                t2 = 2*tp*(g00[:,1,1] + g00[:,1,Ny-1] + g00[:,Nx-1,1]+ g00[:,Nx-1,Ny-1])
            elif geometry == "triangular":
                #factor of 2 appears because g00 = 0.5*(gup + gdn), I think
                t1 = 2*(g00[:,0,1]+g00[:,1,0]+g00[:,0,Ny-1]+g00[:,Nx-1,0] + g00[:,Nx-1,1] + g00[:,1,Ny-1]) 
                #TODO: add tp terms
                t2 = 0
            else:
                raise NotImplementedError(meas_name)
            datm,date = jackknife(t1+t2,s)

        elif meas_name =="potential":
            if not (trans_sym and Norb==1): raise NotImplementedError(meas_name)
            d, double_occ  = util.load(path,'meas_eqlt/density',\
                "meas_eqlt/double_occ")
            #use only filled bins by applying mask
            d,double_occ= np.squeeze(d[mask],axis=1),\
                np.squeeze(double_occ[mask],axis=1)

            #potential energy = t3
            t3 = U*(double_occ)
            datm,date = jackknife(t3,s)

        elif meas_name == "compress":
            if not (trans_sym and Norb == 1): raise NotImplementedError(meas_name)
            # special case: use Edwin's jackknife function, so return avg of type float
            nn, dens= util.load(path,"meas_eqlt/nn","meas_eqlt/density")
            nn,dens = np.squeeze(nn[mask]),np.squeeze(dens[mask],axis=1)
            nn = np.reshape(nn,(-1,Nx,Ny),order='F')
            nn_q0 = np.sum(nn,axis=(1,2)) 
            tryf = lambda s, sx, sy: beta*((sx.T/s.T).T.real - (sy.T/s.T).T.real**2 * Nx * Ny)
            datm,date = util.jackknife(s,nn_q0,dens,f=tryf)
        else:
            raise NotImplementedError(meas_name)
        dm_dict[meas_name] = datm
        de_dict[meas_name] = date
        
    return dm_dict,de_dict


def readmu(path,fname,show=False,nflux_string="?"):
    betas_list = []; 
    Us_list = [];
    mu_dict = {};

    print(f"mu info source: {path}{fname}")
    with open(path+fname,"r") as f:
        c = f.readlines();
        for i in range(len(c)):
            if "target n = " in c[i]:
                nt = float(c[i][10:-1])
            if "beta" in c[i]:
                bi = c[i].find("beta");ui = c[i].find("U")
                bs = c[i][bi+4:ui-1];
                Us = c[i][ui+1:-1]
                if bs not in betas_list:
                    betas_list.append(bs); 
                if Us not in Us_list:
                    Us_list.append(Us)
                mu = float(c[i-1][1:-2])
                mu_dict[(bs,Us)] = mu #full precision float
    

    beta_arr = np.array(list(map(float,betas_list))); order = np.argsort(beta_arr)
    U_arr = np.array(list(map(float,Us_list))); Uorder = np.argsort(U_arr)
    nT = beta_arr.shape[0]; nU = U_arr.shape[0]
    #if want to plot chemical potential as function of temperature for each U
    if show:
        print(f"target filling = {nt}")
        print(f"beta list = {betas_list}")
        print(f"U list = {Us_list}")
        #colors = plt.cm.viridis(np.linspace(1,0,nU))
        for j in Uorder:
            Us = Us_list[j]
            mu_arr = np.empty(nT)
            for i in range(nT):
                try:
                    mu_arr[i] = mu_dict[(betas_list[i],Us)]
                except KeyError as e:
                    print("KeyError: (beta, U) = ", e)
                    mu_arr[i] = np.nan
            plt.plot(1/beta_arr[order],mu_arr[order],'.-',label=f"U={Us}, nflux={nflux_string}")
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.xlabel(r'temperature $T/t$')
        plt.ylabel(r"chemical potential $\mu$")
        plt.grid(True)
        plt.xscale('log')
        plt.xlim(np.min(1/beta_arr)/2,np.max(1/beta_arr)*2)

    return betas_list, Us_list, mu_dict