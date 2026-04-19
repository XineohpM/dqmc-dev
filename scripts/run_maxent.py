import maxent

from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt
import traceback

def perform_maxent(chi,  omega_grid, metadata, 
                   append=None,
                   bs=1, anneal_arr = None, checks=False, op_type='boson', sym=True, 
                   **mkwargs):
        """Performs MaxEnt on correlations of the form O(tau)O^{dagger}. Wrapper for maxent module. 
        Args: 
            chi: (Nbin,L) lhs of the imaginary time data to invert
            omega_grid: tuple containing (omega, domega) arrays for the frequency grid
            metadata: dictionary with metadata including "dt", "beta", "L", "nbin"
        Keyword Args:
            bs: number of bootstrap samples to perform
            append: (Nbin,1) array to append as the tau=beta component of G
            anneal_arr: initial model array to use in maxent. If None, uses flat model.
            checks: if True, plots bootstrap results for visual inspection
            op_type: 'boson' or 'fermion' kernel type
            sym: if True, uses bosonic kernel symmetrized about omega = 0
            mkwargs: additional keyword arguments to pass to maxent.MaxEnt function

        Returns: 
            A dictionary with components  
                "A": (bs, N_omega) array with best estimate of A(omega_i) * domega_i
                "s": (bs, N_omega) array with Re[L_{OO^{\dagger}}](\omega) with all bootstrap elements 
        """
        
        dt = metadata["dt"]
        beta = metadata["beta"]
        nbin = metadata["nbin"]

        if sym and (append is None):
            append = np.zeros((nbin,1),dtype=float)
        elif (not sym) and (append is None):
            raise ValueError("Must provide append array at tau=beta with correct symmetries for nonsymmetric bosonic kernel")

        # drop last row/column for bosonic non-sym kernel after preprocessing for maxent
        drop = True if not sym and op_type == "boson" else False 
        omega, domega = omega_grid
        nw = omega.shape[0] 

        # default to flat model 
        if anneal_arr is None:
            anneal_arr = maxent.model_flat(domega)
        
        s_bs = np.full((bs,nw),np.nan,dtype=float)
        A_bs = np.full((bs,nw),np.nan,dtype=float)

        for i in tqdm(range(bs)): # progress bar looping over bootstraps
            try:
                resample = np.random.randint(nbin,size=nbin) #sample with replacement

                # preprocess data for maxent
                pre = maxent.Preprocess(chi[resample], dt, beta, grid_info = (omega,domega),
                                        op_type = op_type, sym=sym, model_arr = anneal_arr, append=append[resample])
                
                # drop extra datapoint if nonsymmetric bosonic kernel
                if drop:
                    pre["tau"] = pre["tau"][:-1]
                    pre["lhs"] = pre["lhs"][:,:-1]
                    pre["K"] = pre["K"][:-1,:]

                #  best estimate of A(omega_i) *    domega_i
                A = maxent.MaxEnt(pre, **mkwargs)
                s = (A/domega)*pre["norm"]*np.pi 
                A_bs[i,:] = A
                s_bs[i,:] = s
            except:
                # Maxent failed, fill with NaNs
                A_bs[i,:] = np.NaN
                s_bs[i,:] = np.NaN
                traceback.print_exc()

        if checks: 
            L = metadata["L"]
            plt.figure()
            plt.ylabel(r"$L(\omega)$ bootstrap")
            plt.plot(omega,s_bs.T,lw=1,color='k')

            plt.figure()
            plt.ylabel("raw maxent output bootstrap")
            plt.plot(omega, A_bs.T,lw=1,color='k')

            plt.figure()
            plt.ylabel("imaginary time data reproduction bootstrap")
            #note: errorbar is += 1 std error of mean
            plt.errorbar(np.arange(L)*dt,chi.mean(0),\
                yerr = np.std(chi, axis=0,ddof=1)/np.sqrt(nbin),fmt='s',label="data")
            for i in range(bs):
                plt.plot(pre["tau"], pre["K"] @ A_bs[i,:] * pre["norm"],lw=1,color='k')
            plt.legend(loc='best')
            plt.show()

        return { "A": np.nanmean(A_bs,axis=0), "s": s_bs}

def get_lhs_data():
    return np.random.rand(50,100)  # dummy data for testing, replace with actual data

def main():
    # prepare imaginary time data, shape (Nbin, L). If using nonsymmetric bosonic kernel, should include tau=beta point
    lhs = get_lhs_data() 
    omega_grid = maxent.gen_grid(400//2, 0, 2.1, lambda x: 0.4*np.sinh(2.5*x))
    metadata = {"dt": 0.05, "beta": 5.0, "L": 100, "nbin": lhs.shape[0]}

    maxent_results = perform_maxent(lhs, omega_grid, metadata, 
                                    append=None, bs=1, anneal_arr = None, 
                                    checks=False, op_type='boson',printout=True, inspect=False)

if __name__ == "__main__":
    main()