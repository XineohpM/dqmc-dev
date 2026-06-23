# adapted from Katherine and Edwin's maxent code 

import numpy as np
from scipy import linalg as la
from scipy.special import xlogy
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.stats
import matplotlib.pyplot as plt

#---Checked kernels, same as edwin, except fermion kernel sign----
def Kernel_B(beta,tau,omega,sym=False):
    '''Bosonic kernel, general (nonsymmetric) case
    Returns:
        (ntau,nomega) float array'''
    assert tau.max() <= beta and tau.min() >= 0
    #avoid pure zero by adding machine eps to zero
    omega[omega == 0] += np.finfo(float).eps 
    if sym:
        top = omega*( np.exp(-np.outer(tau,omega)) + \
        	np.exp(-np.outer(beta-tau,omega)) )
    else:
        top = omega*np.exp(-np.outer(tau,omega))
        
    bot = 1-np.exp(-beta*omega)
    return top/bot


def Kernel_F(beta,tau,omega,sym=False):
    '''Fermionic kernel, general (nonsymmetric) case. 
    Opposite sign to edwin.
    Returns:
        (ntau,nomega) float array'''
    assert tau.max() <= beta and tau.min() >= 0
    if sym:
        raise NotImplementedError("Fermion symm case not written")
    return np.exp(-np.outer(tau,omega))/(1+np.exp(-beta*omega))

def Kernel_T(beta,tau,omega):
    assert tau.max() <= beta and tau.min() == 0
    omega[omega == 0] += np.finfo(float).eps 
    top = np.exp(np.outer(tau,omega)) - np.exp(np.outer((beta-tau),omega))
    bot = 1-np.exp(beta*omega)
    return top/bot

#------------------------------------------------------------------
def Entropy(A, m):
    '''Nonpositive entropy term, sum(-A*log(A/m))'''
    #USING XLOGY to handle A = 0 case
    #return np.sum((-xlogy(A,A/m)))
    return np.sum((A-m-xlogy(A,A/m)))

def Chi_Sq(A,   Kp,   Gp,  W):
    '''Chi^2 error generalized least squares problem'''
    KA_G = Kp @ A  - Gp;
    return np.vdot(KA_G * W, KA_G)

def Qp(A,*args):
    '''Objective function to maximize, alpha*S[A,m] - Chi^2[A]/2'''
    m, alpha, Kp,   Gp,  W = args
    return Entropy(A,m)*alpha - Chi_Sq(A,  Kp,   Gp,  W)/2

def Qm(A, *args):
    '''Objective function to minimize, -alpha*S[A,m] + Chi^2[A]/2'''
    m,  alpha, Kp,   Gp,  W = args
    return -Entropy(A,m)*alpha + Chi_Sq(A,  Kp,   Gp,  W)/2

#-----------------------------------------------------------------------

def MaxEnt(pre,  alpha_arr = np.logspace(1, 9, 1+20*(9-1)),  method = 'BT', \
    printout=False, inspect=False):
    #print(inspect)
    '''Perform MaxEnt, pick best alpha based on method spec.
    Args:
        pre: dictionary with preprocessed data
        alpha_arr : array of alpha values to chose from
        op_type : 'fermion' or 'boson'
        method = 'historic' or "classic" or "bryan" or "BT" (default)
        printout : bool, whether to print norm info
        inspect : bool, whether to plot intermediate checks
        
    Returns:
        Array: (N_omega,) array = best estimate of A(omega_i) * domega_i
    '''
    eigv_threshold = 1e-8  # clip covar matrix eigenval up if < max eigen_val * this
    svd_threshold = 1e-12  # drop kernel singular vals if < max singular_val * this
    #-------------------------------------------------------------------------------------
    G = pre["lhs"] #data
    K = pre["K"] #kernel
    m = pre["m"] #model
    tau = pre["tau"] #tau grid

    #calc_A(G, K, m, alpha_arr, plot=True, useBT=False)
    #-------------------------------------------------------------------------------------


    #covariance matrix processing
    Nbin, _ = G.shape
    C = np.cov(G,rowvar=False)/Nbin
    s,Q = la.eigh(C); #s = sigma^2 array
    #clip small eigenvalues upwards to threshold
    mask = np.abs(s) <= np.max(s)*eigv_threshold
    #print(C.shape)
    s[mask] = np.max(s)*eigv_threshold; #commented before
    dof = tau.shape[0] - np.count_nonzero(mask)

    if printout:
        print(f'Choose optimal alpha using {method} method')
        print(f"covariance matrix condition number: {np.linalg.cond(C):.3g}")
        print(f"Using {dof}/{tau.shape[0]} eigenvals in C")

    if inspect:
        plt.figure();
        plt.title(f'Covariance matrix C eigenvals')
        plt.plot(s,'.')
        plt.yscale("log")
        plt.ylabel(r'$\sigma^2_{\ell}$')
        plt.xlabel(r"eigval index $\ell$")
        plt.grid(True)
        plt.show()
    
    
    #rotate kernel and data to make covariance matrix diagonal
    Kp = Q.conj().T @ K;
    Gp = Q.conj().T @ np.mean(G,axis=0) #avg data value
    W = 1/s #vector of 1/sigma^2

    
    #precalculate SVD of Kp
    V, sigma, Uh = np.linalg.svd(Kp,full_matrices = False)
    # drop singular values less than threshold
    mask = sigma >= svd_threshold*np.max(sigma)
    if printout:
        print(f"Using {np.count_nonzero(mask)}/{Kp.shape[0]} singular values of Kp")
    if inspect:
        plt.figure();
        plt.title('kernel mat K singular values')
        plt.plot(sigma,'.')
        plt.yscale("log")
        plt.xlabel("singular val index")
        plt.grid(True)
        plt.show()

    #reduce matrix dimensions
    V = V[:,mask]
    sigma = sigma[mask]
    Uh = Uh[mask,:]
    precalc_svd = V, sigma, Uh

    #-------------------------------Tune alpha value----------------------------------
    #if given a scalar alpha_arr, do not tune alpha
    if np.isscalar(alpha_arr):
        print("input alpha array is scalar, no tuning")
        A_out, _, _ =  MaxEnt_Fixed_Alpha(Gp,W,Kp,m,alpha_arr,precalc_svd)
        return A_out

    #get optimized A, chi as function of alpha
    N_alpha = alpha_arr.shape[0]
    A_arr = np.full((N_alpha,m.shape[0]),np.nan,dtype=float)
    Q_arr = np.full(N_alpha,np.nan,dtype=float)
    lnP_arr = np.full(N_alpha,np.nan,dtype=float)
    Ngood_arr = np.full(N_alpha,np.nan,dtype=float)
    chi2_arr = np.full(N_alpha,np.nan,dtype=float)
    for i in range(N_alpha):
        A_arr[i,:],lnP_arr[i],Ngood_arr[i] = MaxEnt_Fixed_Alpha(Gp,W,Kp,m,\
            alpha_arr[i],precalc_svd)
        Q_arr[i] = Qp(A_arr[i], m, alpha_arr[i], Kp,   Gp,  W)
        chi2_arr[i] = Chi_Sq(A_arr[i],   Kp,   Gp,  W)

    if method == 'historic':
        #goal: find alpha that produces A giving Chi^2 = dof
        #root finding does not need to be very precise
        #removed this method since MaxEnt_Fixed_Alpha now returns tuples
        #this method is kinda bad anyways
        '''
        obj_f = lambda al : Chi_Sq(MaxEnt_Fixed_Alpha(Gp,W,Kp,m,al,precalc_svd)[0], Kp,   Gp,  W) - dof
        sol = root_scalar(obj_f,x0 = alpha_arr[0],x1 = alpha_arr[-1],rtol=1e-2)
        alpha_out = sol.root
        A_out,_ = MaxEnt_Fixed_Alpha(Gp,W,Kp,m,alpha_out,precalc_svd)'''
        raise NotImplementedError(f"{method} not implemented")
    elif method == 'classic':
        pos = np.argmax(lnP_arr)
        alpha_out = alpha_arr[pos]
        A_out = A_arr[pos,:]
        if inspect:
            plt.figure();
            plt.title(f"dof = {dof}, Ngood = {Ngood_arr[pos]}")
            plt.plot(alpha_arr,np.exp(lnP_arr),'.', ms = 2, color="g")
            plt.axvline(x=alpha_out,lw=1,color="g")
            plt.xscale("log")
            #plt.yscale("log")
            plt.grid(True)
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$\log P(\alpha|G)$")
            plt.show();
    elif method == 'bryan':
        #TODO: better integration than rectangle rule
        dalpha = np.diff(alpha_arr)
        dalpha = np.insert(dalpha,0,dalpha[0])
        Z = np.sum(np.exp(lnP_arr)*dalpha)
        print("probability normalization factor = ",Z)
        #spl = InterpolatedUnivariateSpline(alpha_arr,np.exp(lnP_arr))
        #print(spl.integral(alpha_arr[0],alpha_arr[-1]))
        #assert dalpha.shape == alpha_arr.shape
        A_out = np.sum((np.exp(lnP_arr)*dalpha)[:,None]*A_arr,axis=0)/Z
        #not sure this alpha_out in the average sense is accurate at all
        alpha_out = np.sum(np.exp(lnP_arr)*dalpha*alpha_arr)/Z
        if inspect:
            plt.figure();
            plt.title(f"dof = {dof}")
            plt.plot(alpha_arr,np.exp(lnP_arr),'.', ms = 2, color='m')
            plt.xscale("log")
            #plt.yscale("log")
            plt.grid(True)
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$P(\alpha|G)$")
            plt.show();
    elif method == 'BT':
        #from Bergeron Tremblay 2016 paper
        log_a = np.log(alpha_arr)
        log_c = np.log(chi2_arr)
        #cubic spline fit, get curvature
        spl = InterpolatedUnivariateSpline(log_a,log_c,ext=2,check_finite=True)
        top = spl(log_a,nu=2)
        bot = np.power((1+np.power(spl(log_a,nu=1),2)),1.5)
        #BT soln 
        pos = np.argmax(top/bot) #find max of signed curvature
        alpha_out = alpha_arr[pos]
        A_out = A_arr[pos,:]
        if inspect:
            plt.figure();
            plt.title(f"curvature, dof={dof}, Ngood={Ngood_arr[pos]}")
            plt.plot(alpha_arr,top/bot,'.',ms=2,color='b')
            plt.axvline(x=alpha_out,lw=1,color="b")
            plt.axhline(y=0,lw=1,color="k")
            plt.grid(True)
            plt.xscale("log")
            plt.show()
        if printout:
            print(
                f"BT selected alpha = {alpha_out:.6e} "
                f"(index {pos}/{len(alpha_arr)-1}), "
                f"chi2 = {chi2_arr[pos]:.6g}"
            )
    else:
        raise ValueError('invalid method spec')
    
    #----------------------------End tune alpha value----------------------------------------
    c2lo, c2hi = scipy.stats.chi2.interval(0.95, dof)
    if inspect:
        plt.figure()
        plt.plot(alpha_arr,chi2_arr,lw=1,label = r"likelihood $\chi^2$")
        #plt.plot(alpha_arr,lnP_arr,lw=1)
        plt.title(f"dof = {dof}")
        plt.xscale("log");plt.yscale('log')
        plt.xlabel(r"$\alpha$")
        plt.axvline(x=alpha_out,color='b',lw=1,label = rf"{method} optimal $\alpha$")
        plt.axhline(y=c2lo,color='k',ls="-.",lw=1,label = r"$\chi^2$ 0.95 lower bound")
        plt.axhline(y=dof, color='k',ls="-", lw=1)
        plt.axhline(y=c2hi,color='k',ls="--",lw=1,label = r"$\chi^2$ 0.95 upper bound")
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show()

    
    cf = Chi_Sq(A_out,Kp,Gp,W)
    if printout:   
        print(f"alpha = {alpha_out:.3f}, chi^2/dof = {cf:.3f}/{dof} = {cf/dof:.3f}",
            f" sum(A*domega) = {A_out.sum():.6f}",
            f" entropy = {np.sum((A_out-m)):.3g} + {np.sum(-xlogy(A_out,A_out/m)):.3g},",
            f" -2*alpha*S = {-2*alpha_out*Entropy(A_out,m):.3g}\n")
        
    if cf > c2hi:
        print(f"\033[93mChi-squared error = {cf:.3f} > {c2hi:.3f}\033[0m")
        #return np.full(A_out.shape,np.nan)
    if np.abs(A_out.sum() - 1) > 5e-2:
        print(f"\033[91msum(A*domega) = {A_out.sum():.6f}, return A nan\033[0m")
        return np.full(A_out.shape,np.nan)


        
    return A_out



#beta, tau, omega, L, Nbin, dt?
def MaxEnt_Fixed_Alpha(Gp, W, Kp, m, alpha, precalc_svd = None):
    '''Perform Bryans Optiization Algorithm for fixed alpha value
    Args:
        Gp: symmetry appropriate (N_tau,) data, rotated and normalized, divided by sign
        W: (N_tau,) data errors
        Kp: (N_tau, N_omega) kernel
        m: m(omega_i) * domega_i, shape (N_omega,) 
        alpha: user specified alpha value
        precalc_svd:
    Returns:
        best estimate of A(omega_i) * domega_i, shape (N_omega,), given fixed alpha
    '''
    svd_threshold = 1e-12  # drop kernel singular vals if < max singular_val * this
    maxiter = 1234 #max number of root finding iterations in Bryans algorithm
    max_step_size = m.sum();
    small_step_threshold = 0.125
    mu_multiplier, mu_min, mu_max = 2.0, alpha/4, alpha*1e100
    dQ_threshold = 1e-10
    conseq_threshold = 7

    #m[np.abs(m) < 1e-16] = np.nan
    
    #-----------------------Bryan's optimization algorithm--------------------------------------
    # svd of kernel: K = V Sigma U.H
    if precalc_svd is None:
        print("Getting svd of Kp")
        V, sigma, Uh = np.linalg.svd(Kp,full_matrices = False)
        # drop singular values less than threshold
        mask = sigma >= svd_threshold*np.max(sigma)
        print(f"Using {np.count_nonzero(mask)}/{Kp.shape[0]} singular values of Kp")

        #reduce matrix dimensions
        V = V[:,mask]
        sigma = sigma[mask]
        Uh = Uh[mask,:]
    else:
        V, sigma, Uh = precalc_svd
    
    #precalculated stuff, doesnt depend on state u
    U = Uh.conj().T 
    sigVT = sigma.conj()[:,None] * V.conj().T
    M = np.dot(sigVT * W, V * sigma)
    I = np.identity(sigma.shape[0])
    
    #initialize u state
    u = np.zeros(sigma.shape) #variable we optimize over
    #quantities that depend on u state
    A = m * np.exp(np.dot(U,u))
    T = (Uh * A) @ U
    f = alpha * u + (sigVT * W) @ (Kp @ A - Gp)
    q_old = Qm(A,m,alpha,Kp,Gp,W)
    #assert np.all(np.isfinite(A)) and np.all(np.isfinite(T)) and \
    #    np.all(np.isfinite(f)) and np.all(np.isfinite(q_old))
    
    nit = 0; n_conseq = 0; mu = alpha #LM param
    #Newton root finding in u space, with extra LM param
    while nit < maxiter:
        #assert np.all(np.isfinite(T))
        #propose a du step
        #jac = (alpha + mu)*I + M @ T
        Xi, P = np.linalg.eigh(T)
        assert np.all(np.abs(Xi[Xi<0])<2e-14), f"error {np.abs(Xi[Xi<0])}"
        Xi[ Xi < 0] = 0
        #print("P is U?", np.allclose(P,U.T))
        AA = np.sqrt(Xi)[:,None] * P.T @ M @ P * np.sqrt(Xi)
        #print("A matrix", AA)
        Lam, R = np.linalg.eigh(AA)
        #print(Lam)
        Yinv = (R.T * np.sqrt(Xi)) @ P.T
        Yinvu = - R.T @ (np.sqrt(Xi)*(P.T @ f))/(alpha + mu + Lam)
        du =  - (f + M @ (P @ (np.sqrt(Xi) * (R @ Yinvu)))) / (alpha + mu)
        step_size = np.dot(Yinvu,Yinvu) ; 

        A = m * np.exp(np.dot(U,u+du))
        A[A > m.max()*1e3] = m.max()*1e3
        #if m too small, then exp term can overflow
        if np.max(np.dot(U,u+du)) > 100:
            print(f"alpha = {alpha}, iter = {nit}, max U@u",np.max(np.dot(U,u+du)), \
                f"U norm {np.linalg.norm(U):.3g} u norm: {np.linalg.norm(u+du)} qnew = { Qm(A,m,alpha,Kp,Gp,W)}\
                qold = {q_old}")
            #uncomment the code below to plot
            #plt.figure()
            #plt.plot(m,label="m")
            #plt.plot(A,label="A")
            #plt.ylim(0,0.1)
            #plt.legend(loc="best")
            #plt.show()
        q_new = Qm(A,m,alpha,Kp,Gp,W)
        if step_size > max_step_size or \
            np.any(np.logical_not(np.isfinite(A))) or q_new/q_old > 1e3:
            #print(f"reject_step {nit}, increase mu")
            A = m * np.exp(np.dot(U,u))
            mu = np.clip(mu*mu_multiplier, mu_min, mu_max)
            nit +=1
            continue

        
        #turns out, Q ratio is too big, have to reject this step
        if q_new/q_old > 1e3 :
            u -= du
            A = m * np.exp(np.dot(U,u))
            T = (Uh * A) @ U
            f = alpha * u + (sigVT * W) @ (Kp @ A -Gp)
            print(f"WARNING: Q ratio = {q_new/q_old}, step_size = {step_size}")
            
        #print(f"accept step {nit},update u,T,f,Q")
        u += du
        T = (Uh * A) @ U
        f = alpha * u + (sigVT * W) @ (Kp @ A -Gp)
        #this step size small, decrease mu for next iter
        if step_size < small_step_threshold:
            mu = mu/mu_multiplier if mu > mu_min else 0

        #count consequtive dQ/Q < dQ_threshold
        if np.abs(q_old/q_new-1) < dQ_threshold: n_conseq +=1
        else: n_conseq = 0
        #if converged, then break
        if n_conseq >= conseq_threshold: break
        #refresh q value
        q_old = q_new
        nit += 1
    
    #if nit == maxiter:
    #print(f"\033[4miIter {nit} reached, alpha={alpha:.3g}, probably no converge\033[0m")

    lam =  np.linalg.eigvalsh(np.sqrt(A)[:,None] * U@M@Uh * np.sqrt(A))
    Ngood=np.sum(lam/(lam+alpha))
    #print("logP(alpha)",-np.log(alpha))
    #print("Q",Qp(A,m,alpha,Kp,Gp,W))
    #print("extra",0.5*np.log(alpha/(lam+alpha)).sum())
    # print(f"alpha = {alpha}, iter = {nit}, max U@u",np.max(np.dot(U,u+du)), f"U norm {np.linalg.norm(U):.3g} u norm: {np.linalg.norm(u+du)} m min {np.min(m)} qnew = { Qm(A,m,alpha,Kp,Gp,W)}")
    # plt.figure()
    # plt.plot(m)
    # plt.plot(A)
    # plt.ylim(0,0.1)
    # plt.show()
    #not using jeffrys prior
    lnP = + Qp(A,m,alpha,Kp,Gp,W) + 0.5*np.sum(np.log(alpha/(lam+alpha)))
        
    return A, lnP, Ngood


#-----------------------------------------------------------------
def Preprocess(G, dt, beta, grid_info, op_type = "boson", sym = None, \
    append = None,model_arr = None):
    '''
    Args:
        G: (Nbin, L) float array, divided by sign, no end, no rotate, no norm
        dt: float
        beta: float = dt*L
        grid_info: (w,dw) typle
        op_type: "boson" or "fermion", "boson" by default 
        sym: True or False
        append: (Nbin,1) array, only used when op_type = 'boson' and sym=False
        model_arr: None or model array

    Returns:
        a dictionary with fields
            "tau": tau grid, (Ntau,) float array
            "m": model, (Nw,) float array
            "lhs" : G, LHS of AC, div by sign, sym shape ok, normalized, before rotation
            "norm" : normalization factor so that A integrates to 1, float
            "K": K, (Ntau,Nomega) kernel
    '''
    #grid info is user specified, check consistent with symmetry
    omega,domega = grid_info
    assert np.all(np.diff(omega)>0), f"w not strictly increasing"
    assert np.all(domega > 0), f"dw has nonpositive values"
    if sym:
        assert omega.min() > 0, f"sym = {sym}, only positive w needed"
    else:
        assert omega.min() < 0, f"sym = {sym}, need negative w"
    
    #Process based on if particle-hole symmetry
    Nbin, L = G.shape

    #Create tau grid, kernel, Normalize G
    if op_type == "boson":
        if sym: 
            G0 = np.reshape(G[:,0],(Nbin,1))
            G = np.concatenate((G,G0),axis=1)
            #normalization
            spl = InterpolatedUnivariateSpline(np.arange(L+1)*dt,G.mean(0),ext=2,check_finite=True)
            norm_factor = spl.integral(0,beta/2)
            #print(op_type,sym,norm_factor)
            #process based on symmetry, keep only fist half of interval
            Grev = np.fliplr(G)
            G = 0.5*(G + Grev)[:,:(L//2+1)]
            tau = np.arange(L//2+1)*dt
        else:
            #append extra data at tau = beta when not symmetric
            G = np.concatenate((G,append),axis=1)
            #normalization
            spl = InterpolatedUnivariateSpline(np.arange(L+1)*dt,G.mean(0),ext=2,check_finite=True)
            norm_factor = spl.integral(0,beta)
            #keep extra bin
            tau = np.arange(L+1)*dt
            #print(op_type,sym,norm_factor)
        K = Kernel_B(beta,tau,omega,sym=sym);
    elif op_type == 'fermion':
        if sym:
            raise NotImplementedError
        else:
            norm_factor = 1;
            G0 = np.reshape(G[:,0],(Nbin,1))
            res = np.random.randint(Nbin,size=Nbin)
            #is this way OK?
            G = np.concatenate((G,norm_factor-G0[res]),axis=1)
            tau = np.arange(0,L+1)*dt
            K = Kernel_F(beta,tau,omega,sym=sym);
    else:
        raise ValueError(f"{op_type} operator, symmetry = {sym} invalid")
    
    #anneal model
    if model_arr is not None:
        assert model_arr.shape[0] == omega.shape[0]
        #clip small values to avoid exp(U@u) overflow in bryan's algorithm
        model_arr[model_arr < model_arr.max()*1e-4] = model_arr.max()*1e-4
        print("model norm before normalization:",model_arr.sum())
        m = model_arr/model_arr.sum()
        print("processed model min", m.min())
    else:
        #default to flat model
        m = model_flat(domega)

    #return structure
    d = {"tau": tau,
         "m": m,
         "lhs" : G/norm_factor, #div by sign, sym shape ok, normalized
         "norm" : norm_factor, #normalization factor
         "K": K}
    
    return d



def model_flat(dw):
    return dw/dw.sum()


def gen_grid(nw, x_min, x_max, w_x):
    """
    generate grid with nw points scaled by the function w_x.

    w[i] = w_x((i+0.5)/nw * (x_max-x_min) + x_min)
    dw[i] = w_x((i+1)/nw * (x_max-x_min) + x_min) -
            w_x(i/nw * (x_max-x_min) + x_min)

    returns w, dw
    """
    x_all = np.linspace(x_min, x_max, 2*nw+1)
    w_all = np.apply_along_axis(w_x, 0, x_all)
    return w_all[1::2], np.abs(np.diff(w_all[::2]))