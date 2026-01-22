from glob import glob
import sys
import numpy as np
import util


def get_mu_n(path):
    '''From all files in path, get chemical potential and filling info
    Args:
        path
    Returns:
        tuple (mu, density, density_err)
    '''
    n_sample, sign, density = \
        util.load(path, "meas_eqlt/n_sample", "meas_eqlt/sign",
                        "meas_eqlt/density")
    mask = (n_sample == n_sample.max())
    if not mask.all():
        print(f"{path} incomplete: {mask.sum()}/{len(n_sample)}")
    sign, density = sign[mask], density[mask]
    dsum = density.sum(1)
    valid = (np.isfinite(sign)) & (np.isfinite(dsum)) & (sign != 0)
    sign = sign[valid]; density = density[valid]; dsum = dsum[valid]
    if sign.size < 3:
        return util.load_firstfile(path, "metadata/mu")[0], np.nan, np.nan
    nj = util.jackknife(sign, dsum)

    return util.load_firstfile(path, "metadata/mu")[0], nj[0], nj[1]


def get_mu(targets, paths):
    '''
    Given list of taget filling levels and a list of paths,
    find list of chemical potentials that will satisfy given targets
    Args:
        list of target filling levels in [0,2]
        list of paths, usually mu_*/
    Returns:
        Tuple 
        [0]: all mu, n data
        [1]: list of mu values
    '''
    data = np.array([get_mu_n(path) for path in paths])
    #sort rows by size of mu
    data = data[np.argsort(data[:, 0])]
    # Drop rows with invalid density (NaN/Inf). This prevents polyfit/roots from
    # receiving NaNs when some mu points had zero-sign bins filtered out upstream.
    finite_mask = np.isfinite(data[:, 1])
    if not np.all(finite_mask):
        data = data[finite_mask]
    if data.shape[0] < 3:
        # Not enough valid points to fit a quadratic; bail out cleanly.
        # Return NaNs for mus to signal "need more/other mu points".
        return data, np.array([np.nan for _ in targets], dtype=float)
    
    mus = np.zeros(len(targets))
    for i in range(len(targets)):
        y = data[:, 1] - targets[i]

        closest = np.abs(y).argmin()
        j = np.clip(closest, 1, data.shape[0] - 2)
        if j != closest:
            print(f"warning: target {targets[i]} out of range of data")

        p = np.polyfit(data[j-1:j+2, 0], y[j-1:j+2], 2)
        r = np.roots(p)
        mus[i] = r[np.abs(r - data[closest, 0]).argmin()]
    
    return data, mus


def main(argv):
    target = float(argv[1])
    paths = argv[2:] #linux system automatic path expansion

    data, mus = get_mu([target], paths)
    
    #last col:how much does filling change with each mu?
    diffs = np.zeros(data.shape[0])
    diffs[:-1] = data[1:, 1] - data[:-1, 1]
    
    print(np.hstack((data, diffs[:, None])))
    print(mus)


if __name__ == "__main__":
    main(sys.argv)
