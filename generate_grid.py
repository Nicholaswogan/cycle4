import warnings
warnings.filterwarnings('ignore')

import multiprocessing as mp
import pickle
from tqdm import tqdm
import os
import numpy as np
from scipy import interpolate
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

import utils
import planets

class GridInterpolator():

    def __init__(self, filename, gridvals):

        # read the file
        res = []
        inds = []
        with open(filename,'rb') as f:
            while True:
                try:
                    tmp = pickle.load(f)
                    res.append(tmp[2])
                    inds.append(tmp[0])
                except EOFError:
                    break

        # Reorder results to match gridvals
        res_new = [0 for a in res]
        for i,r in enumerate(res):
            res_new[inds[i]] = res[i]
        
        self.gridvals = gridvals
        self.gridshape = tuple([len(a) for a in gridvals])
        self.results = res_new

    def make_array_interpolator(self, key):
        # Make interpolate for key
        interps = []
        for j in range(len(self.results[0][key])):
            val = np.empty(len(self.results))
            for i,r in enumerate(self.results):
                val[i] = r[key][j]
            interp = interpolate.RegularGridInterpolator(self.gridvals, val.reshape(self.gridshape))
            interps.append(interp)

        def interp_arr(vals):
            out = np.empty([len(interps)])
            for i,interp in enumerate(interps):
                out[i] = interp(vals)
            return out
        
        return interp_arr
    
def listener(q, filename):

    while True:

        m = q.get()
        if m == 'kill':
            break

        # Write to it
        with open(filename, 'ab') as f:
            pickle.dump(m,f)

def get_inputs(gridvals):
    tmp = np.meshgrid(*gridvals, indexing='ij')
    inputs = np.empty((tmp[0].size,len(tmp)))
    for i,t in enumerate(tmp):
        inputs[:,i] = t.flatten()
    return inputs

# This stuff has to be top level but does not have to change

def worker(i, x, q):
    res = model(x)
    q.put((i,x,res,))

def main(gridvals, filename, ncores):

    inputs = get_inputs(gridvals) # get inputs

    # Exception if the file already exists
    if os.path.isfile(filename):
        raise Exception(filename+' already exists!')
    else:
        if not os.path.isfile(filename):
            with open(filename, 'wb') as f:
                pass

    manager = mp.Manager()
    q = manager.Queue()
    with mp.Pool(ncores+1) as pool:

        # Put listener to work first
        watcher = pool.apply_async(listener, (q,filename,))

        # Fire off workers
        jobs = []
        for i in range(inputs.shape[0]):
            x = inputs[i,:]
            job = pool.apply_async(worker, (i, x, q))
            jobs.append(job)

        # Collect results from the workers through the pool result queue
        for job in tqdm(jobs): 
            job.get()

        # Kill the listener
        q.put('kill')

# Make the model function

# Initialize the climate model
CLIMATE_MODEL = utils.AdiabatClimateThermalEmission(
    planets.TRAPPIST1c.Teq,
    planets.TRAPPIST1c.mass,
    planets.TRAPPIST1c.radius,
    planets.TRAPPIST1.radius,
    stellar_surface_file='input/hlsp_hazmat_phoenix_synthspec_trappist-1_1a_v1_fullres_correctunits.txt',
    stellar_surface_scaling=0.85,
)
CLIMATE_MODEL.verbose = False
_ = CLIMATE_MODEL.initialize_picaso_from_clima(
    os.path.join(os.environ['picaso_refdata'],'opacities/')+'opacities_1_30_R10000.db',
    opannection_kwargs={'wave_range':[3,21]},    
)
WAVL_PICASO = utils.grid_at_resolution(4,20,100)
WAVL_CLIMA_INDS = np.array([
    32, 33, 34, 35, 36, 37, 38, 39, 
    40, 41, 42, 43, 44, 45, 46, 47, 
    48, 49, 50, 51, 52, 53, 54, 55, 
    56
])

def model(x):
    log10PH2O, log10PCO2, log10PO2, log10PSO2, chi, albedo = x

    c = CLIMATE_MODEL
    c.set_custom_albedo(np.array([1.0]), np.array([albedo]))
    c.chi = chi
    P_i = np.ones(len(c.species_names))*1e-15
    P_i[c.species_names.index('H2O')] = 10.0**log10PH2O
    P_i[c.species_names.index('CO2')] = 10.0**log10PCO2
    P_i[c.species_names.index('O2')] = 10.0**log10PO2
    P_i[c.species_names.index('SO2')] = 10.0**log10PSO2
    P_i *= 1.0e6 # convert to dynes/cm^2

    # Compute climate
    converged = c.RCE_robust(P_i)

    # Save the P-T profile
    P = np.append(c.P_surf,c.P)
    T = np.append(c.T_surf,c.T)
    
    # Get emission spectra
    _, _, fpfs = c.fpfs()
    fpfs = fpfs[WAVL_CLIMA_INDS]
    _, _, fpfs_picaso = c.fpfs_picaso(wavl=WAVL_PICASO)

    # Save results as 32 bit floats
    result = {}
    result['converged'] = converged
    result['x'] = x.astype(np.float32)
    result['P'] = P.astype(np.float32)
    result['T'] = T.astype(np.float32)
    result['fpfs'] = fpfs.astype(np.float32)
    result['fpfs_picaso'] = fpfs_picaso.astype(np.float32)

    return result

def get_gridvals():
    log10PH2O = np.arange(-4,2.01,1) # bars
    log10PCO2 = np.arange(-7,2.01,0.5)
    log10PO2 = np.arange(-5,2.01,1)
    log10PSO2 = np.arange(-7,0.0,1)
    chi = np.array([0.1, 0.25, 0.4]) # heat efficiency term
    albedo = np.arange(0,0.401,0.1)
    gridvals = (log10PH2O,log10PCO2,log10PO2,log10PSO2,chi,albedo)
    return gridvals

if __name__ == "__main__":
    filename = 'results/TRAPPIST1c_grid.pkl' # Specify output filename
    ncores = 4 # Specify number of cores
    gridvals = get_gridvals()
    main(gridvals, filename, ncores)