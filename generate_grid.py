import warnings
warnings.filterwarnings('ignore')

import multiprocessing as mp
import pickle
from tqdm import tqdm
import os
import numpy as np
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

import utils
import planets

# Initialize the climate model
CLIMATE_MODEL = utils.AdiabatClimateThermalEmission(
    planets.LTT1445Ab.Teq,
    planets.LTT1445Ab.mass,
    planets.LTT1445Ab.radius,
    planets.LTT1445A.Teff,
    planets.LTT1445A.metal,
    planets.LTT1445A.logg,
    planets.LTT1445A.radius,
    nz=50
)
CLIMATE_MODEL.verbose = False

def model(x):
    log10PH2O, log10PCO2, log10PO2, chi, albedo = x

    c = CLIMATE_MODEL
    c.set_custom_albedo(np.array([1.0]), np.array([albedo]))
    c.chi = chi
    P_i = 10.0**(np.array([log10PH2O, log10PCO2, log10PO2]))*1.0e6
    converged = c.RCE_robust(P_i)
    _, _, fpfs = c.fpfs()

    # Save results as 32 bit floats
    result = {}
    result['converged'] = converged
    result['x'] = x.astype(np.float32)
    result['P'] = c.P.astype(np.float32)
    result['T'] = c.T.astype(np.float32)
    result['fpfs'] = fpfs.astype(np.float32)

    return result

def worker(x, q):
    q.put(model(x))

def listener(q, filename):
    while True:
        m = q.get()
        if m == 'kill':
            break

        # Open the file if needed
        if not os.path.isfile(filename):
            with open(filename, 'wb') as f:
                pass

        # Write to it
        with open(filename, 'ab') as f:
            pickle.dump(m,f)

def get_inputs():
    log10PH2O = np.arange(-4,2.01,0.5) # bars
    log10PCO2 = np.arange(-7,2.01,0.25)
    log10PO2 = np.arange(-5,2.01,0.5)
    chi = np.array([0.1, 0.2, 0.3, 0.4]) # heat efficiency term
    albedo = np.arange(0,0.401,0.05)

    inputs = []
    for i1 in range(log10PH2O.shape[0]):
        for i2 in range(log10PCO2.shape[0]):
            for i3 in range(log10PO2.shape[0]):
                for i4 in range(chi.shape[0]):
                    for i5 in range(albedo.shape[0]):
                        tmp = [log10PH2O[i1], log10PCO2[i2], log10PO2[i3], chi[i4], albedo[i5]]
                        inputs.append(tmp)

    inputs = np.array(inputs)
    return inputs

def main():
    filename = 'results/tmp.pkl' # Specify output filename
    ncores = 4 # Specify number of cores
    inputs = get_inputs() # get inputs

    # Below here does not need changing

    # Exception if the file already exists
    if os.path.isfile(filename):
        raise Exception(filename+' already exists!')

    manager = mp.Manager()
    q = manager.Queue()
    with mp.Pool(ncores+1) as pool:

        # Put listener to work first
        watcher = pool.apply_async(listener, (q,filename,))

        # Fire off workers
        jobs = []
        for i in range(inputs.shape[0]):
            x = inputs[i,:]
            job = pool.apply_async(worker, (x, q))
            jobs.append(job)

        # Collect results from the workers through the pool result queue
        for job in tqdm(jobs): 
            job.get()

        # Kill the listener
        q.put('kill')

if __name__ == "__main__":
   main()