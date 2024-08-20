import numpy as np
import yaml

def SNR_of_integration(filterdict, tau):
    num_integrations = tau/filterdict['time-per-integration'] + 1
    std = filterdict['std-per-integration']/np.sqrt(num_integrations)
    SNR = filterdict['extracted-flux']/std
    return SNR

def fpfs_std(SNR_in, SNR_out):
    return np.sqrt((1/SNR_in)**2 + (1/SNR_out)**2)

def get_error(filtername, tau_in, tau_out):

    data = yaml.safe_load(ETC_RESULTS)
    filterdict = data[filtername]
    
    SNR_in = SNR_of_integration(filterdict, tau_in)
    SNR_out = SNR_of_integration(filterdict, tau_out)
    
    std = fpfs_std(SNR_in, SNR_out)
    filterdict['SNR_in'] = SNR_in
    filterdict['SNR_out'] = SNR_out
    filterdict['fpfs_std'] = std
    filterdict['name'] = filtername

    return filterdict

def get_errors(filters, nvisit, tau_in, tau_out):
    data_bins = np.empty((len(filters),2))
    errs = np.empty(len(filters))
    for i,f in enumerate(filters):
        res = get_error(f, tau_in, tau_out)
        errs[i] = res['fpfs_std']/np.sqrt(nvisit[i])
        data_bins[i,:] = np.array(res['wavelength-range'])
    return errs, data_bins

ETC_RESULTS = """
F1280W:
  groups-per-integration: 5
  wavelength-range: [11.6, 14.1]
  time-per-integration: 1.79711997 # s
  extracted-flux: 1518318.18 # e-/s
  std-per-integration: 1957.15 # e-/s

F1500W:
  groups-per-integration: 8
  wavelength-range: [13.6, 16.5]
  time-per-integration: 2.69567997 # s
  extracted-flux: 1151739.44 # e-/s
  std-per-integration: 1051.79 # e-/s

F1800W:
  groups-per-integration: 19
  wavelength-range: [16.6,19.5]
  time-per-integration: 5.99039997 # s
  extracted-flux: 489944.13 # e-/s
  std-per-integration: 410.17 # e-/s
"""