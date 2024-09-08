import numpy as np
import requests

class Star:
    radius : float # relative to the sun
    Teff : float # K
    metal : float # log10(M/H)
    kmag : float
    logg : float
    planets : dict # dictionary of planet objects

    def __init__(self, radius, Teff, metal, kmag, logg, planets):
        self.radius = radius
        self.Teff = Teff
        self.metal = metal
        self.kmag = kmag
        self.logg = logg
        self.planets = planets
        
class Planet:
    radius : float # in Earth radii
    mass : float # in Earth masses
    Teq : float # Equilibrium T in K
    transit_duration : float # in seconds
    eclipse_duration: float # in seconds
    a: float # semi-major axis in AU
    stellar_flux: float # W/m^2
    
    def __init__(self, radius, mass, Teq, transit_duration, eclipse_duration, a, stellar_flux):
        self.radius = radius
        self.mass = mass
        self.Teq = Teq
        self.transit_duration = transit_duration
        self.eclipse_duration = eclipse_duration
        self.a = a
        self.stellar_flux = stellar_flux

LTT1445Ab = Planet(
    radius=1.305, # Winters et al. (2022)
    mass=2.87, # Winters et al. (2022)
    Teq=424, # Winters et al. (2022)
    transit_duration=1.36*60*60, # JWST_ranking_20200525.xlsx
    eclipse_duration=1.36*60*60, # Munazza
    a=0.03813, # Winters et al. (2022)
    stellar_flux=7331.0 # 424 K equilibrium temp
)

LTT1445Ac = Planet(
    radius=1.147, # Winters et al. (2022)
    mass=1.54, # Winters et al. (2022)
    Teq=508, # Winters et al. (2022)
    transit_duration=0.4824*60*60, # Winters et al. (2022)
    eclipse_duration=0.482*60*60, # Munazza
    a=0.02661, # Winters et al. (2022)
    stellar_flux=15106.0 # 508 K equilibrium temp
)

LTT1445A = Star(
    radius=0.265, # Winters et al. (2022)
    Teff=3340, # Winters et al. (2022)
    metal=-0.34, # Winters et al. (2022)
    kmag=6.496, # COMPASS JWST_ranking_20200525.xlsx
    logg=4.97, # Exo.Mast
    planets={'b':LTT1445Ab,'c':LTT1445Ac}
)

# Agol et al. (2021) for all parameters
TRAPPIST1b = Planet(
    radius=1.116,
    mass=1.374,
    Teq=397.31,
    transit_duration=np.nan,
    eclipse_duration=np.nan,
    a=1.154e-2,
    stellar_flux=4.153*1361,
)

TRAPPIST1c = Planet(
    radius=1.097,
    mass=1.308,
    Teq=339.5,
    transit_duration=np.nan,
    eclipse_duration=np.nan,
    a=1.580e-2,
    stellar_flux=2.214*1361,
)

TRAPPIST1 = Star(
    radius=0.1192,
    Teff=2566,
    metal=np.nan,
    kmag=np.nan,
    logg=5.2396,
    planets={'b':TRAPPIST1b,'c':TRAPPIST1c}
)

def download_trappist1_spectrum():
    url = 'http://archive.stsci.edu/hlsps/hazmat/hlsp_hazmat_phoenix_synthspec_trappist-1_1a_v1_fullres.txt'
    r = requests.get(url)
    lines = r.content.decode().split('\n')
    wv = np.array([float(a.split()[0]) for a in lines[1:-1]])
    F = np.array([float(a.split()[1]) for a in lines[1:-1]])
    wv = wv/10 # convert from A to nm
    F = F*10 # convert from erg/s/cm^2/A to mW/m2/nm
    # removed duplicate wavelengths
    wv, inds = np.unique(wv,return_index=True)
    F = F[inds]
    
    fmt = '{:30}'
    with open('input/hlsp_hazmat_phoenix_synthspec_trappist-1_1a_v1_fullres_correctunits.txt','w') as f:
        f.write('Wavelength (nm)               Stellar flux (mW/m^2/nm)        \n')
        for i in range(wv.shape[0]):
            f.write(fmt.format('%.15e'%wv[i]))
            f.write(fmt.format('%.15e'%F[i]))
            f.write('\n')
