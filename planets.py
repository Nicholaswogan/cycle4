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








