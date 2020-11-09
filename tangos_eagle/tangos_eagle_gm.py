from tangos.properties.pynbody import PynbodyPropertyCalculation
from tangos.properties import PropertyCalculation
import pynbody
import numpy as np

from astropy.cosmology import z_at_value
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

class r200(PynbodyPropertyCalculation):
    names = "r200"
    
    def calculate(self, particle_data, existing_properties):

        return pynbody.analysis.halo.virial_radius(particle_data,cen=existing_properties['shrink_center'],overden=200,rho_def='critical')

    def region_specification(self, existing_properties):
        return pynbody.filt.Sphere(existing_properties['max_radius']*2,
                                   existing_properties['shrink_center'])

    def requires_property(self):
        return ["shrink_center", "max_radius"]


class M200(PynbodyPropertyCalculation):
    names = "M200"
    
    def calculate(self, particle_data, existing_properties):
        return particle_data['mass'].sum()

    def region_specification(self, existing_properties):
        return pynbody.filt.Sphere(existing_properties['r200'],
                                   existing_properties['shrink_center'])

    def requires_property(self):
        return ["shrink_center", "r200"]


class CGMfractions(PynbodyPropertyCalculation):
    names = 'f_CGM', 'f_b', 'f_star', 'f_ISM'
    
    def calculate(self, pdata, existing):

        f_b_cosmic = pdata.properties['omegaB0']/pdata.properties['omegaM0']
        sfr = pdata.g['StarFormationRate']
        gas_mass = pdata.g['mass']
        star_mass = pdata.star['mass']
        sf = np.where(sfr>0.)[0]
        nsf = np.where(sfr==0.)[0]

        f_CGM = (np.sum(gas_mass[nsf]) / existing['M200']) / f_b_cosmic

        f_ISM = (np.sum(gas_mass[sf]) / existing['M200']) / f_b_cosmic

        f_star = (np.sum(star_mass) / existing['M200']) / f_b_cosmic

        f_b = ((np.sum(gas_mass) + np.sum(star_mass)) / existing['M200']) / f_b_cosmic

        return f_CGM, f_b, f_star, f_ISM

    def region_specification(self, existing):
        return pynbody.filt.Sphere(existing['r200'],
                                   existing['shrink_center'])

    def requires_property(self):
        return ["shrink_center", "r200", 'M200']


class BH(PynbodyPropertyCalculation):
    names = 'M_BH', 'Mdot_BH', 'E_AGN'
    
    def calculate(self, pdata, existing):

        BH_erg_per_g = 0.1 * 0.15 * (299792458.*100.)**2 * pynbody.units.Unit('erg')/pynbody.units.Unit('g')

        bh_masses = pdata.bh['BH_Mass']

        bh_locate = np.argmax(bh_masses)

        E_AGN = (pdata.bh['BH_Mdot'].astype(np.float64).in_units('g s**-1') * BH_erg_per_g).in_units('erg s**-1')
        
        return bh_masses[bh_locate], pdata.bh['BH_Mdot'].in_units('Msol yr**-1')[bh_locate], E_AGN[bh_locate]

    def region_specification(self, existing):
        return pynbody.filt.Sphere(existing['r200'],
                                   existing['shrink_center'])

    def requires_property(self):
        return ["shrink_center", "r200"]


class aexp_prior(PynbodyPropertyCalculation):
    names = "aexp_300Myr_ago"

    def calculate(self, s, e):

        cosmo = FlatLambdaCDM(100.*s.properties['h'],Om0=s.properties['omegaM0'],Ob0=s.properties['omegaB0'])

        aexp_300Myr = 1./(1.+z_at_value(cosmo.age, (cosmo.age(s.properties['Redshift']).value-0.3) * u.Gyr))

        return aexp_300Myr
        

class stellar(PynbodyPropertyCalculation):
    names = 'Mstar_30kpc', 'SFR_300Myr'
    
    def calculate(self, pdata, existing):

        Mstar_30kpc = pdata.star['mass'].sum()

        formationtime = pdata.star['aform']
        sfr_integrated = (np.sum(pdata.star['InitialMass'].in_units('Msol')[formationtime>existing['aexp_300Myr_ago']]) /3e8) * pynbody.units.Unit('yr**-1') # in Msol yr^-1

        return Mstar_30kpc, sfr_integrated

    def region_specification(self, existing):
        return pynbody.filt.Sphere('30 kpc',
                                   existing['shrink_center'])

    def requires_property(self):
        return ["shrink_center", 'aexp_300Myr_ago']


class morphokinematics(PynbodyPropertyCalculation):
    names = 'kappa_co', 'discfrac', 'vrot_over_sigma', 'disp_anisotropy', 'stellar_J_mag'

    def calculate(self, pdata, existing):

        from kinematics import kinematics_diagnostics

        with pynbody.transformation.translate(pdata, -existing['shrink_center']):

            pdata.wrap()

            kappa, discfrac, vrotsig, delta, zaxis, Momentum = kinematics_diagnostics(pdata.star['pos'],pdata.star['mass'],pdata.star['vel'],aperture=30.,CoMvelocity=True)

            return kappa, discfrac, vrotsig, delta, Momentum

    def requires_property(self):
        return ["shrink_center"]


