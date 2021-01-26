from tangos.properties.pynbody import PynbodyPropertyCalculation
from tangos.properties import PropertyCalculation
from tangos.properties import LivePropertyCalculation
import pynbody
import numpy as np

from astropy.cosmology import z_at_value
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from sys import stdout

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


class formed_hotphase(PynbodyPropertyCalculation):
    names = 'f_star_hot'
    
    def calculate(self, pdata, existing):

        star_mass = pdata.star['mass']
        maxtemp = pdata.star['MaximumTemperature']

        return np.sum(star_mass[maxtemp>np.power(10.,5.5)])/np.sum(star_mass)

    def region_specification(self, existing):
        return pynbody.filt.Sphere(existing['r200'],
                                   existing['shrink_center'])

    def requires_property(self):
        return ["shrink_center", "r200"]


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
    names = 'kappa_co', 'discfrac', 'vrot_over_sigma', 'disp_anisotropy', 'stellar_J_mag', 'ellipticity', 'triaxiality', 'alpha_T'

    '''
    This property is slightly over-complicated to allow the calculation of kinematic quantities for trackers.
    The issue is that the properties must be calculated with respect to some origin, and the tracker particles don't know where that is.
    When computing for a normal galaxy, can just use the pre-calculated shrink_center, but the tracker doesn't have this property.
    Instead, we need to find the subfind gn/sgn most of the tracker stars are associated with, and find the centre of that.
    All morphokinematics will be calculated with this method for consistency. I think it should give the same answer for non-trackers.
    '''

    def preloop(self, particle_data, timestep_object):
        self.centres, self.hnumber, self.dbid, self.finder_id = timestep_object.calculate_all("shrink_center","halo_number()","dbid()","finder_id()")

    def calculate(self, pdata, existing):

        
        from kinematics import kinematics_diagnostics, morphological_diagnostics

        tsgn = np.array(pdata.star['TangosSubGroupNumber'],dtype=np.int64)

        # Find the most common tangos halo
        values, counts = np.unique(tsgn,return_counts=True)
        tangossubgroup = values[np.argmax(counts)]

        centre = self.centres[self.finder_id==tangossubgroup]

        pos = np.array(pdata.star['pos']) - centre
        mass = np.array(pdata.star['mass'])
        vel = np.array(pdata.star['vel'])

        kappa, discfrac, vrotsig, delta, zaxis, Momentum = kinematics_diagnostics(pos,mass,vel,aperture=30.,CoMvelocity=True)

        ellip, triax, transform, abc = morphological_diagnostics(pos,mass,vel,aperture=30.,CoMvelocity=True,reduced_structure=True)

        return kappa, discfrac, vrotsig, delta, Momentum, ellip, triax, (ellip**2 + 1 - triax)/2.




class HaloFilename(LivePropertyCalculation):
    names = "halo_filename"

    def live_calculate(self, halo):
        return str(halo.timestep.filename)


class exsitu(PynbodyPropertyCalculation):
    names = 'f_ex_situ', 'f_in_situ'

    def preloop(self, particle_data, timestep_object):

        # Get the main branches of each halo (expansion factor and TangosSubGroupNumber)

        self.treedict = {}

        for h in timestep_object.halos[:]:

            a, fid, fname = h.calculate_for_progenitors('a()','finder_id()','halo_filename()')

            # Identify each halo by its Tangos finder number at the current timestep
            h_identifier = str(fid[0])

            # Store the main branches, and the gas particles in them
            self.treedict[h_identifier] = {}
            self.treedict[h_identifier]['a'] = a
            self.treedict[h_identifier]['tangos_sgn'] = fid
            self.treedict[h_identifier]['filename'] = fname


    def calculate(self, pdata, existing):

        print('\n')

        # Grab the main branch of this halo
        # Reverse the order for easy searching - time and aexp will go in ascending order
        h_identifier = str(existing['finder_id'])
        branch_a = self.treedict[h_identifier]['a'][::-1]
        branch_tangos_sgn = self.treedict[h_identifier]['tangos_sgn'][::-1]
        branch_fname = self.treedict[h_identifier]['filename'][::-1]

        print('Halo ',h_identifier)
        print('a ',branch_a)
        print('tangos sgn ',branch_tangos_sgn)
        print('file name ',branch_fname)

        star_mass = np.array(pdata.star['mass'],dtype=np.float32)
        star_pid = np.array(pdata.star['iord'],dtype=np.int64)
        star_aexp_form = np.array(pdata.star['aform'],dtype=np.float32)

        # print(star_aexp_form)

        # Get the index in the branch before each star particle formed
        index_formed = np.searchsorted(branch_a,star_aexp_form) - 1

        print('branch index pre formation ',index_formed)

        # Boolean array to hold whether star formed in situ
        formed_insitu = np.ones(len(star_pid),dtype=bool)

        # Default is in-situ. Stars formed before the tree starts will be considered in situ.

        stdout.flush()

        for f in np.unique(index_formed):

            if f<0:
                continue

            print('branch index ',f)

            print('file name ',branch_fname[f])

            # The TangosSubGroupNumber we need for the SF to have been in-situ
            target_tangos_sgn = branch_tangos_sgn[f]

            # Load the gas particle IDs and TangosSubGroupNumbers
            snap = pynbody.load(branch_fname[f]+'/snap_'+branch_fname[f][-12:])
            gas_pid = np.array(snap.gas['iord'],dtype=np.int64)
            gas_tangos_sgn = np.array(snap.gas['TangosSubGroupNumber'],dtype=np.int64)

            # Gas particle ids in that target subgroup
            gas_pid = gas_pid[gas_tangos_sgn==target_tangos_sgn]

            # Indices of the star particles we're looking for in this snapshot
            to_look_for = np.where(index_formed == f)[0]

            print('looking for ',len(to_look_for),' of ',len(index_formed),' star particles')

            formed_insitu[to_look_for] = np.isin(star_pid[to_look_for],gas_pid)

            stdout.flush()

        print('in situ array ',formed_insitu)

        print(len(np.where(formed_insitu==True)[0]),' particles formed in situ')
        print(len(np.where(formed_insitu==False)[0]),' particles formed ex situ')

        f_ex_situ = np.sum(star_mass[formed_insitu==False])/np.sum(star_mass)

        print('Ex-situ fraction ',f_ex_situ)
        # print(star_pid)
        # print(star_aexp_form)
        stdout.flush()

        return f_ex_situ, 1.-f_ex_situ

        # for s in range(len(star_pid)):





    # def requires_property(self):
    #     return ["a()","finder_id()"]
