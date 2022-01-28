from tangos.properties.pynbody import PynbodyPropertyCalculation
from tangos.properties import PropertyCalculation
from tangos.properties import LivePropertyCalculation
from tangos import input_handlers
import pynbody
import numpy as np

# from astropy.cosmology import z_at_value
# import astropy.units as u
# from astropy.cosmology import FlatLambdaCDM
from sys import stdout
import h5py as h5
from glob import glob








class SUBFIND_centre(PynbodyPropertyCalculation):
    names = 'subfind_centre_potential'

    def preloop(self, particle_data, timestep_object):

        # Load in the GroupNumbers, SubGroupNumbers and centres of potential from SUBFIND

        from pathlib import Path

        fname = timestep_object.filename
        tag = fname[-12:]

        fname = Path(fname)
        subfind_cat = str(Path(fname.parent)) + '/groups_'+tag+'/eagle_subfind_tab_' + tag

        self.cat_gn = catalogue_read(subfind_cat,'GroupNumber','Subhalo')
        self.cat_sgn = catalogue_read(subfind_cat,'SubGroupNumber','Subhalo')
        self.cat_cop = catalogue_read(subfind_cat,'CentreOfPotential','Subhalo') * 1e3 * (particle_data.properties['a']/particle_data.properties['h'])


    def calculate(self, pdata, existing):

        gn = pdata.dm['GroupNumber'][0]
        sgn = pdata.dm['SubGroupNumber'][0]

        return self.cat_cop[np.where((self.cat_gn==gn)&(self.cat_sgn==sgn))[0]][0]



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
        

def nfw(r,args):
    return np.log10(args[0] / ((r/args[1])*(1.+(r/args[1]))**2))

def residuals(args, profile, r):
    return profile - nfw(r,args)
    
class concentration(PynbodyPropertyCalculation):
    names = "concentration"

    def preloop(self, particle_data, timestep_object):
        from scipy.stats import binned_statistic
        from scipy.optimize import least_squares
        
        self.bs = binned_statistic
        self.lsq = least_squares

    def calculate(self, particle_data, existing_properties):
        
        r200 = existing_properties['r200']
        print(r200)

        with pynbody.transformation.inverse_translate(particle_data,existing_properties['shrink_center']):
        
            particle_data.wrap()
            coords = particle_data.dm['pos']
            mass = particle_data.dm['mass']/1e10

            print(coords)

            r = np.sqrt(np.einsum('...j,...j->...',coords,coords))

            print(r)

            bins = np.logspace(1.,np.log10(r200),50)
            bincentres = (bins[:-1]+bins[1:])/2.

            print(bins)

            mass_binned,_,_ = self.bs(r,mass,bins=bins,statistic='sum')

            print(mass_binned)

            shell_volumes = (4./3.) * np.pi * (bins[1:]**3-bins[:-1]**3)

            print(shell_volumes)

            density_profile = np.log10(mass_binned/shell_volumes)

            print(density_profile)

            opt = self.lsq(residuals,np.array([10.,200.]),args=(density_profile,bincentres))

            rho_0, R_s = opt.x[0], opt.x[1]

            print(rho_0,R_s)

            return r200/R_s


    def region_specification(self, existing_properties):
        return pynbody.filt.Sphere(existing_properties['r200'],
                                   existing_properties['shrink_center'])

    def requires_property(self):
        return ["shrink_center", "r200"]






class CGMfractions(PynbodyPropertyCalculation):
    names = 'f_CGM', 'f_b', 'f_star', 'f_ISM'

    def preloop(self, particle_data, timestep_object):

        self.handler = timestep_object.simulation.output_handler_class


    def calculate(self, pdata, existing):

        f_b_cosmic = pdata.properties['omegaB0']/pdata.properties['omegaM0']

        gas_mass = pdata.g['mass']
        star_mass = pdata.star['mass']

        if self.handler is input_handlers.eagle.EagleLikeInputHandler:
            sfr = pdata.g['StarFormationRate']
            sf = np.where(sfr>0.)[0]
            nsf = np.where(sfr==0.)[0]

        elif self.handler is input_handlers.pynbody.ChangaInputHandler:
            density = pdata.g['rho'].in_units('g cm^-3')
            temp = pdata.g['temp']
            sf = np.where((density>0.2*constants.m_p_cgs)&(temp<1e4))[0]
            nsf = np.where((density<0.2*constants.m_p_cgs)|(temp>1e4))[0]
        else:
            raise IOError('Unknown input handler')

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


class GalacticGas(PynbodyPropertyCalculation):
    names = 'Mgas_30kpc_sf', 'Mgas_30kpc_all'

    def preloop(self, particle_data, timestep_object):

        self.handler = timestep_object.simulation.output_handler_class

    def calculate(self, pdata, existing):

        gas_mass = pdata.g['mass']

        if self.handler is input_handlers.eagle.EagleLikeInputHandler:
            sfr = pdata.g['StarFormationRate']
            sf = np.where(sfr>0.)[0]
            # nsf = np.where(sfr==0.)[0]

        elif self.handler is input_handlers.pynbody.ChangaInputHandler:
            density = pdata.g['rho'].in_units('g cm^-3')
            temp = pdata.g['temp']
            sf = np.where((density>0.2*constants.m_p_cgs)&(temp<1e4))[0]
            # nsf = np.where((density<0.2*constants.m_p_cgs)|(temp>1e4))[0]
        else:
            raise IOError('Unknown input handler')

        return np.sum(gas_mass[sf]), np.sum(gas_mass)

    def region_specification(self, existing):
        return pynbody.filt.Sphere('30 kpc',existing['shrink_center'])

    def requires_property(self):
        return ["shrink_center"]




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
    # names = 'M_BH', 'Mdot_BH', 'E_AGN'
    names = 'M_BH', 'Mdot_BH'

    def preloop(self, particle_data, timestep_object):

        if timestep_object.simulation.output_handler_class is input_handlers.pynbody.ChangaInputHandler:
            raise IOError('Providing class is incompatible with CHANGA. Use the built-in BH class from tangos instead.')

    def calculate(self, pdata, existing):

        BH_erg_per_g = 0.1 * 0.15 * (299792458.*100.)**2 * pynbody.units.Unit('erg')/pynbody.units.Unit('g')

        bh_masses = pdata.bh['BH_Mass']
        print(np.log10(bh_masses))

        # When no BH mass is greater than 10^6.5 (approx. 2x seed mass), pick the BH closest to the centre to prevent instability in selection
        if not np.any(np.log10(bh_masses)>6.5):
            bh_r = pdata.bh['pos']-existing['shrink_center']
            bh_locate = np.argmin(np.einsum('...j,...j->...',bh_r,bh_r))
        else:
            bh_locate = np.argmax(bh_masses)


        E_AGN = (pdata.bh['BH_Mdot'].astype(np.float64).in_units('g s**-1') * BH_erg_per_g).in_units('erg s**-1')

        return bh_masses[bh_locate], pdata.bh['BH_Mdot'].in_units('Msol yr**-1')[bh_locate], E_AGN[bh_locate]

    def region_specification(self, existing):
        return pynbody.filt.Sphere(existing['r200'],
                                   existing['shrink_center'])

    def requires_property(self):
        return ["shrink_center", "r200"]



class BH_detailed_params(PynbodyPropertyCalculation):
    # names = 'M_BH', 'Mdot_BH', 'E_AGN'
    names = 'BH_CumlAccrMass', 'BH_CumlNumSeeds', 'BH_Density', 'BH_Pressure', 'BH_SoundSpeed', 'BH_Coordinates', 'BH_Velocity', 'BH_EddingtonRate', 'BH_EnergyReservoir', 'BH_AccretionLength'

    def preloop(self, particle_data, timestep_object):

        if timestep_object.simulation.output_handler_class is input_handlers.pynbody.ChangaInputHandler:
            raise IOError('Providing class is incompatible with CHANGA. Use the built-in BH class from tangos instead.')

    def calculate(self, pdata, existing):

        bh_masses = np.array(pdata.bh['BH_Mass'].in_units('Msol'),dtype=np.float64)
        # bh_locate = np.argmax(bh_masses)

        # When no BH mass is greater than 10^6.5 (approx. 2x seed mass), pick the BH closest to the centre to prevent instability in selection
        if not np.any(np.log10(bh_masses)>6.5):
            bh_r = pdata.bh['pos']-existing['shrink_center']
            bh_locate = np.argmin(np.einsum('...j,...j->...',bh_r,bh_r))
        else:
            bh_locate = np.argmax(bh_masses)

        cuml_accr_mass = pdata.bh['BH_CumlAccrMass']
        cuml_num_seeds =np.int32(pdata.bh['BH_CumlNumSeeds'])
        bh_density = pdata.bh['BH_Density'].in_units('g cm^-3')
        # bh_mdot = pdata.bh['BH_Mdot'].in_units('Msol yr^-1')
        bh_pressure = pdata.bh['BH_Pressure'].in_units('g cm^-1 s^-2')
        bh_soundspeed = pdata.bh['BH_SoundSpeed'].in_units('km s^-1')
        bh_energyreservoir = pdata.bh['BH_EnergyReservoir'].in_units('erg')
        bh_position = pdata.bh['pos']
        bh_vel = pdata.bh['vel']
        # bh_surr_vel = pdata.bh['BH_SurroundingGasVel']
        bh_accretionlength = pdata.bh['BH_AccretionLength']

        eddington_rate = ((np.float64(4.*np.pi)*np.float64(constants.G_cgs)*bh_masses*np.float64(constants.m_sol_cgs)*np.float64(constants.m_p_cgs))/(np.float64(0.1*constants.thompson_cgs)*np.float64(constants.c_CGS))) * np.float64(constants.year_s/constants.m_sol_cgs)

        return cuml_accr_mass[bh_locate], cuml_num_seeds[bh_locate], bh_density[bh_locate], bh_pressure[bh_locate], bh_soundspeed[bh_locate], bh_position[bh_locate], bh_vel[bh_locate], eddington_rate[bh_locate], bh_energyreservoir[bh_locate], bh_accretionlength[bh_locate]

    def region_specification(self, existing):
        return pynbody.filt.Sphere(existing['r200'],
                                   existing['shrink_center'])

    def requires_property(self):
        return ["shrink_center", "r200"]



class BH_vicinity(PynbodyPropertyCalculation):
    names = 'BH_soft_ap_mass', 'BH_kpc_ap_mass', 'BH_soft_ap_meandensity', 'BH_kpc_ap_meandensity', 'BH_soft_ap_mediandensity', 'BH_kpc_ap_mediandensity', 'BH_kpc_ap_vrot_over_sigma', 'BH_kpc_ap_sigma0', 'BH_kpc_ap_sigmaZ', 'BH_kpc_ap_kappa', 'BH_kpc_ap_J', 'BH_kpc_ap_vrad', 'BH_kpc_ap_vrad_in', 'BH_kpc_ap_vrad_out','BH_kpc_ap_N','BH_acclength_mass', 'BH_acclength_mediandensity', 'BH_acclength_J', 'BH_acclength_kappa', 'BH_Vphi'

    def preloop(self, particle_data, timestep_object):

        if timestep_object.simulation.output_handler_class is input_handlers.pynbody.ChangaInputHandler:
            raise IOError('Providing class is incompatible with CHANGA. Use the built-in BH class from tangos instead.')

        # Get the gravitational softening at this redshift
        z = timestep_object.redshift
        if z >= 2.8:
            self.softening = 1.33 * 1./(1.+z) # 1.33 comoving kpc
        else:
            self.softening = 0.35

        from kinematics import kinematics_diagnostics

        self.kinematics = kinematics_diagnostics


    def calculate(self, pdata, existing):

        N_softs_or_kpc = np.arange(1,11)

        BH_soft_ap_mass = np.zeros(10)
        BH_kpc_ap_mass = np.zeros(10)

        BH_soft_ap_meandensity = np.zeros(10)
        BH_kpc_ap_meandensity = np.zeros(10)

        BH_soft_ap_mediandensity = np.zeros(10)
        BH_kpc_ap_mediandensity = np.zeros(10)

        BH_kpc_ap_vrot_over_sigma = np.zeros(10)
        BH_kpc_ap_sigma0 = np.zeros(10)
        BH_kpc_ap_sigmaZ = np.zeros(10)
        BH_kpc_ap_kappa = np.zeros(10)
        BH_kpc_ap_J = np.zeros(10)
        BH_kpc_ap_vrad = np.zeros(10)
        BH_kpc_ap_vrad_in = np.zeros(10)
        BH_kpc_ap_vrad_out = np.zeros(10)

        BH_kpc_ap_N = np.zeros(10,dtype=np.int64)

        # # Centre the velocity wrt the current halo centre
        # try:
        #     pynbody.analysis.halo.vel_center(pdata)
        #     transformation.v_translate(target, -vcen)
        # except ValueError:
        #     # Velocity subtraction failed.

        with pynbody.transformation.inverse_translate(pdata,existing['BH_Coordinates']):

            pynbody.transformation.v_translate(pdata, -existing['BH_Velocity'])

            bh_h = existing['BH_AccretionLength']

            # Quantities within the accretion length
            particles = pdata.g[pynbody.filt.Sphere(bh_h)]
            mass = particles['mass'].in_units('Msol')

            if len(mass) == 0:
                BH_acclength_mass, BH_acclength_mediandensity, BH_acclength_J, BH_acclength_kappa = 0.,0.,0.,0.
            else:

                dens = particles['rho'].in_units('g cm^-3')
                pos = np.array(particles['pos'].in_units('kpc'))
                vel = np.array(particles['vel'].in_units('km s^-1'))
                BH_acclength_kappa, _, _, _, _, _, _, _, j = self.kinematics(pos,mass,vel,aperture=existing['BH_AccretionLength'],CoMvelocity=False)
                BH_acclength_J = j * constants.kpc_SI/1000. # Now in units of km^2 s^-1
                BH_acclength_mass = np.sum(mass)
                BH_acclength_mediandensity = np.median(dens)


                def kernel(r,h):
                    return (21./(2.*np.pi*h**3))*np.power(1.-r/h,4.)*(1.+4.*r/h)

                r = np.sqrt(np.einsum('...j,...j->...',pos,pos))

                # V_phi
                v_phi = np.linalg.norm(np.sum(np.cross(pos,vel)*np.array(mass)[:,np.newaxis]*kernel(r,bh_h)[:,np.newaxis])/(np.sum(np.array(mass)*kernel(r,bh_h))*bh_h))



            for n, nsofts in enumerate(N_softs_or_kpc):

                particles = pdata.g[pynbody.filt.Sphere('%f kpc'%(nsofts*self.softening))]

                mass = particles['mass'].in_units('Msol')

                if len(mass) == 0:
                    BH_soft_ap_mass[n], BH_soft_ap_meandensity[n], BH_soft_ap_mediandensity[n] = 0., 0., 0.
                else:
                    BH_soft_ap_mass[n] = np.sum(mass)
                    dens = particles['rho'].in_units('g cm^-3')
                    BH_soft_ap_meandensity[n] = np.mean(dens)
                    BH_soft_ap_mediandensity[n] = np.median(dens)


                particles = pdata.g[pynbody.filt.Sphere('%i kpc'%nsofts)]

                mass = particles['mass'].in_units('Msol')

                if len(mass) != 0:

                    BH_kpc_ap_N[n] = len(mass)

                    BH_kpc_ap_mass[n] = np.sum(mass)
                    dens = particles['rho'].in_units('g cm^-3')
                    BH_kpc_ap_meandensity[n] = np.mean(dens)
                    BH_kpc_ap_mediandensity[n] = np.median(dens)

                    pos = np.array(particles['pos'].in_units('kpc'))
                    vel = np.array(particles['vel'].in_units('km s^-1'))
                    r = np.sqrt(np.einsum('...j,...j->...',pos,pos))
                    r_hat = np.zeros(np.shape(pos))
                    for j in range(3): # probably a better way of doing this, but this is hacked from old code
                        r_hat[:,j] = pos[:,j]/r
                    vrad = np.einsum('...j,...j->...',vel,r_hat) # v . rhat -> radial velocity in km s-1

                    infall = np.where(vrad<0.)
                    out = np.where(vrad>0.)
                    BH_kpc_ap_vrad[n] = np.sum(vrad*mass)/np.sum(mass)
                    BH_kpc_ap_vrad_in[n] = np.sum(vrad[infall]*mass[infall])/np.sum(mass[infall])
                    BH_kpc_ap_vrad_out[n] = np.sum(vrad[out]*mass[out])/np.sum(mass[out])

                    BH_kpc_ap_kappa[n], discfrac, BH_kpc_ap_vrot_over_sigma[n], BH_kpc_ap_sigma0[n], BH_kpc_ap_sigmaZ[n], delta, zaxis, momentum_temp, j = self.kinematics(pos,mass,vel,aperture=np.float32(nsofts),CoMvelocity=False)

                    # Momentum currently in units of Msol kpc km s^-1
                    # We want a specific angular momentum

                    # BH_kpc_ap_J[n] = (momentum_temp * constants.kpc_SI/1000.) / np.sum(mass) # Now in units of km^2 s^-1
                    BH_kpc_ap_J[n] = j * constants.kpc_SI/1000. # Now in units of km^2 s^-1

                else:
                    BH_kpc_ap_N[n] = 0
                    BH_kpc_ap_mass[n], BH_kpc_ap_meandensity[n], BH_kpc_ap_mediandensity[n], BH_kpc_ap_kappa[n], BH_kpc_ap_vrot_over_sigma[n], BH_kpc_ap_sigma0[n], BH_kpc_ap_sigmaZ[n], BH_kpc_ap_J[n] = 0., 0., 0., -1., -1., -1., -1., -1.
                    BH_kpc_ap_vrad[n] = np.nan
                    BH_kpc_ap_vrad_in[n] = np.nan
                    BH_kpc_ap_vrad_out[n] = np.nan

        return BH_soft_ap_mass, BH_kpc_ap_mass, BH_soft_ap_meandensity, BH_kpc_ap_meandensity, BH_soft_ap_mediandensity, BH_kpc_ap_mediandensity, BH_kpc_ap_vrot_over_sigma, BH_kpc_ap_sigma0, BH_kpc_ap_sigmaZ, BH_kpc_ap_kappa, BH_kpc_ap_J, BH_kpc_ap_vrad, BH_kpc_ap_vrad_in, BH_kpc_ap_vrad_out, BH_kpc_ap_N,BH_acclength_mass, BH_acclength_mediandensity, BH_acclength_J, BH_acclength_kappa, v_phi



    def region_specification(self, existing):
        return pynbody.filt.Sphere(existing['r200'],
                                   existing['shrink_center'])

    def requires_property(self):
        return ["shrink_center", "r200", "BH_Coordinates", "BH_Velocity", "BH_AccretionLength"]








class stellar(PynbodyPropertyCalculation):
    names = 'Mstar_30kpc', 'SFR_300Myr', 'SFR_100Myr'

    def preloop(self, particle_data, timestep_object):

        self.handler = timestep_object.simulation.output_handler_class

        # if self.handler is input_handlers.eagle.EagleLikeInputHandler:
        #     cosmo = FlatLambdaCDM(100.*particle_data.properties['h'],Om0=particle_data.properties['omegaM0'],Ob0=particle_data.properties['omegaB0'])
        #     self.aexp_300Myr = 1./(1.+z_at_value(cosmo.age, (cosmo.age(timestep_object.redshift).value-0.3) * u.Gyr))

    def calculate(self, pdata, existing):

        Mstar_30kpc = pdata.star['mass'].sum()

        t_now = pdata.properties['time'].in_units("Gyr")
        tform = pdata.star['tform'].in_units("Gyr")
        mask_300Myr = (t_now-tform)<0.3
        mask_100Myr = (t_now-tform)<0.1

        # elif self.handler is input_handlers.eagle.EagleLikeInputHandler:
        #     formationtime = pdata.star['aform']
        #     sfr_integrated = (np.sum(pdata.star['InitialMass'].in_units('Msol')[formationtime>self.aexp_300Myr]) /3e8) * pynbody.units.Unit('yr**-1') # in Msol yr^-1

        return Mstar_30kpc, pdata.star['mass'][mask_300Myr].sum()/3e8, pdata.star['mass'][mask_100Myr].sum()/1e8

    def region_specification(self, existing):
        return pynbody.filt.Sphere('30 kpc',
                                   existing['shrink_center'])

    def requires_property(self):
        return ["shrink_center",]




class cooling(PynbodyPropertyCalculation):
    names = 't_cool_r200', 't_cool_0p5r200', 't_cool_0p2r200'

    def preloop(self, particle_data, timestep_object):

        import emission
        self.z = timestep_object.redshift
        if self.z > 8.989: # this is the furthest back the CLOUDY tables go
            self.do_timestep = False
        else:
            self.do_timestep = True
            self.cloudy = emission.cloudy(redshift=self.z)

    def calculate(self, pdata, existing):

        if not self.do_timestep:
            return 0., 0., 0.

        r200_fractions = [1.,0.5,0.2]
        tcools = np.zeros(len(r200_fractions))

        for f, fraction in enumerate(r200_fractions):

            particles = pdata.g[pynbody.filt.Sphere(fraction*existing['r200'], existing['shrink_center'])]

            sfrmask = np.where(particles['StarFormationRate']==0.)[0]

            mass = particles['mass'].in_units('g')[sfrmask]
            temp = particles['temp'][sfrmask]
            density = particles['rho'].in_units('g cm^-3')[sfrmask]
            internalenergy = particles['u'].in_units('cm^2 s^-2')[sfrmask]

            abunds = np.zeros((len(mass),11))

            abunds[:,0] = particles["SmoothedElementAbundance/Hydrogen"][sfrmask]
            abunds[:,1] = particles["SmoothedElementAbundance/Helium"][sfrmask]
            abunds[:,2] = particles["SmoothedElementAbundance/Carbon"][sfrmask]
            abunds[:,3] = particles["SmoothedElementAbundance/Nitrogen"][sfrmask]
            abunds[:,4] = particles["SmoothedElementAbundance/Oxygen"][sfrmask]
            abunds[:,5] = particles["SmoothedElementAbundance/Neon"][sfrmask]
            abunds[:,6] = particles["SmoothedElementAbundance/Magnesium"][sfrmask]
            abunds[:,7] = particles["SmoothedElementAbundance/Silicon"][sfrmask]
            abunds[:,8] = abunds[:,7]*0.6054160
            abunds[:,9] = abunds[:,7]*0.0941736
            abunds[:,10] = particles["SmoothedElementAbundance/Iron"][sfrmask]

            masses_in_u = np.array([1.00794,4.002602,12.0107,14.0067,15.9994,20.1797,24.3050,28.0855,32.065,40.078,55.845])
            num_ratios = np.zeros(np.shape(abunds))
            for col in range(len(abunds[0,:])): # convert mX/mtot to mX/mH
                num_ratios[:,col] = abunds[:,col] / abunds[:,0]
            for element in range(len(abunds[0,:])-1):
                num_ratios[:,element+1] *= masses_in_u[0]/masses_in_u[element+1] # convert mX/mH to nX/nH (H automatically done before)

            n_H = density * abunds[:,0]/constants.m_H_cgs # convert into nH cm^-3

            luminosity = self.cloudy.particle_luminosity(temp,n_H,num_ratios,mass,density)

            # t_cool = internalenergy*mass/luminosity
            # t_cool /= constants.Gyr_s # convert to Gyr

            excision = np.where(~((temp<np.power(10.,5.5))&(n_H>0.1)))[0]

            tcools[f] = (np.sum(internalenergy[excision]*mass[excision])/np.sum(luminosity[excision]))/constants.Gyr_s


        return tcools[0], tcools[1], tcools[2]



    def region_specification(self, existing):
        return pynbody.filt.Sphere(existing['r200'],
                                   existing['shrink_center'])
    def requires_property(self):
        return ["shrink_center", "r200"]






class morphokinematics(PynbodyPropertyCalculation):
    names = 'kappa_co', 'discfrac', 'vrot_over_sigma', 'sigma0', 'sigma_z', 'disp_anisotropy', 'stellar_J_mag','stellar_J_per_mass', 'ellipticity', 'triaxiality', 'alpha_T'

    '''
    This property is slightly over-complicated to allow the calculation of kinematic quantities for trackers.
    The issue is that the properties must be calculated with respect to some origin, and the tracker particles don't know where that is.
    When computing for a normal galaxy, can just use the pre-calculated shrink_center, but the tracker doesn't have this property.
    Instead, we need to find the subfind gn/sgn most of the tracker stars are associated with, and find the centre of that.
    All morphokinematics will be calculated with this method for consistency. I think it should give the same answer for non-trackers.
    '''

    def preloop(self, particle_data, timestep_object):

        from kinematics import kinematics_diagnostics, morphological_diagnostics

        self.kinematics = kinematics_diagnostics
        self.morphology = morphological_diagnostics

        self.handler = timestep_object.simulation.output_handler_class
        self.centres, self.finder_id = timestep_object.calculate_all("shrink_center","finder_id()")

        if self.handler is input_handlers.pynbody.ChangaInputHandler:
            snap = particle_data.halos(make_grp=True)
            self.iord = particle_data.star['iord']
            self.grp = particle_data.star['grp']

    def calculate(self, pdata, existing):

        if self.handler is input_handlers.pynbody.ChangaInputHandler:
            fid = self.grp[np.searchsorted(self.iord,pdata.star['iord'])]
        elif self.handler is input_handlers.eagle.EagleLikeInputHandler:
            fid = np.array(pdata.star['TangosSubGroupNumber'],dtype=np.int64)
        else:
            raise IOError('Unknown input handler.')

        # Find the most common tangos halo
        values, counts = np.unique(fid,return_counts=True)
        most_common_id = values[np.argmax(counts)]

        centre = self.centres[self.finder_id==most_common_id]

        pos = np.array(pdata.star['pos']) - centre
        mass = np.array(pdata.star['mass'])
        vel = np.array(pdata.star['vel'])

        kappa, discfrac, vrotsig, sigma0, sigmaZ, delta, zaxis, Momentum, j = self.kinematics(pos,mass,vel,aperture=30.,CoMvelocity=True)

        ellip, triax, transform, abc = self.morphology(pos,mass,vel,aperture=30.,CoMvelocity=True,reduced_structure=True)

        return kappa, discfrac, vrotsig, sigma0, sigmaZ, delta, Momentum, j, ellip, triax, (ellip**2 + 1 - triax)/2.




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





def catalogue_read(subfind_loc,quantity,table):
    '''
    Read in FOF or SUBFIND catalogues.
    NO unit corrections will be done.
    '''

    assert table in ['FOF','Subhalo'],'table must be either FOF or Subhalo'

    num_chunks = len(glob(subfind_loc+'*.hdf5'))

    for c in range(num_chunks):
        with h5.File(subfind_loc+'.%i.hdf5'%c,'r') as f:

            if c == 0:
                data_arr = np.array(f['/'+table+'/%s'%(quantity)])
            else:
                data_arr = np.append(data_arr,np.array(f['/'+table+'/%s'%(quantity)]),axis=0)

    return data_arr









class constants:
    '''
    Useful constants for working with EAGLE data
    '''

    unit_mass_cgs = np.float64(1.989e43)
    unit_time_cgs = 3.085678E19
    unit_density_cgs = 6.769911178294543e-31
    unit_velocity_cgs = 100000.
    unit_energy_cgs = 1.989E53
    unit_length_cgs = 3.085678E24
    unit_pressure_cgs = 6.769911178294542E-21

    m_sol_SI = 1.989e30
    m_sol_cgs = 1.989e33

    kpc_SI = 3.0856e16 * 1e3
    Mpc_SI = 3.0856e16 * 1e6
    Mpc_cgs = Mpc_SI * 100.
    kpc_cgs = kpc_SI * 100.
    G_SI = 6.6726e-11
    G_cgs = 6.6726e-8

    Gyr_s = 3.1536e16
    year_s = Gyr_s/1e9

    c_SI = 299792458.
    c_CGS = c_SI*100.

    BHAR_cgs = 6.445909132449984e23
    BH_erg_per_g = (0.1 * 0.15 * c_CGS**2) # eps_r eps_f c**2
    SN_erg_per_g = 8.73e15

    m_H_cgs = 1.6737e-24
    m_p_cgs = 1.6726219e-24

    Z_sol = 0.0127

    boltzmann_cgs = np.float64(1.38064852e-16)
    boltzmann_eV_per_K = np.float64(8.61733035e-5)

    thompson_cgs = 6.65245e-25
    ergs_per_keV = 1.6021773e-9
