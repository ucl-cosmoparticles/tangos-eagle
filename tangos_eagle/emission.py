# -*- coding: utf-8 -*-

import numpy as np
import h5py as h5
from sys import exit
import math
from copy import deepcopy



class apec(object):
    def __init__(self,apec_table_path,energy_band=[0.5,2.]):
        '''
        Class for the extraction of interpolated spectra from the APEC plasma emission model, and for computing particle X-ray luminosities with them.

        Initialise with:

        apec_table_path: str
        The path to the APEC emission table, included with this module under eagle_tools/emission_tables/

        energy_band: [min_energy,max_energy]
        The energy range of the chosen X-ray band (in keV) Default is the soft X-ray band (0.5-2.0 keV)

        '''

        # Load the cooling table
        apec_table = h5.File(apec_table_path,'r')
        cooling_table = apec_table['spectra']
        self.log_temp_bins = cooling_table['LOG_PLASMA_TEMP']   # ROWS ARE TEMPERATURE
        self.energy_bins = cooling_table['ENERGY'][0,:]  # COLUMNS ARE PHOTON ENERGY

        # Atomic data for the 11 elements in the cooling table
        self.element_list = np.array([['H','HYDROGEN'],['He','HELIUM'],['C','CARBON'],['N','NITROGEN'],['O','OXYGEN'],['Ne','NEON'],\
                                      ['Mg','MAGNESIUM'],['Si','SILICON'],['S','SULPHUR'],['Ca','CALCIUM'],['Fe','IRON']])
        self.atomic_numbers = np.array([1.,2.,6.,7.,8.,10.,12.,14.,16.,20.,26.])
        self.masses_in_u = np.array([1.00794,4.002602,12.0107,14.0067,15.9994,20.1797,24.3050,28.0855,32.065,40.078,55.845])
        self.AG_abundances = 10**(np.array([12.,10.99,8.56,8.05,8.93,8.09,7.58,7.55,7.21,6.36,7.67])-12.) # APEC assumes the solar abundances from Anders & Grevesse 1989 in n_X/n_H

        # Mask the cooling table to the chosen energy band.
        self.energy_band = energy_band
        self.Ebinwidth = self.energy_bins[1]-self.energy_bins[0] # Energy bin width
        self.energy_indices = np.where((self.energy_bins>=energy_band[0])&(self.energy_bins<=energy_band[1]))[0] # Mask of indices to integrate specta over
        self.energy_bins = self.energy_bins[self.energy_indices]

        self.table = {}

        for e, element in enumerate(self.element_list[:,1]):

            self.table[element] = cooling_table[element][:,self.energy_indices]


    def get_spectra(self,element,target_T):
        '''
        Extracts spectra from the APEC table, for a given element, interpolated to the given target temperatures.
        Assumes the solar abundance ratios of Anders & Grevesse 1989.

        Parameters:

        element: str
        The chosen element, in all caps, e.g. 'HYDROGEN', 'NEON'

        target_T: 1D array, length N
        The temperatures of the N particles to get spectra for.

        Returns:

        interpolated_spectra: 2D array, shape (N,len(energy_indices))
        The interpolated spectra for each temperature. Each row is the spectrum of a particle.
        '''

        # Take the logarithm of the target temperature
        target_T = np.log10(target_T)

        # If particle temperatures are out of the APEC range, set them to the limits of APEC (10^4-10^9 K)
        target_T[target_T>np.amax(self.log_temp_bins)] = np.amax(self.log_temp_bins)
        target_T[target_T<np.amin(self.log_temp_bins)] = np.amin(self.log_temp_bins)

        # Linearly interpolate the tables between redshifts according to (y-y0)/(x-x0) = (y1-y0)/(x1-x0)

        T_loc = np.searchsorted(self.log_temp_bins,target_T,side='left')

        T0 = self.log_temp_bins[T_loc-1]
        T1 = self.log_temp_bins[T_loc]

        spectra_0 = np.take(self.table[element], T_loc-1, axis=0)
        spectra_1 = np.take(self.table[element], T_loc, axis=0)

        return ((spectra_1-spectra_0) * ((target_T-T0)/(T1-T0))[:, np.newaxis]) + spectra_0



    def get_Xe_Xi_mu(self,abundances): # Convert mass abundances from GADGET (mX/mtot) to number abundances (nX/nH) and get Xe, Xi, mu
        '''
        Obtain Xe = n_e/n_H, Xi = n_i/n_H and the mean molecular weight, mu, for given particle mass abundances.

        Parameters:

        abundances: 2D array, shape (N,11)
        The mass abundances (m_X/m_tot) of all 11 elements tracked in EAGLE, stacked into an array where each row contains the abundances for a particle.

        Returns:

        Xe, Xi and mu as length-N arrays.
        '''

        assert len(abundances[0,:]) == 11, 'Must be 11 elements in input abundance array, i.e. shape (N,11)'

        # Initialise values for pure hydrogen
        Xe = np.ones(len(abundances[:,0]))
        Xi = np.ones(len(abundances[:,0]))
        mu = np.ones(len(abundances[:,0]))*0.5

        for col in range(len(abundances[0,:])): # convert mX/mtot to mX/mH
            abundances[:,col] /= abundances[:,0]

        for element in range(len(abundances[0,:])-1):
            mu += abundances[:,element+1]/(1.+self.atomic_numbers[element+1])
            Xe += abundances[:,element+1]*self.atomic_numbers[element+1] # Assuming complete ionisation
            Xi += abundances[:,element+1]

        return Xe, Xi, mu


    def convert_abundances(self,abundances):
        '''
        Convert particle mass abundances (m_X/m_tot) into abundance ratios by number (n_X/n_H)

        Parameters:

        abundances: 2D array, shape (N,11)
        The mass abundances (m_X/m_tot) of all 11 elements tracked in EAGLE, stacked into an array where each row contains the abundances for a particle.

        Returns:

        A 2D array, shape (N,11), of number abundance ratios.
        '''

        assert len(abundances[0,:]) == 11, 'Must be 11 elements in input abundance array, i.e. shape (N,11)'

        for col in range(len(abundances[0,:])): # convert mX/mtot to mX/mH
            abundances[:,col] /= abundances[:,0]

        for element in range(len(abundances[0,:])-1):
            abundances[:,element+1] *= self.masses_in_u[0]/self.masses_in_u[element+1] # convert mX/mH to nX/nH (H automatically done before)

        return abundances


    def cooling_function(self,temperature,abundances):
        '''
        Compute the total APEC cooling function for a set of particles, normalised to the particle abundance ratios.
        The cooling function is defined as the cooling rate per unit volume divided by n_e*n_H, in units of erg cm^6 s^-1.

        Parameters:

        temperature: 1D array, length N
        The particle temperatures [K]

        abundances: 2D array, shape (N,11)
        The mass abundances (m_X/m_tot) of all 11 elements tracked in EAGLE, stacked into an array where each row contains the abundances for a particle.

        Returns:

        total_cooling: 1D array, length N
        The cooling functions of the particles, normalised to the correct abundance ratios.

        '''

        # Convert m_X/m_tot to n_X/n_H
        num_abund_array = self.convert_abundances(abundances)

        # Initialise output array
        total_cooling = np.zeros(len(temperature))

        # Loop over 11 elements
        for e, element in enumerate(self.element_list[:,1]):

            # Interpolate spectra to correct temperature, integrate over energy band and normalise to correct element abundance, then add to total.
            # The rates in the tables are multiplied by the energy bin width, so to integrate the spectra we need only sum the array.
            total_cooling += np.sum(self.get_spectra(element,temperature),axis=1) * num_abund_array[:,e]/self.AG_abundances[e]

        return total_cooling


    def xray_luminosity_nochunks(self,temperature,density,mass,abundances):
        '''
        Directly compute the X-ray luminosity of a set of particles.

        N.B. The integration of the spectra is vectorised and does not loop over particles.
        For large numbers of particles this requires very large amounts of memory and will grind to a halt.
        Please use apec.xray_luminosity instead.

        Parameters:

        temperature: 1D array, length N
        The particle temperatures [K]

        density: 1D array, length N
        The particle densities in cgs units [g cm^-3]

        mass: 1D array, length N
        The particle masses in cgs units [g]

        abundances: 2D array, shape (N,11)
        The mass abundances (m_X/m_tot) of all 11 elements tracked in EAGLE, stacked into an array where each row contains the abundances for a particle.

        Returns:

        Lx_out: array, length N
        The X-ray luminosities of the particles in erg s^-1.
        '''

        m_H = 1.6737e-24 # in CGS units
        Xe, Xi, mu = self.get_Xe_Xi_mu(abundances)

        # Invoke the cooling_function function and multiply by n_e*n_H*V, following Crain et al (2010) and other papers.
        return (Xe/(Xe+Xi)**2) * (density/(mu*m_H)) * (mass/(mu*m_H)) * self.cooling_function(temperature,abundances)


    def xray_luminosity(self,temperature,density,mass,abundances,chunk_size=1000):
        '''
        Compute the X-ray luminosity of a set of particles.
        To greatly speed up the calculation, particles are split into chunks of size "chunk_size" (default is 1000)
        before the spectra are interpolated and the X-ray luminosity is computed in a vectorised way.

        Parameters:

        temperature: 1D array, length N
        The particle temperatures [K]

        density: 1D array, length N
        The particle densities in cgs units [g cm^-3]

        mass: 1D array, length N
        The particle masses in cgs units [g]

        abundances: 2D array, shape (N,11)
        The mass abundances (m_X/m_tot) of all 11 elements tracked in EAGLE, stacked into an array where each row contains the abundances for a particle.

        chunk_size: int, default=1000
        The size of the chunks to split the particles into, to facilitate the calculation for large numbers of particles.

        Returns:

        Lx_out: array, length N
        The X-ray luminosities of the particles in erg s^-1.
        '''

        m_H = 1.6737e-24 # in CGS units

        N = len(temperature)

        # Split the particles into chunks
        chunks = np.append(np.arange(N,step=chunk_size),N)

        Lx_out = np.empty(N)

        for c in range(len(chunks)-1):

            Xe, Xi, mu = self.get_Xe_Xi_mu(abundances[chunks[c]:chunks[c+1],:])

            Lx_out[chunks[c]:chunks[c+1]] = (Xe/(Xe+Xi)**2) * (density[chunks[c]:chunks[c+1]]/(mu*m_H)) * (mass[chunks[c]:chunks[c+1]]/(mu*m_H)) * self.cooling_function(temperature[chunks[c]:chunks[c+1]],abundances[chunks[c]:chunks[c+1],:])

        return Lx_out













class cloudy(object):
    def __init__(self,redshift=0.):

        self.metal_list = ['Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Sulphur','Calcium','Iron']

        self.metal_cooling = {}

        # If there isn't a CLOUDY output for our redshift, we need to linearly interpolate the tables

        # Load the CLOUDY output redshifts
        z_list = np.loadtxt('/share/rcifdata/jdavies/simulations/EAGLE/BG_Tables/CoolingTables/redshifts.dat')[1:]

        # If we have the right output, load it in and we're initialised
        if redshift in z_list:
            with h5.File('/share/rcifdata/jdavies/simulations/EAGLE/BG_Tables/CoolingTables/z_%.3f.hdf5'%(z_list[z_list==redshift]),'r') as table:

                self.solar_number_ratios = np.array(table['Header/Abundances/Solar_number_ratios'])
                self.n_H_bins = np.array(table['Total_Metals/Hydrogen_density_bins'])
                self.T_bins = np.array(table['Total_Metals/Temperature_bins'])
                self.He_bins = np.array(table['Metal_free/Helium_number_ratio_bins'])
                self.metal_free_cooling = np.array(table['Metal_free/Net_Cooling'])
                self.solar_electron_density = np.array(table['Solar/Electron_density_over_n_h'])
                self.electron_density = np.array(table['Metal_free/Electron_density_over_n_h'])

                for metal in self.metal_list:
                    self.metal_cooling[metal] = np.array(table[metal+'/Net_Cooling'])


        else:
            # Grab the two redshifts we need to interpolate between
            z_loc = np.searchsorted(z_list,redshift,side='left')
            z0 = z_list[z_loc-1]
            z1 = z_list[z_loc]

            metal_cooling_0 = {}
            metal_cooling_1 = {}

            with h5.File('/share/rcifdata/jdavies/simulations/EAGLE/BG_Tables/CoolingTables/z_%.3f.hdf5'%(z0),'r') as table:

                # The bins are redshift-independent, so might as well initialise them here
                self.solar_number_ratios = np.array(table['Header/Abundances/Solar_number_ratios'])
                self.n_H_bins = np.array(table['Total_Metals/Hydrogen_density_bins'])
                self.T_bins = np.array(table['Total_Metals/Temperature_bins'])
                self.He_bins = np.array(table['Metal_free/Helium_number_ratio_bins'])

                # Get the cooling tables at z0
                metal_free_cooling_0 = np.array(table['Metal_free/Net_Cooling'])
                solar_electron_density_0 = np.array(table['Solar/Electron_density_over_n_h'])
                electron_density_0 = np.array(table['Metal_free/Electron_density_over_n_h'])

                for metal in self.metal_list:
                    metal_cooling_0[metal] = np.array(table[metal+'/Net_Cooling'])


            with h5.File('/share/rcifdata/jdavies/simulations/EAGLE/BG_Tables/CoolingTables/z_%.3f.hdf5'%(z1),'r') as table:

                # ..and at z1
                metal_free_cooling_1= np.array(table['Metal_free/Net_Cooling'])
                solar_electron_density_1 = np.array(table['Solar/Electron_density_over_n_h'])
                electron_density_1 = np.array(table['Metal_free/Electron_density_over_n_h'])


                for metal in self.metal_list:
                    metal_cooling_1[metal] = np.array(table[metal+'/Net_Cooling'])

            # Linearly interpolate the tables between redshifts according to (y-y0)/(x-x0) = (y1-y0)/(x1-x0)

            self.metal_free_cooling = (((redshift-z0)/(z1-z0)) * (metal_free_cooling_1-metal_free_cooling_0)) + metal_free_cooling_0
            self.solar_electron_density = (((redshift-z0)/(z1-z0)) * (solar_electron_density_1-solar_electron_density_0)) + solar_electron_density_0
            self.electron_density = (((redshift-z0)/(z1-z0)) * (electron_density_1-electron_density_0)) + electron_density_0

            for metal in self.metal_list:

                self.metal_cooling[metal] = (((redshift-z0)/(z1-z0)) * (metal_cooling_1[metal]-metal_cooling_0[metal])) + metal_cooling_0[metal]



    def assign_cloudybins(self,p_T,p_n_H,p_He_ratio):
        '''
        Find the CLOUDY n_H, T and n_He/n_H bins for one or many particles
        '''

        temp_indices = searchsort_locate(self.T_bins,p_T)
        nH_indices = searchsort_locate(self.n_H_bins,p_n_H)
        heliumfrac_indices = searchsort_locate(self.He_bins,p_He_ratio)

        return temp_indices, nH_indices, heliumfrac_indices


    def cooling_rate_per_unit_volume_not_interpolated(self,p_T,p_n_H,p_num_ratio): # The particle abundances must be n_X/n_H of shape (N,11)

        '''
        Returns the cooling rate per unit volume - Lambda/n_H^2 [erg s^-1 cm^3] for one or many particles.
        '''

        p_He_ratio = p_num_ratio[:,1]/p_num_ratio[:,0]

        T_indices, n_H_indices, He_ratio_indices = self.assign_cloudybins(p_T, p_n_H, p_He_ratio)

        cooling_rate = self.metal_free_cooling[He_ratio_indices,T_indices,n_H_indices]

        for m, metal in enumerate(self.metal_list):
            # Add the contribution from each metal, normalised by the element abundance in the particle
            cooling_rate += self.metal_cooling[metal][T_indices,n_H_indices] * p_num_ratio[:,m+2]/self.solar_number_ratios[m+2]

        return cooling_rate



    def cooling_rate_per_unit_volume_interpolated(self,p_T,p_n_H,p_num_ratio): # The particle abundances must be n_X/n_H of shape (N,11)

        '''
        Returns the cooling rate per unit volume - Lambda/n_H^2 [erg s^-1 cm^3] for one or many particles.
        Obtains this by bilinearly interpolating the CLOUDY tables to the given T and nH, and trilinearly interpolates the metal-free contribution to the given nH, T, and He fraction
        '''

        # Copy these in memory before adjusting them
        p_T = deepcopy(p_T)
        p_n_H = deepcopy(p_n_H)
        p_num_ratio = deepcopy(p_num_ratio)

        p_He_ratio = p_num_ratio[:,1]/p_num_ratio[:,0]

        # If particle values are out of the CLOUDY bounds, set them to the limits of CLOUDY

        p_T[p_T>np.amax(self.T_bins)] = np.amax(self.T_bins)
        p_T[p_T<np.amin(self.T_bins)] = np.amin(self.T_bins)

        p_n_H[p_n_H>np.amax(self.n_H_bins)] = np.amax(self.n_H_bins)
        p_n_H[p_n_H<np.amin(self.n_H_bins)] = np.amin(self.n_H_bins)

        p_He_ratio[p_He_ratio>np.amax(self.He_bins)] = np.amax(self.He_bins)
        p_He_ratio[p_He_ratio<np.amin(self.He_bins)] = np.amin(self.He_bins)

        # Temperatures to interpolate between - these are ARRAYS - one element per particle
        T_loc = np.searchsorted(self.T_bins,p_T,side='left')
        T_loc[T_loc==0] = 1 # prevent errors when subtracting 1 from 1st element
        T1 = self.T_bins[T_loc-1]
        T2 = self.T_bins[T_loc]

        # Densities to interpolate between - these are ARRAYS - one element per particle
        n_loc = np.searchsorted(self.n_H_bins,p_n_H,side='left')
        n_loc[n_loc==0] = 1
        n1 = self.n_H_bins[n_loc-1]
        n2 = self.n_H_bins[n_loc]

        # He number ratios to interpolate between - these are ARRAYS - one element per particle
        He_loc = np.searchsorted(self.He_bins,p_He_ratio,side='left')
        He_loc[He_loc==0] = 1
        H1 = self.He_bins[He_loc-1]
        H2 = self.He_bins[He_loc]


        # We get the metal-free cooling rate and electron abundance using trilinear interpolation
        #T=x, nH=y, He=z

        # Get the coordinate differences
        dT = (p_T-T1)/(T2-T1)
        dn = (p_n_H-n1)/(n2-n1)
        dH = (p_He_ratio-H1)/(H2-H1)

        def interpolate_trilinear(datacube,i,j,k,dx,dy,dz):
            '''
            3D interpolation.
            '''
            Q00 = datacube[i-1,j-1,k-1] * (1.-dx) + datacube[i,j-1,k-1] * dx
            Q01 = datacube[i-1,j-1,k] * (1.-dx) + datacube[i,j-1,k] * dx
            Q10 = datacube[i-1,j,k-1] * (1.-dx) + datacube[i,j,k-1] * dx
            Q11 = datacube[i-1,j,k] * (1.-dx) + datacube[i,j,k] * dx

            Q0 = Q00*(1.-dy) + Q10*dy
            Q1 = Q01*(1.-dy) + Q11*dy

            return Q0*(1.-dz) + Q1*dz


        # Initialise the cooling rate with the metal-free contribution
        cooling_rate = interpolate_trilinear(self.metal_free_cooling,He_loc,T_loc,n_loc,dH,dT,dn)

        p_electron_density = interpolate_trilinear(self.electron_density,He_loc,T_loc,n_loc,dH,dT,dn)


        # Bilinearly interpolate the solar electron abundances

        Q11 = self.solar_electron_density[T_loc-1,n_loc-1]
        Q12 = self.solar_electron_density[T_loc-1,n_loc]
        Q21 = self.solar_electron_density[T_loc,n_loc-1]
        Q22 = self.solar_electron_density[T_loc,n_loc]
        p_solar_electron_density = ((T2-p_T)*(n2-p_n_H)*Q11 + (p_T-T1)*(n2-p_n_H)*Q21 + (T2-p_T)*(p_n_H-n1)*Q12 + (p_T-T1)*(p_n_H-n1)*Q22)/((T2-T1)*(n2-n1))


        for m, metal in enumerate(self.metal_list):

            # Now find the cooling rate at these four points to interpolate between

            L11 = self.metal_cooling[metal][T_loc-1,n_loc-1]
            L12 = self.metal_cooling[metal][T_loc-1,n_loc]
            L21 = self.metal_cooling[metal][T_loc,n_loc-1]
            L22 = self.metal_cooling[metal][T_loc,n_loc]

            # Bilinear interpolation
            interp_cooling = ((T2-p_T)*(n2-p_n_H)*L11 + (p_T-T1)*(n2-p_n_H)*L21 + (T2-p_T)*(p_n_H-n1)*L12 + (p_T-T1)*(p_n_H-n1)*L22)/((T2-T1)*(n2-n1))

            # Metal contributions to the cooling rate, following Wiersma et al. (2009)
            cooling_rate += interp_cooling * (p_electron_density/p_solar_electron_density) * p_num_ratio[:,m+2]/self.solar_number_ratios[m+2]

        return cooling_rate


    def particle_luminosity(self,p_T,p_n_H,p_num_ratio,p_mass,p_density):
        '''
        Compute the luminosity in erg s^-1 for one or many particles.
        Needs temperature [K], n_H [cm^-3], array of n_X/n_H for 11 elements, mass [g] and density [g cm^-3]
        '''

        emissivity_per_vol = self.cooling_rate_per_unit_volume_interpolated(p_T,p_n_H,p_num_ratio)

        return emissivity_per_vol * p_n_H**2 * (p_mass/p_density)

































############## OLD DEPRECATED FUNCTIONS ##################################################




def calculate_Lx(total_cooling,density,mass,mean_mol_weight,X_e,X_i):
    m_H = 1.6737e-24 # in CGS units
    return (X_e/(X_e+X_i)**2) * (density/(mean_mol_weight*m_H)) * (mass/(mean_mol_weight*m_H)) * total_cooling

def add_S_Ca(abund_array): # Input the abundances by mass, not by number. Adds Ca and S abundances based on ratios with Silicon.
    abund_array = np.insert(abund_array,8,abund_array[:,7]*0.6054160,axis=1) # Insert the Sulphur column
    abund_array = np.insert(abund_array,9,abund_array[:,7]*0.0941736,axis=1) # Insert the Calcium column
    return abund_array

def mass_to_num_abundance(abund_array): # returns abundances in n_X/n_H
    masses_in_u = np.array([1.00794,4.002602,12.0107,14.0067,15.9994,20.1797,24.3050,28.0855,32.065,40.078,55.845])
    for element in range(len(abund_array[0,:])):
        abund_array[:,element] *= masses_in_u[0]/masses_in_u[element]
        abund_array[:,element] /= abund_array[:,0]
    return abund_array

def mass_density_toCGS(mass,density): # Ensures mass and density arrays are to high enough precision and converts to CGS for Lx calc
    unit_mass_cgs = 1.989e43
    unit_density_cgs = 6.769911178294543e-31
    unit_length_cgs = 3.085678e24
    mass = mass.astype(np.float64)
    density = density.astype(np.float64)
    mass *= unit_mass_cgs
    density *= unit_density_cgs
    return mass, density

def get_Xe_Xi_mu(num_abund_array,abund_array): # mu requires mass abundances
    atomic_numbers = np.array([1.,2.,6.,7.,8.,10.,12.,14.,16.,20.,26.])
    Xe = np.ones(len(abund_array[:,0])) # Initialise values for hydrogen
    Xi = np.ones(len(abund_array[:,0]))
    mu = np.ones(len(abund_array[:,0]))*0.5

    for element in range(len(num_abund_array[0,:])-1):
        Xe += num_abund_array[:,element+1]*atomic_numbers[element+1] # Assuming complete ionisation
        Xi += num_abund_array[:,element+1]
        mu += abund_array[:,element+1]/(1.+atomic_numbers[element+1]) # Ian's method - not correct??
    return Xe,Xi,mu

def get_numfractions(abund_array): # Convert mass abundances (mX/mtot) to number (nX/nH) and get Xe, Xi, mu
    masses_in_u = np.array([1.00794,4.002602,12.0107,14.0067,15.9994,20.1797,24.3050,28.0855,32.065,40.078,55.845])
    atomic_numbers = np.array([1.,2.,6.,7.,8.,10.,12.,14.,16.,20.,26.])
    Xe = np.ones(len(abund_array[:,0])) # Initialise values for hydrogen
    Xi = np.ones(len(abund_array[:,0]))
    mu = np.ones(len(abund_array[:,0]))*0.5
    for col in range(len(abund_array[0,:])): # convert mX/mtot to mX/mH
        abund_array[:,col] /= abund_array[:,0]
    for element in range(len(abund_array[0,:])-1):
        mu += abund_array[:,element+1]/(1.+atomic_numbers[element+1])
        abund_array[:,element+1] *= masses_in_u[0]/masses_in_u[element+1] # convert mX/mH to nX/nH (H automatically done before)
        Xe += abund_array[:,element+1]*atomic_numbers[element+1] # Assuming complete ionisation
        Xi += abund_array[:,element+1]
    return abund_array, Xe, Xi, mu

def get_numfractions_from_m_over_mH(abund_array): # Convert mass abundances (mX/mH) to number (nX/nH) and get Xe, Xi, mu
    masses_in_u = np.array([1.00794,4.002602,12.0107,14.0067,15.9994,20.1797,24.3050,28.0855,32.065,40.078,55.845])
    atomic_numbers = np.array([1.,2.,6.,7.,8.,10.,12.,14.,16.,20.,26.])
    Xe = np.ones(len(abund_array[:,0])) # Initialise values for hydrogen
    Xi = np.ones(len(abund_array[:,0]))
    mu = np.ones(len(abund_array[:,0]))*0.5
    for element in range(len(abund_array[0,:])-1):
        mu += abund_array[:,element+1]/(1.+atomic_numbers[element+1])
        abund_array[:,element+1] *= masses_in_u[0]/masses_in_u[element+1] # convert mX/mH to nX/nH (H automatically done before)
        Xe += abund_array[:,element+1]*atomic_numbers[element+1] # Assuming complete ionisation
        Xi += abund_array[:,element+1]
    return abund_array, Xe, Xi, mu


def searchsort_locate(array,values):
    indices = np.searchsorted(array, values, side="left")

    for i in range(len(indices)):
        if indices[i] > 0 and (indices[i] == len(array) or math.fabs(values[i] - array[indices[i]-1]) < math.fabs(values[i] - array[indices[i]])):
            indices[i] -= 1

    return indices












class apec_old(object):
    def __init__(self,band_option):
        self.band_option = band_option
        if band_option == 'ROSAT':
            energy_range = [0.5,2.]
            path = '/hpcdata0/arijdav1/APEC_cooling_tables/APEC_spectra_0.02_80.0keV_res_10eV_interp.hdf5'
            self.Ebinwidth = 0.01 # in keV
        elif band_option == 'bolometric':
            energy_range = [0.02,80.]
            path = '/hpcdata0/arijdav1/APEC_cooling_tables/APEC_spectra_0.02_80.0keV_res_10eV_interp.hdf5'
        elif band_option == 'MASSIVE':
            energy_range = [0.3,5.]
            path = '/hpcdata0/arijdav1/APEC_cooling_tables/APEC_spectra_0.02_80.0keV_res_10eV_interp.hdf5'
        else:
            raise IOError('Band options are "ROSAT", "MASSIVE" or "bolometric"')

        apec_table = h5.File(path,'r')
        self.tabledict = apec_table['spectra']
        self.log_temp_bins = self.tabledict['LOG_PLASMA_TEMP']   # ROWS ARE TEMPERATURE
        self.energy_bins = self.tabledict['ENERGY'][0,:]  # COLUMNS ARE PHOTON ENERGY
        self.log_energy_bins = np.log10(self.energy_bins)
        self.n_Tbins = len(self.log_temp_bins)
        self.n_Ebins = len(self.energy_bins)
        self.AG_abundances = 10**(np.array([12.,10.99,8.56,8.05,8.93,8.09,7.58,7.55,7.21,6.36,7.67])-12.) # IN UNITS OF n_H!
        self.element_list = np.array([['H','HYDROGEN'],['He','HELIUM'],['C','CARBON'],['N','NITROGEN'],['O','OXYGEN'],['Ne','NEON'],\
                                      ['Mg','MAGNESIUM'],['Si','SILICON'],['S','SULPHUR'],['Ca','CALCIUM'],['Fe','IRON']])
        self.atomic_numbers = np.array([1,2,6,7,8,10,12,14,16,20,26])

        # Convert the APEC dictionary to a numpy array for speedier use
        self.tables = np.zeros((len(self.element_list[:,1]),self.n_Tbins,self.n_Ebins))
        for e, elem in enumerate(self.element_list[:,1]):
            self.tables[e,:,:] = self.tabledict[elem]

        self.int_mask = np.where((self.energy_bins>=energy_range[0])&(self.energy_bins<=energy_range[1]))[0] # Mask of indices to integrate specta over


    # def total_spectrum_plot(self,perkeV=True):
    #     plt.figure(figsize=(8,6))
    #     total_table = np.sum(self.tables,axis=0)
    #     temp_bins = np.array([6.,7.,8.,9.])
    #     num_plots = len(temp_bins)
    #     indices = np.searchsorted(self.log_temp_bins,temp_bins)
    #     colours = safe_colours.distinct_list(num_plots)
    #     ebins = 10.**self.log_energy_bins

    #     softband = np.where((ebins>0.5)&(ebins<2.))[0]

    #     for i in range(num_plots):
    #         if perkeV:
    #             spectrum = np.log10(total_table[indices[i],:]) - np.log10(self.Ebinwidth)
    #         else:
    #             spectrum = np.log10(total_table[indices[i],:])
    #         plt.plot(10**self.log_energy_bins,spectrum,c=colours[i],label=r'$T=10^{%i}\,\mathrm{K}$'%(int(temp_bins[i])))

    #     if perkeV:
    #         plt.ylabel(r'$\log\left(d\Lambda/dE_{\gamma}\right)\,[\mathrm{erg}\,\mathrm{cm}^{3}\,\mathrm{s}^{-1}\,\mathrm{keV}^{-1}]$',fontsize=16)
    #         plt.ylim(-28,-21)
    #     else:
    #         plt.ylim(-30,-23)
    #         plt.ylabel(r'$\log(\Lambda)\,[\mathrm{erg}\,\mathrm{cm}^{3}\,\mathrm{s}^{-1}]$',fontsize=16)

    #     plt.axvline(x=0.5,ls='--',c='gray')
    #     plt.axvline(x=2.,ls='--',c='gray')
    #     plt.legend(loc='lower right',frameon=False)
    #     plt.xlabel(r'$E_{\gamma}$ $[\mathrm{keV}]$',fontsize=16)
    #     plt.xlim(0.2,3.)

    #     plt.savefig('/home/arijdav1/Dropbox/phd/figures/cooling_calculations/totalspectra_%s.png'%(self.band_option),dpi=400,bbox_inches='tight')
    #     plt.show()

    # def indiv_elements_plot(self,logtemp=7.,perkeV=True):
    #     plt.figure(figsize=(8,6))
    #     #element_symbols = self.element_list[:,0]
    #     element_symbols = ['H','O','Ne','Mg','Si','Fe']
    #     el_indices = [np.where(self.element_list[:,0]==el)[0][0] for el in element_symbols]
    #     num_plots = len(element_symbols)
    #     temp_idx = (np.abs(10.**self.log_temp_bins-10**logtemp)).argmin()
    #     colours = safe_colours.distinct_list(num_plots)

    #     for i in range(num_plots):
    #         if perkeV:
    #             spectrum = np.log10(self.tables[el_indices[i],temp_idx,:]) - np.log10(self.Ebinwidth)
    #         else:
    #             spectrum = np.log10(self.tables[el_indices[i],temp_idx,:])
    #         plt.plot(10**self.log_energy_bins,spectrum,c=colours[i],label=element_symbols[i])

    #     if perkeV:
    #         plt.ylabel(r'$\log\left(d\lambda/dE_{\gamma}\right)\,[\mathrm{erg}\,\mathrm{cm}^{3}\,\mathrm{s}^{-1}\,\mathrm{keV}^{-1}]$',fontsize=16)
    #         plt.ylim(-27,-21.5)
    #     else:
    #         plt.ylim(-29,-23.5)
    #         plt.ylabel(r'$\log(\lambda)\,[\mathrm{erg}\,\mathrm{cm}^{3}\,\mathrm{s}^{-1}]$',fontsize=16)

    #     plt.annotate(r'$T=10^{%i}\,\mathrm{K}$'%(logtemp),xy=(2.3,-23.),fontsize=16)

    #     plt.axvline(x=0.5,ls='--',c='gray')
    #     plt.axvline(x=2.,ls='--',c='gray')

    #     plt.legend(loc='upper right',ncol=3,fancybox=True)
    #     plt.xlabel(r'$E_{\gamma}$ $[\mathrm{keV}]$',fontsize=16)
    #     plt.xlim(0.2,3.)
    #     plt.savefig('/home/arijdav1/Dropbox/phd/figures/cooling_calculations/elementspectra_%i_%s.png'%(logtemp,self.band_option),dpi=400,bbox_inches='tight')
    #     plt.show()

    def assign_curves(self,particle_temperatures): # Assign an APEC temperature index to each particle before performing cooling

        return searchsort_locate(self.log_temp_bins,np.log10(particle_temperatures))

    def single_element_cooling(self,el_idx,temp_idx,abund_number_ratio):
        spectrum = self.tables[el_idx,temp_idx,:]*abund_number_ratio/self.AG_abundances[el_idx] # Normalise spectrum to solar ratios of A&G '89
        return np.sum(spectrum[self.int_mask])

    def total_cooling(self,temp_idx,abund_number_ratio_list):
        total_rate = 0.
        for i in range(len(self.element_list[:,0])):
            total_rate += self.single_element_cooling(i,temp_idx,abund_number_ratio_list[i])
        return total_rate

    def single_element_spectrum(self,el_idx,temp_idx,abund_number_ratio):
        spectrum = self.tables[el_idx,temp_idx,:]*abund_number_ratio/self.AG_abundances[el_idx] # Normalise spectrum to solar ratios of A&G '89
        return spectrum[self.int_mask]

    def cooling_spectrum(self,temp_idx,abund_number_ratio_list,perkeV=True):
        total_spectrum = np.zeros(len(self.int_mask))
        for i in range(len(self.element_list[:,0])):
            total_spectrum += self.single_element_spectrum(i,temp_idx,abund_number_ratio_list[i])
        if perkeV:
            return total_spectrum/self.Ebinwidth
        else:
            return total_spectrum



class apec_new(object):
    def __init__(self,energy_band=[0.5,2.]):

        path = '/hpcdata0/arijdav1/APEC_cooling_tables/APEC_spectra_0.02_80.0keV_res_10eV_interp.hdf5'
        apec_table = h5.File(path,'r')
        self.Ebinwidth = 0.01 # in keV

        self.energy_band = energy_band
        cooling_table = apec_table['spectra']
        self.log_temp_bins = cooling_table['LOG_PLASMA_TEMP']   # ROWS ARE TEMPERATURE
        self.energy_bins = cooling_table['ENERGY'][0,:]  # COLUMNS ARE PHOTON ENERGY. Only consider bins within band


        self.element_list = np.array([['H','HYDROGEN'],['He','HELIUM'],['C','CARBON'],['N','NITROGEN'],['O','OXYGEN'],['Ne','NEON'],\
                                      ['Mg','MAGNESIUM'],['Si','SILICON'],['S','SULPHUR'],['Ca','CALCIUM'],['Fe','IRON']])

        self.atomic_numbers = np.array([1.,2.,6.,7.,8.,10.,12.,14.,16.,20.,26.])
        self.masses_in_u = np.array([1.00794,4.002602,12.0107,14.0067,15.9994,20.1797,24.3050,28.0855,32.065,40.078,55.845])

        self.AG_abundances = 10**(np.array([12.,10.99,8.56,8.05,8.93,8.09,7.58,7.55,7.21,6.36,7.67])-12.) # Abundances from Anders & Grevesse 1989 in n_X/n_H


        self.energy_indices = np.where((self.energy_bins>=energy_band[0])&(self.energy_bins<=energy_band[1]))[0] # Mask of indices to integrate specta over
        self.energy_bins = self.energy_bins[self.energy_indices]

        self.table = {}

        for e, element in enumerate(self.element_list[:,1]):

            self.table[element] = cooling_table[element][:,self.energy_indices]



    def get_spectrum(self,element,target_T):

        # Get the table for the given element and mask it to our energy band
        table = self.table[element]

        # Take the logarithm of the target temperature
        target_T = np.log10(target_T)

        # If particle temperatures are out of the APEC range, set them to the limits of APEC (10^4-10^9 K)

        if target_T > np.amax(self.log_temp_bins):
            target_T = np.amax(self.log_temp_bins)

        if target_T < np.amin(self.log_temp_bins):
            target_T = np.amin(self.log_temp_bins)

        # Linearly interpolate the tables between redshifts according to (y-y0)/(x-x0) = (y1-y0)/(x1-x0)

        T_loc = np.searchsorted(self.log_temp_bins,target_T,side='left')

        T0 = self.log_temp_bins[T_loc-1]
        T1 = self.log_temp_bins[T_loc]

        spectra_0 = np.take(table, T_loc-1, axis=0)
        spectra_1 = np.take(table, T_loc, axis=0)

        temp_interpolator = (target_T-T0)/(T1-T0)

        spectra_difference = spectra_1-spectra_0

        return (spectra_difference * temp_interpolator) + spectra_0


    def get_Xe_Xi_mu(self,abundances): # Convert mass abundances from GADGET (mX/mtot) to number abundances (nX/nH) and get Xe, Xi, mu

        assert len(abundances) == 11, 'Must be 11 elements in input abundance array, i.e. shape (N,11)'

        # Initialise values for pure hydrogen
        Xe, Xi, mu = 1., 1., 0.5

        # convert mX/mtot to mX/mH
        abundances[1:] /= abundances[0]

        for element in range(len(abundances)-1):
            mu += abundances[element+1]/(1.+self.atomic_numbers[element+1])
            Xe += abundances[element+1]*self.atomic_numbers[element+1] # Assuming complete ionisation
            Xi += abundances[element+1]

        return Xe, Xi, mu


    def convert_abundances(self,abundances):

        assert len(abundances) == 11, 'Must be 11 elements in input abundance array, i.e. shape (N,11)'

        # convert mX/mtot to mX/mH
        abundances[1:] /= abundances[0]

        # convert mX/mH to nX/nH
        abundances *= self.masses_in_u[0]/self.masses_in_u

        # for element in range(len(abundances)-1):
        #     abundances[:,element+1] *= self.masses_in_u[0]/self.masses_in_u[element+1] # convert mX/mH to nX/nH (H automatically done before)

        return abundances


    def cooling_rate(self,temperature,abundances):

        num_abundances = self.convert_abundances(abundances)

        total_cooling = 0.

        for e, element in enumerate(self.element_list[:,1]):

            abundance_normalisation = num_abundances[e]/self.AG_abundances[e]
            total_cooling += np.sum(self.get_spectrum(element,temperature)) * abundance_normalisation

        return total_cooling

    def xray_luminosity(self,temperature,density,mass,abundances):
        m_H = 1.6737e-24 # in CGS units
        Xe, Xi, mu = self.get_Xe_Xi_mu(abundances)

        return (Xe/(Xe+Xi)**2) * (density/(mu*m_H)) * (mass/(mu*m_H)) * self.cooling_rate(temperature,abundances)
