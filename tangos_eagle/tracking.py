import numpy as np
from sys import exit, argv, stdout

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pynbody
import tangos
from tangos import tracking


class Tracking:
    '''
    Tools for identifying particles to track based on already-calculated halo quantities.
    For example, one might want to track all the star particles that were in the galaxy at the point of its max Vrot/sigma
    ...or maybe track the gas that was in the CGM when it was at its most gas-rich.

    Initialises a dictionary containing the history of your desired quantity for the ICs you specify, and containing strings
    pointing to each tangos simulation/timestep/halo.

    These strings can be used to load the particles at the timestep/halo of interest, and the class contains a function for
    creating a tangos tracker with these particles.
    '''


    def __init__(self,quantity,models,ics,
                        halo = 0,
                        volume='halo_049',
                        num_seeds=0,
                        default_seed='SEED42',
                        sim_base='EAGLE_GM_'):

            # If only one model/IC/property specified, cast into a list
            if isinstance(models,str):
                models = [models,]
            if isinstance(ics,str):
                ics = [ics,]
            if isinstance(models,str):
                properties = [properties,]

            self.models = models
            self.ics = ics
            self.quantity = quantity
            self.halo = halo
            self.volume = volume
            self.num_seeds = num_seeds
            self.default_seed = default_seed
            self.sim_base = sim_base

            self.seeds = [self.default_seed,]
            self.seeds.extend(['SEED%02d'%s for s in np.arange(self.num_seeds)+1])

            # Initialise storage dictionary
            self.histories = {}

            for m, model in enumerate(self.models):

                self.histories[model] = {}

                if model == 'DMONLY':
                    self.seeds = [default_seed,]

                for i, ic in enumerate(self.ics):

                    self.histories[model][ic] = {}
                    
                    for s, seed in enumerate(self.seeds):

                        self.histories[model][ic][seed] = {}

                        s = tangos.get_simulation(self.sim_base+'_'+volume+'_'+model+'_'+ic+'_'+seed)

                        h = s.timesteps[-1].halos[self.halo]

                        if quantity == 'SSFR':
                            sfr, t, paths = h.calculate_for_progenitors('SFR_300Myr',"t()",'path()')
                            Mstar = h.calculate_for_progenitors('Mstar_30kpc')
                            qu = sfr/Mstar

                        elif quantity == 'stellar_J_mag':
                            j, t, paths = h.calculate_for_progenitors('stellar_J_mag',"t()",'path()')
                            Mstar = h.calculate_for_progenitors('Mstar_30kpc')
                            qu = j/Mstar
                        else:
                            qu, t, paths = h.calculate_for_progenitors(self.quantity,"t()","path()")

                        self.histories[model][ic][seed]['data'] = qu
                        self.histories[model][ic][seed]['t'] = t

                        if 'EAGLE' in self.sim_base:

                            converted_paths = []
                            for path in paths:
                                converted_paths.append(self._convert_eagle_path(path))

                            paths = converted_paths

                        self.histories[model][ic][seed]['path'] = paths



    def create_tracker(self,particles,path):

        simpath = path.split('/')[0]
        tracking_number = tracking.new(simpath, particles)
        print('Created tracker has ID ',tracking_number); stdout.flush()


    def _convert_eagle_path(self,path):
        '''
        Convert an EAGLE path to one understandable by tangos.

        The paths returned by tangos include lots of /'s. The load functions expect the format sim/timestep/halo,
        where sim contains no /'s.

        This function reassembles the path into a format tangos can understand, replacing all the /'s in 
        sim with _ sql wildcards.

        '''

        split_path = path.split('/')

        simstr = split_path[:-3]
        timestepstr = 'data%%' + split_path[-2]
        halostr = split_path[-1]

        return '_'.join(simstr) + '/' + timestepstr + '/' + halostr




if __name__ == '__main__':

    seed = 'SEED42'

    # ics = ['z2_od_0p950','z2_od_1p000','z2_od_1p050']
    ics = ['z2_od_1p000','z2_od_1p050']

    track = Tracking('vrot_over_sigma','RECAL',ics,default_seed=seed,sim_base='EAGLE_GM')

    for ic in ics:

        history = track.histories['RECAL'][ic][seed]['data']
        paths = track.histories['RECAL'][ic][seed]['path']

        path = paths[np.argmax(history)]

        halo = tangos.get_halo(path)
        h = halo.load()

        track.create_tracker(h.stars,path)

    # print(h.load()) 
    # print(h.timestep.load_region(pynbody.filt.Sphere('1 kpc',h['shrink_center']))) 
    # print(h.timestep.load_region(pynbody.filt.Sphere('1000 kpc',h['shrink_center']))) 


    
    
