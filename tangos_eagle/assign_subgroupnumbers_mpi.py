'''
MPI script for assigning SubGroupNumber and ParticleBindingEnergy to EAGLE snapshots, based on the SUBFIND output particle data.
Unbound particles are assigned SubGroupNumber 2**30 and ParticleBindingEnergy 0.
The path to the simulation data is set by a combination of simulation_dir and run_dir, which should be edited by the user.
'''

import numpy as np
import h5py as h5
from sys import argv, exit, stdout
import glob
from pathlib import Path

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def split(container, count):
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]

def flush():
    stdout.flush()


########################################################################

simulation_dir = '/share/rcifdata/jdavies/simulations/'

run_dir = str(argv[1])

########################################################################

full_path = simulation_dir+run_dir+'/data/'

tags_split = None


if rank == 0:
    # Get our snapshot identifiers
    simpaths = sorted(glob.glob(full_path+'snapshot*'))
    tags = [p[-12:] for p in simpaths]
    # Split into however many cores are available.
    tags_split = split(tags, size)


# Scatter jobs across cores.
tags_split = comm.scatter(tags_split, root=0)

print('Rank: ',rank, ', recvbuf received: ',tags_split)


for t, tag in enumerate(tags_split):
    print(tag); flush()

    # Load the bound particles in as one big array #################################################################################
    print('Loading subfind particle data...'); flush()

    num_chunks = len(glob.glob(full_path+'particledata_'+tag+'/eagle*.hdf5'))

    pdata = {}

    empty = False

    with h5.File(full_path+'particledata_'+tag+'/eagle_subfind_particles_'+tag+'.0.hdf5','r') as f:

        num_part = f['Header'].attrs['NumPart_Total']
        avail_ptypes = ['PartType%i'%p for p, num in enumerate(num_part) if num>0]
        avail_ptype_inds = [p for p, num in enumerate(num_part) if num>0]

        if not avail_ptypes:
            print('No bound particles in this snapshot'); flush()
            empty = True
        else:
            for ptype in avail_ptypes:
                pdata[ptype] = {}

                pdata[ptype]['ParticleIDs'] = np.empty(num_part[int(ptype[-1])],dtype=np.int64)
                pdata[ptype]['SubGroupNumber'] = np.empty(num_part[int(ptype[-1])],dtype=np.int64)
                pdata[ptype]['ParticleBindingEnergy'] = np.empty(num_part[int(ptype[-1])],dtype=np.float32)

        f.close()

    if not empty:

        offsets = np.zeros(len(avail_ptypes),dtype=int)

        for c in range(num_chunks):
            with h5.File(full_path+'particledata_'+tag+'/eagle_subfind_particles_'+tag+'.%i.hdf5'%c,'r') as f:

                n_this_file = f['Header'].attrs['NumPart_ThisFile'][avail_ptype_inds]

                for p, ptype in enumerate(avail_ptypes):

                    if n_this_file[p]>0:
                        pdata[ptype]['ParticleIDs'][offsets[p]:offsets[p]+n_this_file[p]] = np.array(f[ptype+'/ParticleIDs'],dtype=np.int64)
                        pdata[ptype]['SubGroupNumber'][offsets[p]:offsets[p]+n_this_file[p]] = np.array(f[ptype+'/SubGroupNumber'],dtype=np.int64)
                        pdata[ptype]['ParticleBindingEnergy'][offsets[p]:offsets[p]+n_this_file[p]] = np.array(f[ptype+'/SubGroupNumber'],dtype=np.float32)

                offsets += n_this_file

                f.close()

    # Sort the particledata for easier matching
    print('Sorting subfind particles by their IDs...'); flush()
    for ptype in avail_ptypes:
        psort = np.argsort(pdata[ptype]['ParticleIDs'])
        pdata[ptype]['ParticleIDs'] = pdata[ptype]['ParticleIDs'][psort]
        pdata[ptype]['SubGroupNumber'] = pdata[ptype]['SubGroupNumber'][psort]
        pdata[ptype]['ParticleBindingEnergy'] = pdata[ptype]['ParticleBindingEnergy'][psort]

    # Now load in each snapshot chunk sequentially and match #################################################################################

    files = glob.glob(full_path+'snapshot_'+tag+'/snap*.hdf5')

    print ('Matching...'); flush()
    for fp, filepath in enumerate(files):
        print('File ',fp+1,' of ',len(files)); flush()

        with h5.File(filepath,'a') as f:

            num_this_chunk = f['Header'].attrs['NumPart_ThisFile']
            snap_ptypes = ['PartType%i'%p for p, num in enumerate(num_this_chunk) if num>0]
            snap_ptype_inds = [p for p, num in enumerate(num_this_chunk) if num>0]

            for p, ptype in enumerate(snap_ptypes):

                if 'SubGroupNumber' in f[ptype].keys():
                    del f[ptype]['SubGroupNumber']
                if 'ParticleBindingEnergy' in f[ptype].keys():
                    del f[ptype]['ParticleBindingEnergy']

                if empty or ptype not in avail_ptypes:

                    sgn_dataset = f[ptype].create_dataset('SubGroupNumber',data=np.ones(num_this_chunk[snap_ptype_inds[p]],dtype=np.int64)*2**30)
                    bind_dataset = f[ptype].create_dataset('ParticleBindingEnergy',data=np.zeros(num_this_chunk[snap_ptype_inds[p]],dtype=np.float32))

                else:
                    # Do matching!

                    snap_pids = f[ptype]['ParticleIDs']
                    matched_sgns = np.empty(len(snap_pids),dtype=np.int64)
                    matched_bind = np.empty(len(snap_pids),dtype=np.float32)

                    is_bound = np.isin(snap_pids,pdata[ptype]['ParticleIDs'])

                    # Assign "unbound" value to unbound particles
                    matched_sgns[~is_bound] = 2**30
                    matched_bind[~is_bound] = 0.

                    # Match subgroupnumbers from particledata.
                    # Since we've already checked the particles exist in the particledata, searchsorted can be used

                    match = np.searchsorted(pdata[ptype]['ParticleIDs'],snap_pids[is_bound])

                    matched_sgns[is_bound] = pdata[ptype]['SubGroupNumber'][match]
                    matched_bind[is_bound] = pdata[ptype]['ParticleBindingEnergy'][match]

                    sgn_dataset = f[ptype].create_dataset('SubGroupNumber',data=matched_sgns)
                    bind_dataset = f[ptype].create_dataset('ParticleBindingEnergy',data=matched_bind)

                sgn_dataset.attrs['CGSConversionFactor'] = np.float64(1.)
                sgn_dataset.attrs['VarDescription'] = 'SUBFIND subhalo the particle is in, matched from particledata files.'
                sgn_dataset.attrs['aexp-scale-exponent'] = np.float32(0.)
                sgn_dataset.attrs['h-scale-exponent'] = np.float32(0.)

                bind_dataset.attrs['CGSConversionFactor'] = np.float64(1.989e53)
                bind_dataset.attrs['VarDescription'] = 'Particle binding energy, units h^-1 U_E [erg]'
                bind_dataset.attrs['aexp-scale-exponent'] = np.float32(0.)
                bind_dataset.attrs['h-scale-exponent'] = np.float32(-1.)


            f.close()
