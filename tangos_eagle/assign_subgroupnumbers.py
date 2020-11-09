import numpy as np
import h5py as h5
from sys import argv, exit, stdout
import glob
from pathlib import Path


# from mpi4py import MPI

# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()


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

# Get our snapshot identifiers
simpaths = sorted(glob.glob(full_path+'snapshot*'))
tags = [p[-12:] for p in simpaths]

for t, tag in enumerate(tags):
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

                offsets += n_this_file

                f.close()

    # Sort the particledata for easier matching
    print('Sorting subfind particles by their IDs...'); flush()
    for ptype in avail_ptypes:
        psort = np.argsort(pdata[ptype]['ParticleIDs'])
        pdata[ptype]['ParticleIDs'] = pdata[ptype]['ParticleIDs'][psort]
        pdata[ptype]['SubGroupNumber'] = pdata[ptype]['SubGroupNumber'][psort]

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

                if empty or ptype not in avail_ptypes:

                    sgn_dataset = f[ptype].create_dataset('SubGroupNumber',data=np.ones(num_this_chunk[snap_ptype_inds[p]],dtype=np.int64)*2**30)
                
                else:
                    # Do matching!
                    
                    snap_pids = f[ptype]['ParticleIDs']
                    matched_sgns = np.empty(len(snap_pids),dtype=np.int64)

                    is_bound = np.isin(snap_pids,pdata[ptype]['ParticleIDs'])

                    # Assign "unbound" value to unbound particles
                    matched_sgns[~is_bound] = 2**30

                    # Match subgroupnumbers from particledata.
                    # Since we've already checked the particles exist in the particledata, searchsorted can be used
                    matched_sgns[is_bound] = pdata[ptype]['SubGroupNumber'][np.searchsorted(pdata[ptype]['ParticleIDs'],snap_pids[is_bound])]

                    sgn_dataset = f[ptype].create_dataset('SubGroupNumber',data=matched_sgns)

                sgn_dataset.attrs['CGSConversionFactor'] = np.float64(1.)
                sgn_dataset.attrs['VarDescription'] = 'SUBFIND subhalo the particle is in, matched from particledata files.'
                sgn_dataset.attrs['aexp-scale-exponent'] = np.float32(0.)
                sgn_dataset.attrs['h-scale-exponent'] = np.float32(0.)
                

            f.close()













# all_snaps = [p.name for p in Path(full_path).rglob('snap*.hdf5')]

# tags = [n[5:17] for n in all_snaps]


# for s, snap in enumerate(all_snaps):

#     with h5.File(full_path+'snapshot_'+tags[s]+'/'+all_snaps[s],'r') as f:

#         keys = list(f.keys())
#         avail_ptypes = [key for key in keys if 'PartType' in key]
#         print(avail_ptypes)
#         exit()




# # # Get our snapshot identifiers
# # simpaths = sorted(glob.glob(full_path+'snapshot*'))
# # tags = [p[-12:] for p in simpaths][::-1]

# # print(tags)
