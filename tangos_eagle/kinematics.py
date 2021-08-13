# -*- coding: utf-8 -*-
'''
Modified version of Adrien Thob's routines, for use with tangos_eagle_gm.py
'''


import numpy as np
from scipy import interpolate,linalg


def cumsummedian(a,weights=None):
    """
    Compute the weighted median.

    Returns the median of the array elements.

    Parameters
    ----------
    a : array_like, shape (n, )
        Input array or object that can be converted to an array.
    weights : {array_like, shape (n, ), None}, optional
        Input array or object that can be converted to an array.

    Returns
    -------
    median : float

    """
    if weights is None:
        weights=np.ones(np.array(a).shape)

    A = np.array(a).astype('float')
    W = np.array(weights).astype('float')

    if not(np.product(np.isnan(A))):
        I = np.argsort(A)
        cumweight = np.hstack([0,np.cumsum(W[I])])
        X = np.hstack([0,(cumweight[:-1]+cumweight[1:])/(2*cumweight[-1]),1])
        Y = np.hstack([np.min(A),A[I],np.max(A)])
        P = interpolate.interp1d(X,Y)(0.5)
        return float(P)
    else:
        return np.nan

def center_and_unloop(XYZ,XYZ0,BoxL=np.inf):
    """
    Center and unloop the input coordinates.

    Returns the centered and unlooped coordinates.

    Parameters
    ----------
    XYZ : array_like of dtype float, shape (n, 3)
        Particles coordinates (in unit of length L) such that XYZ[:,0] = X,
        XYZ[:,1] = Y & XYZ[:,2] = Z
    XYZ0 : array_like of dtype float, shape (3, )
        Centre coordinates (in unit of length L) such that XYZ0[0] = X0,
        XYZ0[1] = Y0 & XYZ0[2] = Z0
    BoxL : float, optional
        Length of the looping cubical box. Default is infinity

    Returns
    -------
    XYZ_out : array of dtype float, shape (n, 3)
        Centered and unlooped particles coordinates (in unit of length L) such
        that XYZ[:,0] = X, XYZ[:,1] = Y & XYZ[:,2] = Z

    """

    XYZ_out = XYZ.copy()
    XYZ_out-=XYZ0

    if np.isfinite(BoxL):
        XYZ_out+=BoxL/2.
        XYZ_out%=BoxL
        XYZ_out-=BoxL/2.

    return XYZ_out

def kinematics_diagnostics(XYZ,mass,Vxyz,aperture=0.03,CoMvelocity=True,weighting=None):

    # MODDED- no longer requires binding energies, and does not compute orbital circularity

    """
    Compute the various kinematics diagnostics.

    Returns the kinematics diagnostics for the input particles.

    Parameters
    ----------
    XYZ : array_like of dtype float, shape (n, 3)
        Particles coordinates (in unit of length L) such that XYZ[:,0] = X,
        XYZ[:,1] = Y & XYZ[:,2] = Z
    mass : array_like of dtype float, shape (n, )
        Particles masses (in unit of mass M)
    Vxyz : array_like of dtype float, shape (n, 3)
        Particles coordinates (in unit of velocity V) such that Vxyz[:,0] = Vx,
        Vxyz[:,1] = Vy & Vxyz[:,2] = Vz
    PBE : array_like of dtype float, shape (n, )
        Particles specific binding energies
    aperture : float, optional
        Aperture (in unit of length L) for the computation. Default is 0.03 L
    CoMvelocity : bool, optional
        Boolean to allow the centering of velocities by the considered particles
        centre-of-mass velocity. Default to True
    weighting : array_like of dtype float, shape (n, )
        JD edit - function will now return a weighted kappa_co for all particles
        Can give weights for kappa_co, if weighting is None, weighted kappa = kappa.

    Returns
    -------
    kappa : float
        The kinetic energy fraction invested in co-rotation.
    discfrac : float
        The disc-to-total mass fraction estimated from the counter-rotating
        bulge.
    orbi : float
        The median orbital circularity of the particles values.
    vrotsig : float
        The rotation-to-dispersion ratio .
    delta : float
        The dispersion anisotropy.
    zaxis : array of dtype float, shape (3, )
        The unit vector of the momentum axis (pointing along the momentum direction).
    Momentum : float
        The momentum magnitude (in unit M.L.V).

    """

    # lazy patch for not requiring binding energy
    PBE = np.ones(len(mass))

    particlesall = np.vstack([XYZ.T,mass,Vxyz.T,PBE]).T

    # Compute distances
    distancesall = np.linalg.norm(particlesall[:,:3],axis=1)

    # Restrict particles
    extract = (distancesall<aperture)
    particles = particlesall[extract].copy()
    distances = distancesall[extract].copy()
    if weighting is not None:
        weighting = weighting[extract]
    Mass = np.sum(particles[:,3])

    if CoMvelocity:
        # Compute CoM velocty & correct
        dvVmass = np.nan_to_num(np.sum(particles[:,3][:,np.newaxis]*particles[:,4:7],axis=0)/Mass)
        particlesall[:,4:7]-=dvVmass
        particles[:,4:7]-=dvVmass

    # Compute momentum
    smomentums = np.cross(particles[:,:3],particles[:,4:7])
    momentum = np.sum(particles[:,3][:,np.newaxis]*smomentums,axis=0)
    Momentum = np.linalg.norm(momentum)

    # Compute cylindrical quantities
    zaxis = (momentum/Momentum)
    zheight = np.sum(zaxis*particles[:,:3],axis=1)
    cylposition = particles[:,:3]-zheight[:,np.newaxis]*[zaxis]
    cyldistances = np.sqrt(distances**2-zheight**2)
    smomentumz = np.sum(zaxis*smomentums,axis=1)
    vrots = smomentumz/cyldistances
    vrads = np.sum(cylposition*particles[:,4:7]/cyldistances[:,np.newaxis],axis=1)
    vheis = np.sum(zaxis*particles[:,4:7],axis=1)

    # Compute co-rotational kinetic energy fraction - JD edited
    particle_KE_corot = (particles[:,3]*vrots**2)
    particle_KE = particles[:,3]*(np.linalg.norm(particles[:,4:7],axis=1))**2

    Mvrot2 = np.sum(particle_KE_corot[vrots>0])
    kappa = Mvrot2/np.sum(particle_KE)

    # Compute disc-to-total ratio
    discfrac = 1-2*np.sum(particles[vrots<=0,3])/Mass

    # # Compute orbital circularity
    # sbindingenergy = particles[:,7]; sortE = np.argsort(sbindingenergy); unsortE = np.argsort(sortE)
    # jzE = np.vstack([sbindingenergy,smomentumz]).T[sortE]
    # orbital = (jzE[:,1]/np.maximum.accumulate(np.abs(jzE[:,1])))[unsortE]
    # orbi = np.median(orbital)

    # Compute rotation-to-dispersion and dispersion anisotropy
    Vrot = np.abs(cumsummedian(vrots,weights=particles[:,3]))
    SigmaXY = np.sqrt(np.average(np.sum(particles[:,[3]]*np.vstack([vrads,vrots]).T**2,axis=0)/Mass))#
    SigmaO = np.sqrt(SigmaXY**2-.5*Vrot**2)

    SigmaZ = np.sqrt(np.average(vheis**2,weights=particles[:,3]))
    vrotsig = Vrot/SigmaO
    delta = 1-(SigmaZ/SigmaO)**2

    # Return
    # JD edited - return the momentum per unit mass as well.
    return kappa,discfrac,vrotsig,SigmaO,SigmaZ,delta,zaxis,Momentum,Momentum/np.sum(particles[:,3])

def morphological_diagnostics(XYZ,mass,Vxyz,aperture=0.03,CoMvelocity=True,reduced_structure=True):
    """
    Compute the morphological diagnostics through the (reduced or not) inertia tensor.

    Returns the morphological diagnostics for the input particles.

    Parameters
    ----------
    ----------
    XYZ : array_like of dtype float, shape (n, 3)
        Particles coordinates (in unit of length L) such that XYZ[:,0] = X,
        XYZ[:,1] = Y & XYZ[:,2] = Z
    mass : array_like of dtype float, shape (n, )
        Particles masses (in unit of mass M)
    Vxyz : array_like of dtype float, shape (n, 3)
        Particles coordinates (in unit of velocity V) such that Vxyz[:,0] = Vx,
        Vxyz[:,1] = Vy & Vxyz[:,2] = Vz
    aperture : float, optional
        Aperture (in unit of length L) for the computation. Default is 0.03 L
    CoMvelocity : bool, optional
        Boolean to allow the centering of velocities by the considered particles
        centre-of-mass velocity. Default to True
    reduced_structure : bool, optional
        Boolean to allow the computation to adopt the iterative reduced form of the
        inertia tensor. Default to True

    Returns
    -------
    ellip : float
        The ellipticity parameter 1-c/a.
    triax : float
        The triaxiality parameter (a^2-b^2)/(a^2-c^2).
    Transform : array of dtype float, shape (3, 3)
        The orthogonal matrix representing the 3 axes as unit vectors: in real-world
        coordinates, Transform[0] = major, Transform[1] = inter, Transform[2] = minor.
    abc : array of dtype float, shape (3, )
        The corresponding (a,b,c) lengths (in unit of length L).

    """
    particlesall = np.vstack([XYZ.T,mass,Vxyz.T]).T

    # Compute distances
    distancesall = np.linalg.norm(particlesall[:,:3],axis=1)

    # Restrict particles
    extract = (distancesall<aperture)
    particles = particlesall[extract].copy()
    distances = distancesall[extract].copy()
    Mass = np.sum(particles[:,3])

    # Compute kinematic diagnostics
    if CoMvelocity:
        # Compute CoM velocty, correct
        dvVmass = np.nan_to_num(np.sum(particles[:,3][:,np.newaxis]*particles[:,4:7],axis=0)/Mass)
        particlesall[:,4:7]-=dvVmass
        particles[:,4:7]-=dvVmass

    # Compute momentum
    smomentums = np.cross(particlesall[:,:3],particlesall[:,4:7])
    momentum = np.sum(particles[:,3][:,np.newaxis]*smomentums[extract],axis=0)

    # Compute morphological diagnostics
    s = 1; q = 1; Rsphall = 1+reduced_structure*(distancesall-1); stop = False

    while not('structure' in locals()) or (reduced_structure and not(stop)):

        particles = particlesall[extract].copy()
        Rsph = Rsphall[extract]; Rsph/=np.median(Rsph)

        # Compute structure tensor
        structure = np.sum((particles[:,3]/Rsph**2)[:,np.newaxis,np.newaxis]*(np.matmul(particles[:,:3,np.newaxis],particles[:,np.newaxis,:3])),axis=0)/np.sum(particles[:,3]/Rsph**2)

        # Diagonalise structure tensor
        eigval,eigvec = linalg.eigh(structure)

        # Get structure direct oriented orthonormal base
        eigvec[:,2]*=np.round(np.sum(np.cross(eigvec[:,0],eigvec[:,1])*eigvec[:,2]))

        # Return minor axe
        structmainaxe = eigvec[:,np.argmin(eigval)].copy()

        # Permute base and align Y axis with minor axis in momentum direction
        sign = int(np.sign(np.sum(momentum*structmainaxe)+np.finfo(float).tiny))
        structmainaxe *= sign
        temp = np.array([1,sign,1])*(eigvec[:,(np.argmin(eigval)+np.array([(3+sign)//2,0,(3-sign)//2]))%3])
        eigval = eigval[(np.argmin(eigval)+np.array([(3+sign)//2,0,(3-sign)//2]))%3]

        # Permute base to align Z axis with major axis
        foo = (np.argmax(eigval)//2)*2
        temp = np.array([(-1)**(1+foo//2),1,1])*(temp[:,[2-foo,1,foo]])
        eigval = eigval[[2-foo,1,foo]]

        # Compute change of basis matrix
        transform = linalg.inv(temp)
        stop = (np.max((1-np.sqrt(eigval[:2]/eigval[2])/np.array([q,s]))**2)<1e-4)

        if (reduced_structure and not(stop)):
            q,s = np.sqrt(eigval[:2]/eigval[2])
            Rsphall = linalg.norm(np.matmul(transform,particlesall[:,:3,np.newaxis])[:,:,0]/np.array([q,s,1]),axis=1)
            extract = (Rsphall<aperture/(q*s)**(1/3.))

    Transform = transform.copy()
    ellip = 1-np.sqrt(eigval[1]/eigval[2])
    triax = (1-eigval[0]/eigval[2])/(1-eigval[1]/eigval[2])
    Transform = Transform[...,[2,0,1],:] #so that transform[0] = major, transform[1] = inter, transform[2] = minor
    abc = np.sqrt(eigval[[2,0,1]])

    # Return
    return ellip,triax,Transform,abc
