#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 16:05:48 2020

@author: ogurcan
"""
import numpy as np
import scipy.integrate as spi # for ODE solvers
import h5py as h5 # for saving the results
from time import time # keeping track of time ellapsed.
from scipy.stats import norm #for random forcing
import scipy as sp #using scipy.fft instead of numpy.fft

def oneover(x): # used for handling 1/ksqr without causing division by zero
    res=x*0
    res[x!=0]=1/x[x!=0]
    return res

# the hyperviscosity function
nufn = lambda ksqr,nu=1e-8,nuL=1e-8 : nu*ksqr**2+nuL*oneover(ksqr**4)

zeros=np.zeros
rfft2=sp.fft.rfft2
irfft2=sp.fft.irfft2

def hsymmetrize(uk): #Hermitian symmetrize a 2D array
    Nx=uk.shape[0]
    uk[Nx-1:int(Nx/2):-1,0]=uk[1:int(Nx/2),0].conj()
    return uk

def zero_nyquist(uk): #Zero out the Nyquist modes in x and y
    Nx=uk.shape[0]
    uk[int(Nx/2),:]=0
    uk[:,-1]=0
    return uk

# We define the padded system size, since padded ffts being 2^n is probably faster
Npx,Npy=1024,1024
Nx,Ny=2*np.int_(np.floor((Npx/3,Npy/3))) #actual system size with 2/3 rule
nu=2e-7   #hyperviscosity
nuL=1e-8  #hypoviscosity, needed in 2D because of the inverse cascade.
FA=1e4    #Forcing Amplitude
#Forcing Range defined in polar coordinates
frng={'k0':0.4,'dk':0.2,'thk0':np.pi/2,'dthk':np.pi}
Lx=2*np.pi/0.1    #Box size in X
Ly=2*np.pi/0.1    #Box size in Y
t0,t1=0.0,1000.0  #t range 
dtstep=0.1        #forcing is updated at every dtstep
dtout=1.0         #hdf file is written at every dtout
filename='out.h5' #name of the hdf5 file

def init_matrices(kx,ky,nu):
    ksqr=kx**2+ky**2
    Lnm=zeros(kx.shape,dtype=complex)
    Nlm=zeros(kx.shape,dtype=float)
    Lnm=-nufn(ksqr,nu)
    zero_nyquist(Lnm)
    Nlm=1.0*oneover(ksqr)*((Nx/Npx)*(Ny/Npy))
    zero_nyquist(Nlm)
    return Lnm,Nlm

def init_forcing(kx,ky,frng):
    k0=frng['k0']
    thk0=frng['thk0']
    dk=frng['dk']
    dthk=frng['dthk']
    k=np.sqrt(kx**2+ky**2)
    thk=np.arctan2(ky,kx)
    Fnm=zeros(kx.shape,dtype=complex)
    Fnm[:]=FA*((k<k0+dk) & (k>k0-dk) & (thk<thk0+dthk) & (thk>thk0-dthk))
    return Fnm

# Initializing the k-space grid
dkx=2*np.pi/Lx
dky=2*np.pi/Ly
kx,ky=np.meshgrid(np.r_[0:int(Nx/2),-int(Nx/2):0]*dkx,
                  np.r_[0:int(Ny/2+1)]*dky,indexing='ij')
ksqr=kx**2+ky**2

#Intializing the vectors and the matrices that we need
uk=zeros(kx.shape,dtype=complex)
Lnm,Nlm=init_matrices(kx,ky,nu)
Fnm=init_forcing(kx,ky,frng)
dukdt=zeros(uk.shape,dtype=uk.dtype)
datk=zeros((4,Npx,int(Npy/2+1)),dtype=complex)
rdat=zeros((2,Npx,Npy),dtype=float)

#Initializing the HFD5 file
fl=h5.File(filename,"w",libver='latest')
grp=fl.create_group("params")
grp.create_dataset("nu",data=nu)
grp.create_dataset("nuL",data=nuL)
grp.create_dataset("Lx",data=Lx)
grp.create_dataset("Ly",data=Ly)
grp=fl.create_group("fields")
ukres=grp.create_dataset("uk",(1,)+uk.shape,maxshape=(None,)+uk.shape,dtype=complex)
tres=grp.create_dataset("t",(1,),maxshape=(None,),dtype=float)
fl.swmr_mode = True
ukres[-1,]=uk
tres[-1,]=t0
fl.flush()

#The ODE RHS function for 2D Navier-Stokes 
def f(t,y):
    vk=y.view(dtype=complex).reshape(uk.shape)
    datk.fill(0)
    datk[:,np.r_[0:int(Nx/2),Npx-int(Nx/2):Npx],:int(Ny/2)+1]     \
        =np.array([1j*kx*vk,1j*ky*vk,-1j*kx*ksqr*vk, \
                   -1j*ky*ksqr*vk])
    dat=irfft2(datk)
    rdat=dat[0,]*dat[3,]-dat[1,]*dat[2,]
    resk=rfft2(rdat)
    dukdt=Lnm*vk+resk[np.r_[0:int(Nx/2),Npx-int(Nx/2):Npx],:int(Ny/2)+1]*Nlm+Fnm
    return dukdt.ravel().view(dtype=float)

#save results at each dtout
def saveres(t,y):
    ukres.resize((ukres.shape[0]+1,)+ukres.shape[1:])
    tres.resize((tres.shape[0]+1,))
    ukres[-1,]=y
    tres[-1,]=t
    fl.flush()

#Initialize the ODE Solver
r=spi.RK45(f,t0,uk.ravel().view(dtype=float),t1,rtol=1e-8,atol=1e-6,max_step=dtstep)
print(f"running {__file__}")
print(f"resolution: {Nx}x{Ny}")
print(f"parameters: nu={nu}, FA={FA}")
ct=time()
print("t=",r.t)
print(time()-ct,"secs elapsed")
t=t0
toldout=t
toldstep=t
Fnm0=np.abs(Fnm).copy()
Fnm[:]=Fnm0*(np.pi*norm.rvs(size=Fnm.shape)+1j*np.pi*norm.rvs(size=Fnm.shape))
hsymmetrize(Fnm)
zero_nyquist(Fnm)

#Main ODE solver loop
while r.status=='running' and r.t < t1:
    r.step()
    if(r.t-toldstep>=dtstep):
        Fnm[:]=Fnm0*(np.pi*norm.rvs(size=Fnm.shape)+1j*np.pi*norm.rvs(size=Fnm.shape))
        hsymmetrize(Fnm)
        zero_nyquist(Fnm)
        t+=dtstep
        toldstep=t
        print("t=",t)
        print(time()-ct,"secs elapsed")
    if(r.t-toldout>=dtout):
        dnso=r.dense_output()
        print("writing t=",t)
        saveres(t,dnso(t).view(dtype=complex).reshape(uk.shape))
        toldout=t
fl.close()
