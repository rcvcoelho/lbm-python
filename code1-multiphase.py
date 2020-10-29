""" 
Author: Rodrigo Coelho
Shan-Chen model (multiphase)
Segregation of two phases

Copywrite notice: All rights reserved. Do not share this code without permission.
"""
import numpy as np
import matplotlib.pyplot as plt

tmax = 10000
tau = 0.8           # visc = (tau - 0.5)/3
L = 50              #system size
Q = 9               # number of velocity vectors
f=np.zeros((L,L, Q))
Ux=np.zeros((L,L))
Uy=np.zeros((L,L))
Dens=np.ones((L,L))
Psi=np.ones((L,L))

#weights and velocityvectors
w = [4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 
             1.0/36.0, 1.0/36.0, 1.0/36.0]
ex = [0, 1, 0, -1, 0, 1, -1, -1, 1]
ey = [0, 0, 1, 0, -1, 1, 1, -1, -1]

#equilibrium distribution function 
def feq(k, dens, ux, uy):
    return w[k]*dens*(1.0+3.0*(ex[k]*ux+ey[k]*uy) 
            + 4.5*(ex[k]*ux+ey[k]*uy)**2 - 1.5*(ux*ux+uy*uy))

#initial conditions
def init():
    for j in np.arange(L):
        for i in np.arange(L):
            Ux[i][j] = 0.0
            Uy[i][j] = 0.0 
            Dens[i][j] = 0.99 + 0.02*np.random.random_sample()          
            for k in np.arange(Q):
                f[i][j][k] = feq(k, Dens[i][j], Ux[i][j], Uy[i][j])  
                
#calculation of density and velocity              
def macro():
    for j in np.arange(L):
        for i in np.arange(L):          
            dens = 0.0
            ux = 0.0
            uy = 0.0
            for k1 in np.arange(Q):
                dens = dens + f[i][j][k1]
            for k2 in np.arange(Q):
                ux = ux + f[i][j][k2]*ex[k2]
            for k3 in np.arange(Q):
                uy = uy + f[i][j][k3]*ey[k3]
            Dens[i][j] = dens
            Ux[i][j] = ux/dens
            Uy[i][j] = uy/dens                    

#multiphase model 
def ShanChen():
    for j in np.arange(L):
        for i in np.arange(L):    
            G = -5.5            #temperature related parameter     
            forceX = 0.0
            forceY = 0.0
            for k in np.arange(Q):
                inext = (i + ex[k]) % L
                jnext = (j + ey[k]) % L  
                psinext = (1.0 - np.exp(-Dens[inext][jnext]))
                forceX = forceX + psinext*ex[k]*w[k]
                forceY = forceY + psinext*ey[k]*w[k]
            psi = 1.0 - np.exp(-Dens[i][j])
            forceX = -forceX*G*psi
            forceY = -forceY*G*psi            
            Ux[i][j] = Ux[i][j] + forceX*tau/Dens[i][j]
            Uy[i][j] = Uy[i][j] + forceY*tau/Dens[i][j] 

#Boltzmann equation: collision + streamming
def collision():
    for j in np.arange(L):
        for i in np.arange(L):                   
            for k in np.arange(Q):               
                eq = feq(k, Dens[i][j], Ux[i][j], Uy[i][j])
                f[i][j][k] = f[i][j][k] + (eq - f[i][j][k])/tau

def stream():    
    fbuffer = np.copy(f)    
    for j in np.arange(L):
        for i in np.arange(L):  
            for k in np.arange(Q):
                iprev = (i - ex[k]) % L
                jprev = (j - ey[k]) % L                    
                f[i][j][k] = fbuffer[iprev][jprev][k]            
        
##########################################
#main
init()

for t in np.arange(tmax):
    collision()
    stream()    
    macro()
    ShanChen()
    if((t%10)==0):
        plt.imshow(Dens, interpolation="bilinear", cmap='jet')
        #plt.savefig("mp-%d.png"%t, dpi=200)
        plt.colorbar()
        plt.pause(0.05)
        plt.clf()

