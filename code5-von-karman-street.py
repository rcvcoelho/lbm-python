""" 
Author: Rodrigo Coelho
Flow through a cylinder
"""

import numpy as np
import matplotlib.pyplot as plt

#parameters
tmax = 200000        #maximum number of iterations
visc = 0.01          #kinematic viscosity
Lx = 150            #system size
Ly = 75
uxBC = 0.054        #inlet velocity

#################
tau = 3.0*visc + 0.5        #relaxation time
Q = 9                       #number of velocity vectors
f=np.zeros((Ly,Lx, Q))
Ux=np.zeros((Ly,Lx))
Uy=np.zeros((Ly,Lx))
Dens=np.ones((Ly,Lx))
Obst=np.ones((Ly,Lx))
Y, X = np.mgrid[0:Ly, 0:Lx]

#weights and velocityvectors
w = [4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 
             1.0/36.0, 1.0/36.0, 1.0/36.0]
ex = [0, 1, 0, -1, 0, 1, -1, -1, 1]
ey = [0, 0, 1, 0, -1, 1, 1, -1, -1]
kinv = [0, 3, 4, 1, 2, 7, 8, 5, 6]   #bounce back BC

#equilibrium distribution function 
def feq(k, dens, ux, uy):
    return w[k]*dens*(1.0+3.0*(ex[k]*ux+ey[k]*uy) 
            + 4.5*(ex[k]*ux+ey[k]*uy)**2 - 1.5*(ux*ux+uy*uy))

#initial conditions
def init():
    for j in np.arange(Ly):
        for i in np.arange(Lx):
            Ux[j][i] = 0.0
            Uy[j][i] = 0.0 
            Dens[j][i] = 1.0              
            #obstacles
            if( (i-Lx/4)**2 + (j-Ly/2)**2 <= (Ly/8)**2) :                   
                Obst[j][i] = 1
            else:
                Obst[j][i] = 0            
            for k in np.arange(Q):
                f[j][i][k] = feq(k, Dens[j][i], Ux[j][i], Uy[j][i])                 
                
#calculation of density and velocity              
def macro():
    for j in np.arange(Ly):
        for i in np.arange(Lx):          
            dens = 0.0
            ux = 0.0
            uy = 0.0
            for k1 in np.arange(Q):
                dens = dens + f[j][i][k1]
            for k2 in np.arange(Q):
                ux = ux + f[j][i][k2]*ex[k2]
            for k3 in np.arange(Q):
                uy = uy + f[j][i][k3]*ey[k3]
            Dens[j][i] = dens
            Ux[j][i] = ux/dens
            Uy[j][i] = uy/dens                    

#Boltzmann equation: collision + streamming
def collision():
    for j in np.arange(Ly):
        for i in np.arange(Lx):                   
            for k in np.arange(Q):               
                eq = feq(k, Dens[j][i], Ux[j][i], Uy[j][i])
                f[j][i][k] = f[j][i][k] + (eq - f[j][i][k])/tau

def stream():    
    fbuffer = np.copy(f)    
    for j in np.arange(Ly):
        for i in np.arange(Lx):  
            for k in np.arange(Q):
                iprev = (i - ex[k]) % Lx
                jprev = (j - ey[k]) % Ly   
                if(Obst[j][i] == 0):                 
                    f[j][i][k] = fbuffer[jprev][iprev][k]  
                else:     #bounce back            
                    f[j][i][k] = fbuffer[j][i][kinv[k]] 
                    
def velocityBC():   
    i=0     #left boundary
    ingb = 1
    for j in np.arange(Ly):
        Ux[j][i] = uxBC
        Dens[j][i] = 1.0
        Uy[j][i] = 0.0
        for k in np.arange(Q):
            fneq = f[j][ingb][k] - feq(k, Dens[j][ingb], Ux[j][ingb], Uy[j][ingb])
            f[j][i][k] = feq(k, Dens[j][i], Ux[j][i], Uy[j][i]) + fneq

    i=Lx-1     #left boundary
    ingb = Lx-2
    for j in np.arange(Ly):
        Ux[j][i] = uxBC 
        Dens[j][i] = 1.0
        Uy[j][i] = 0.0
        for k in np.arange(Q):
            fneq = f[j][ingb][k] - feq(k, Dens[j][ingb], Ux[j][ingb], Uy[j][ingb])
            f[j][i][k] = feq(k, Dens[j][i], Ux[j][i], Uy[j][i]) + fneq
                
        
##########################################
#main
init()

for t in np.arange(tmax):
    collision()
    stream()    
    macro()
    velocityBC()
    if((t%100)==0):  #plot the velocity field
        print("time = %d"%t)
        field = Ux
        field=np.ma.masked_where( Obst==1, field)
        im = plt.imshow(field, interpolation="bilinear", cmap='jet')
        im2 = plt.streamplot(X, Y, Ux, Uy, density=1.5, color = "w", linewidth = 1.5)
        #plt.savefig("mp-%d.png"%t, dpi=200)
        plt.colorbar(im)        
        plt.tight_layout()
        plt.savefig("cylinder-%d.pdf"%t)
        plt.pause(0.05)
        plt.clf()
