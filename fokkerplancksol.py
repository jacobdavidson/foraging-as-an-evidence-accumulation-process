#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 16:23:03 2017
Updated Aug 4, 2018

@author: jdavidson
"""
import numpy as np

class FPsolution:

# initialize data structure
# note that n *must* be an odd number 
    def __init__(self,n=51):
        
        # define mesh
        ne=n-1 #number of elements
        dx=2/ne # assumes boundaries at +1 and -1
        mesh=np.cumsum(np.ones(n))*dx-1-dx
        mid=int((n-1)/2)  
        mesh[mid]=0
    
        # define element matrices
        # these use linear shape functions
        mmat=dx/6*np.array([[2,1],[1,2]])  # mass matrix
        d1mat=1/2*np.array([[1,1],[-1,-1]]) # 1st derivative matrix (i.e. it operates to calculate a derivative)
        d2mat=1/dx*np.array([[-1,1],[1,-1]]) # 2nd derivative matrix (i.e. it operates to calculate a 2nd derivative)
    
        # assemble global matrices by summing over element matrices
        Mij=np.zeros([n,n])
        Bij=np.zeros([n,n])
        Aij=np.zeros([n,n])
        
        for i in range(ne):
            Mij[i:i+2,i:i+2]=Mij[i:i+2,i:i+2]+mmat
            Bij[i:i+2,i:i+2]=Bij[i:i+2,i:i+2]+d1mat
            Aij[i:i+2,i:i+2]=Aij[i:i+2,i:i+2]+d2mat
    
        # this is needed for the boundary conditions (comes from integration by parts)
        
        # LOWER BOUNDARY        
        # absorbing lower boundary
#        Aij[0,0] += -1/dx
#        Aij[0,1] += 1/dx
        # Reflecting lower boundary
        Bij[0,0] += 1    
        
        # UPPER BOUNDARY
        # Absorbing upper boundary        
        Aij[-1,-2] += -1/dx
        Aij[-1,-1] += 1/dx
        
        # reflecting upper boundary 
#        Bij[-1,-1] +=1 
        
        # inverse of mass matrix is needed:
        MijInverse=np.linalg.inv(Mij)
        
        # matrices for integrating over all of the space
        IntA=dx*np.concatenate(([1/2], np.ones(n-2), [1/2]))
    
        # save matrices to the data structure
        self.n=n
        self.dx=dx
        self.mid=mid
        self.mesh=mesh
        self.Mij=Mij
        self.MijInverse=MijInverse
        self.Bij=Bij
        self.Aij=Aij
        self.IntA=IntA

    def InterpolateG(self,Gnew,x):
        # returns the value of G, evaluated at x via interpolation
        # this could probably be rewritten, but it works fine        
        if x<-1 or x>1:
            return 0
        else:
            allind=np.array(range(len(self.mesh)))
            first=max(allind[self.mesh<=x])
            last=min(allind[x<=self.mesh])
            return (Gnew[first] + (x-self.mesh[first])/self.dx*(Gnew[last]-Gnew[first]) )


    def getsolution(self,rewardlist,etalist,alpha,sigma,dt,threshold=0.01):
        # the solution will exit either when the remaining probability is <threshold, or numtimesteps is reached
        
        # INPUT:
            # rewardlist:  length of numtimesteps
            # etalist:
                # [eta[0],eta[1],..., eta[numsteps] ].  Input a vector of all the same number if its constant
    
        # OUTPUT
        # allH:  the remaining probability, as a function of time
        # allG:   n by q matrix, where n=number of points in mesh, and q=number of time points, containing the solution for probability density as a function of time

        # SET PARAMETERS AND INITIALIZE THINGS
        # parse the parameters and other things so they are easier to use
        numtimesteps=len(rewardlist)
        dx=self.dx
        y0=0
        # If the initial dist. width is too small, it will give numerical error. So, set a minimum 
        y0width=2.01*dx #the 2.01 multiplier is for numerical reasons -  otherwise there is an inequality below that leads to an error
        
        alpha_y = alpha/etalist
        sigma_y = np.abs(sigma/etalist)
        rewardlist_y = rewardlist/etalist

        # calculate matrix for Time evolution of probability density
        T_Aij = dt*np.matmul(self.MijInverse,self.Aij)        
        T_Bij = dt*np.matmul(self.MijInverse,self.Bij)
    
        # initial distribution:  uniform dist. of width 'y0width', centered at 'y0'
        G0=np.zeros(len(self.mesh))  # initial probability density
        G0[(np.abs(self.mesh-y0)<=y0width/2)]=1/y0width  # NOTE:  this rounds to the nearest number of elements
        
        #  initial amounts of prob density on each side, and total
        # notation:  H for integral of prob density G, over a part or all of distribution
        H0=np.dot(self.IntA,G0)
        # make sure the initial distribution is normalized, then set H_total=1
        G0=G0/H0
        H0=1  # this means that all the probability starts inside the bounds, i.e. no decisions are made before t=0
    
        Gcurr=G0  #initialize 
    
        allG=np.zeros([numtimesteps,len(self.mesh)])
        allH=np.zeros(numtimesteps) # remaining probability
        allG[0]=Gcurr
        allH[0]=H0
    
        # STEP THROUGH IN TIME
        for step in range (0,numtimesteps):
            Tij =  sigma_y[step]**2/2 * T_Aij - alpha_y[step]* T_Bij
                            
            Gnew=Gcurr+np.matmul(Tij,Gcurr)
            
            # apply food reward if there is one
            if not (rewardlist_y[step]==0):
                shiftamount=-rewardlist_y[step]*dt/dx
                    
                # now update the new values of G, with the shift
                temp=0*Gnew.copy()
                for ind in range(len(Gnew)):
                    temp[ind]=self.InterpolateG(Gnew,self.mesh[ind]-dx*shiftamount)
    
                Gnew=temp

            Gnew[-1]=0 # helps numerically to enforce boundary conditions
   
                   
            # probability remaining
            probremaining=np.dot(self.IntA,Gnew)            
            # renormalize if its >1
            if probremaining>1:
                Gnew[Gnew<0]=0
                Gnew=Gnew/probremaining
                probremaining=1
            #end if there is no probability left
            if probremaining<=threshold:
                print("reached threshold")
                break
            # update, and keep going
            Gcurr=Gnew
            
            # save
            allG[step]=Gnew
            allH[step]=probremaining
        
        return allH, allG
