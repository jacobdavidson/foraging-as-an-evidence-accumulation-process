#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model simulation functions
"""

import numpy as np
import scipy.optimize
import scipy.stats


#setting these to 1, without loss of generality, because now the simulations are in these units:
s=1
tau=1
totaltime=1000
dt=0.01
start_for_mean=200
tau_E = 50  # seconds

# Equation to numerically solve in order to determine the optimal energy possible
def optE_eqn(E,rho0,s,tau_patch,Ttravel):
    return (rho0-E-s)/(np.log(rho0/(E+s))+Ttravel/tau_patch) - E - s

def optE_eqn_sigma_rho0(E,rho0,s,tau_patch,Ttravel):
    return (rho0-E-s)/(np.log(rho0/(E+s))+Ttravel/tau_patch) - E - s    

def getEopt(rho0,tau_patch,Ttravel):
    return scipy.optimize.newton(lambda E: optE_eqn(E,rho0,s,tau_patch,Ttravel),s*1.5 )

def getrho0_Eopt(E,tau_patch,Ttravel):
    return scipy.optimize.newton(lambda rho0: optE_eqn(E,rho0,s,tau_patch,Ttravel),1.2*(E+s) ) 

def alpha_opt(E,rho0,tau_patch):  # this optimizes for uncertainty in patch size
    return (rho0-E-s)/(np.log(rho0/(E+s)))

def eta_opt(E,rho0,tau_patch,alpha):
    return tau_patch*( alpha*np.log(rho0/(E+s)) - rho0 + E + s )

def generate_patch_rewards(chunksize,rho0,taupatch,numsteps,dt=dt):  # this is used for FP simulation.  Could adapt to use it below also instead
    if chunksize==0:            
        allreward=rho0*np.exp(-np.arange(numsteps)*dt/taupatch)
        return allreward, allreward  # in this case, reward and density are the same
    else:  # then chunksize>0, so use discrete chunks of reward      
        allrho = np.zeros(numsteps)
        allreward = np.zeros(numsteps)    
        rewardrate = rho0
        for step in range(numsteps):
            k = scipy.stats.poisson.rvs(rewardrate*dt/chunksize)  # number of chunks of reward to get
            reward = k*chunksize/dt
            rewardrate=np.max([rewardrate-k*chunksize/taupatch,0])  # ensure that this doesn't go negative
            allreward[step]=reward 
            allrho[step]=rewardrate
        return allreward, allrho
    

def modelsimulation(totaltime, dt, # time in seconds
                    tau_E, Ttravel, rho0, tau_patch,
                    start_for_mean=200,
                    beta=0, sigma=0, sigma_rho0=0, sigma_taupatch=0, sigma_Ttravel=0,q=0,
                    alphachoice=1, 
                    uchoice=0,
                    E0mult=1,
                    timetrace=False):
    
    
    # solve for optimal energy and optimal patch time
    Eopt=getEopt(rho0,tau_patch,Ttravel)
    Topt=tau_patch*np.log(rho0/(Eopt+s))

    numsteps=np.round(totaltime/dt).astype(int)
    meanstep=np.round(start_for_mean/dt).astype(int)


    if Topt<=0:
        return [np.nan,np.nan,np.nan,np.nan], [np.nan,np.nan,np.nan], [], []
   
    #%%
    
    # variables for storing things, and their initial conditions
    allE = np.zeros(numsteps)
    allx = np.zeros(numsteps)
    allEactual=np.zeros(numsteps)
    patchsteps=[]
    travelsteps=[]
    rho0values=[]
    taupatchvalues=[]
    numfoodrewards=[]
    current_numfoodrewards = 0
    
    
    # initial conditions
    allE[0] = E0mult*Eopt  # minimum in order to survive is s

    # initialize for simulation
    inpatch=False
    rho0currentpatch = rho0 + sigma_rho0*np.random.randn()
#    Ttravel_current = np.max([ dt,Ttravel + sigma_Ttravel*np.random.randn()])  # minimum of 1 timestep to travel between patches
    Ttravel_current = sigma_Ttravel*np.random.exponential(Ttravel) + (1-sigma_Ttravel)*Ttravel
    taupatch_current = np.max([tau_patch + sigma_taupatch*np.random.randn(),0.1])
    patch_startstep = 0
    travel_startstep=0
    allx[0] = 0
    
    # using the value of the 'trade-off' parameter, set the alphafn
    
    alpha_current = rho0*alphachoice
    
    if uchoice==0:
        def ufn(beta,E):
            return 1
    elif uchoice==1:  # exponential
        def ufn(E,beta,A=0):
            return (1-A)*np.exp(-beta*E) + A        
    elif uchoice==2:  #linear, with threshold of 0.5
        def ufn(E,beta,A=0.65):
            return (1-beta*E) * np.heaviside(1-beta*E-A,0) + A*np.heaviside(A-1+beta*E,1)        


    # loop through time
    rewardrate = rho0currentpatch
    for step in range(1,numsteps):
        # check whether should leave the current patch, or if just traveled to a new one
        if inpatch:
            if q==0:            
                rewardrate=rho0currentpatch*np.exp(-(step-patch_startstep)*dt/taupatch_current)
                reward=rewardrate
            else:  # then q>0, so use discrete chunks of reward                
                k = scipy.stats.poisson.rvs(rewardrate*dt/q)  # number of chunks of reward to get
                reward = k*q/dt
                rewardrate=np.max([rewardrate-k*q/tau_patch,0])  # ensure that this doesn't go negative
                current_numfoodrewards += k
      
            uval = ufn(beta,allE[step-1])
            eta_current= eta_opt(allE[step-1],rho0,tau_patch,alpha_current)  # note that this uses alpha, NOT u
            
            allx[step]=allx[step-1]+dt/tau*(-reward * uval**(-np.sign(eta_current)) + alpha_current*uval) + sigma*np.sqrt(dt)*np.random.randn()/tau
            allE[step]=allE[step-1]+dt/tau_E*(reward-s-allE[step-1]) 
            allEactual[step] = reward -s

            if eta_current>=0:
                reachedthreshold=(allx[step]>eta_current)# & (allx[step-1]<eta_current)
            else:
                reachedthreshold=(allx[step]<eta_current)# & (allx[step-1]>eta_current)  # leave this out, because on the occasion that both the threshold and x change in the same time step, this can cause it to mess up!  This is only needed if the threshold is changing sign during the simulation
            if reachedthreshold:
                inpatch=False
                travel_startstep=step+1  # will start travel at the "next step"
#                Ttravel_current = np.max([dt,Ttravel + sigma_Ttravel*np.random.randn()]) 
                Ttravel_current = sigma_Ttravel*np.random.exponential(Ttravel) + (1-sigma_Ttravel)*Ttravel                
                patchsteps.append([patch_startstep,step])
                rho0values.append(rho0currentpatch)
                taupatchvalues.append(taupatch_current)
                numfoodrewards.append(current_numfoodrewards)
                current_numfoodrewards = 0
        else:
            # update E and x variables
            allx[step]=0
            allE[step]=allE[step-1]+dt/tau_E*(-s-allE[step-1])
            allEactual[step]=-s
            # check if traveled long enough to reach a new patch
            if (step-travel_startstep)*dt>=Ttravel_current: # start being in a new patch
                inpatch=True
                patch_startstep=step+1  # because will start getting reward at the next step
                allx[step]=0
                travelsteps.append([travel_startstep,step])
                rho0currentpatch = np.max([rho0 + sigma_rho0*np.random.randn(),0])
                taupatch_current = np.max([tau_patch + sigma_taupatch*np.random.randn(),0.1])
                rewardrate=rho0currentpatch                
     
    #%%
    if len(patchsteps)<10:
        print('ONLY '+str(len(patchsteps))+' PATCHES:  returning null','. sigma=',sigma)
        if timetrace:
            return np.nan, np.nan, np.nan, np.nan, np.nan
#        else:
#            return [np.nan,np.nan,np.nan,np.nan], [np.nan,np.nan,np.nan], [], []
    
    # calculate patch residence times, and actual dist. of travel times
    def stepstotime(tl):
        temp=np.array(tl)
        return (temp[:,1]-temp[:,0])*dt
    
    patchsteps=np.array(patchsteps)
    travelsteps=np.array(travelsteps)
    rho0values=np.array(rho0values)
    taupatchvalues=np.array(taupatchvalues)
    # keep a 'full cycle' for average: start with a patch, end with travel
    steptoavg=[patchsteps[:,0][patchsteps[:,0]>meanstep][0],travelsteps[-1][1]]
    
    patchkeep=(patchsteps[:,0]>=steptoavg[0]) & (patchsteps[:,1]<steptoavg[1]) 
    prts=stepstotime(patchsteps[patchkeep,:])
    traveltimes=stepstotime(travelsteps[(travelsteps[:,0]>steptoavg[0]) & (travelsteps[:,1]<=steptoavg[1]) ,:])
    rho0values=rho0values[patchkeep]
    taupatchvalues=taupatchvalues[patchkeep]
    
    meanEactual=np.mean(allEactual[steptoavg[0]:steptoavg[1]])  # the std dev of this doesn't make sense, so just calculate the mean
    meanE = np.mean(allE[steptoavg[0]:steptoavg[1]])
    stdE = np.std(allE[steptoavg[0]:steptoavg[1]])
    
    meanT = np.mean(prts)
    stdT = np.std(prts)
    
    if timetrace:
        return allE, allEactual, allx, patchsteps, travelsteps, numfoodrewards
    else:
        return [Eopt,meanE,stdE,meanEactual], [Topt,meanT,stdT], prts, traveltimes, rho0values, taupatchvalues, numfoodrewards



def gridsimulationruns(tau_patch_values,Ttravelvalues,Evalues,wp,basenoise,paramvalues,q_extra=0,sigma_extra=0,alphachoice=1):  # wp is 'whichparam', a vector which chooses params
    #    [  0  ,    1     ,       2      ,       3      ,   4 , 5]
    #wp: [sigma, sigma_rho0, sigma_taupatch, sigma_Ttravel, beta, q] 

    grid_rho0values=np.zeros((len(tau_patch_values),len(Ttravelvalues),len(Evalues)))
    grid_E_results = [[[[] for k in range(len(Evalues))] for j in range(len(Ttravelvalues))] for i in range(len(tau_patch_values))]
    grid_T_results = [[[[] for k in range(len(Evalues))] for j in range(len(Ttravelvalues))] for i in range(len(tau_patch_values))]
    grid_prts=[[[[] for k in range(len(Evalues))] for j in range(len(Ttravelvalues))] for i in range(len(tau_patch_values))]
    grid_traveltimes=[[[[] for k in range(len(Evalues))] for j in range(len(Ttravelvalues))] for i in range(len(tau_patch_values))]
    
    for tpnum in range(len(tau_patch_values)):
        for Ttrnum in range(len(Ttravelvalues)):
            
            for Enum in range(len(Evalues)):
                print(tpnum,Ttrnum,Enum)
                tau_patch = tau_patch_values[tpnum]
                Ttravel = Ttravelvalues[Ttrnum] # time to travel between patches
                rho0=getrho0_Eopt(Evalues[Enum],tau_patch,Ttravel)
                grid_rho0values[tpnum,Ttrnum,Enum]=rho0
                
                for p in paramvalues:
#                    print(p)
                    # perform simulation with set parameters, and save values
                    [Eopt,meanE,stdE,meanEactual], [Topt,meanT,stdT], prts, traveltimes, _, _, _ = modelsimulation(totaltime=totaltime, dt=dt, 
                        tau_E=tau_E, Ttravel=Ttravel, rho0=rho0, tau_patch=tau_patch,
                        sigma=(basenoise+wp[0]*p+sigma_extra)*rho0, sigma_rho0=(basenoise+wp[1]*p)*rho0, sigma_taupatch=(basenoise+wp[2]*p)*tau_patch, 
                        sigma_Ttravel=(basenoise+wp[3]*p)*Ttravel, beta=wp[4]*p, q=wp[5]*p+q_extra,
                        start_for_mean=start_for_mean,alphachoice=alphachoice)
                    
                    grid_E_results[tpnum][Ttrnum][Enum].append([Eopt,meanE,stdE,meanEactual])
                    grid_T_results[tpnum][Ttrnum][Enum].append([Topt,meanT,stdT])
                    grid_prts[tpnum][Ttrnum][Enum].append(prts)
                    grid_traveltimes[tpnum][Ttrnum][Enum].append(traveltimes)

    return grid_rho0values, grid_E_results, grid_T_results, grid_prts, grid_traveltimes