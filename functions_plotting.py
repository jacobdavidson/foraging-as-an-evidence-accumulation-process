#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 12:47:31 2018

@author: jacob
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import functions_simulation as sim
from scipy.ndimage.filters import gaussian_filter1d as gfilter
from scipy.interpolate import UnivariateSpline




snscolors=sns.color_palette() 

def maketwoplots(tpx,xlabel,Einput,Tinput,ax0,ax1,lsize=14,ytopE=0,ytopT=0,colornum=0,smooth=0,smoothdegree=3,matchoptcolor=False):

    all_E_results = np.array(Einput)
    all_T_results = np.array(Tinput)
    # apply smoothing filter to make the results look nice
    
    if smooth>0:
        for q in range(1,all_E_results.shape[1]):
            spl = UnivariateSpline(tpx, all_E_results[:,q],k=smoothdegree)
            spl.set_smoothing_factor(smooth)
            all_E_results[:,q] = spl(tpx)
            #all_E_results[:,q]=gfilter(all_E_results[:,q],gsmooth)
        for q in range(all_T_results.shape[1]):
#            all_T_results[:,q]=gfilter(all_T_results[:,q],gsmooth)    
            spl = UnivariateSpline(tpx, all_T_results[:,q],k=smoothdegree)
            spl.set_smoothing_factor(smooth)
            all_T_results[:,q] = spl(tpx)

    # set the color the optimal line
    if matchoptcolor:
        optcolor = snscolors[colornum]
    else:
        optcolor = 'k'

    # Energy rate
    ax0.plot(tpx,all_E_results[:,0],label='$E_{opt}$',color=optcolor)
#    ax0.plot(tpx,np.zeros(len(tpx)),label='Minimum to survive', color=snscolors[3])
    ax0.plot(tpx,all_E_results[:,3],'--',label='$<E>$',color=snscolors[colornum])
    #ax0.plot(tpx,all_E_results[:,1],label='E',color=snscolors[0])
    ax0.fill_between(tpx,all_E_results[:,1]+all_E_results[:,2],all_E_results[:,1]-all_E_results[:,2], alpha=0.2, color=snscolors[colornum],label='$E$')

    ax0.set_ylabel('Energy (units of $s$)',fontsize=lsize*1.1)
    ax0.set_xlabel(xlabel,fontsize=lsize*1.1)
    ax0.set_title('Energy',fontsize=lsize*1.2)
    if ytopE>0:
        ax0.set_ylim(top=ytopE)
    ax0.set_ylim(bottom=0)
    ax0.tick_params(labelsize=lsize)
    
    # Patch residence times
    ax1.plot(tpx,all_T_results[:,0],label='$T_{opt}$',color=optcolor)
    ax1.plot(tpx,all_T_results[:,1],'--',label='$<T>$',color=snscolors[colornum])
    ax1.fill_between(tpx,all_T_results[:,1]+all_T_results[:,2],all_T_results[:,1]-all_T_results[:,2], alpha=0.2, color=snscolors[colornum],label='$T$')
    ax1.set_ylabel('PRT (units of $\\tau$)',fontsize=lsize*1.1)
    ax1.set_xlabel(xlabel,fontsize=lsize*1.1)
    ax1.set_title('Patch residence times',fontsize=lsize*1.2)
    ax1.set_ylim(bottom=0)
    if ytopT>0:
        ax1.set_ylim(top=ytopT)
    ax1.tick_params(labelsize=lsize)

    ax0.legend(fontsize=lsize)    
    ax1.legend(fontsize=lsize)        
    
def plottrace(Evalue,meanEactual,allE,allx,alleta,patchsteps,plotrange=[0,100]):
    numsteps=len(allE)
    timevalues=np.arange(0,numsteps)*sim.dt
    
    f,ax=plt.subplots(2,1,sharex=True)
    f.set_size_inches(8,4)
    ax[0].plot(timevalues,np.ones(numsteps)*Evalue,label='Optimum E',color='k')
    ax[0].plot(timevalues,np.zeros(numsteps),label='minimum to survive',color=snscolors[3])
    ax[0].plot(timevalues,meanEactual*np.ones(numsteps),'--',label='E (actual avg)',color=snscolors[0])
    ax[0].plot(timevalues,allE,label='E(t)',color=snscolors[0])
    ax[0].set_ylabel('Energy')
    ax[0].set_xlim(plotrange)
    ax[0].legend()
    
    ax[1].plot(timevalues,allx,label='x',color=snscolors[0])            
    for k in range(len(patchsteps)):
        if k==1:  # add a legend entry, but only do it once
            ax[1].axvspan((patchsteps[k][0])*sim.dt, patchsteps[k][1]*sim.dt, alpha=0.2, color='gray',label='in patch')
        else:
            ax[1].axvspan((patchsteps[k][0]-1)*sim.dt, (patchsteps[k][1]-0)*sim.dt, alpha=0.2, color='gray')
    ax[1].plot(timevalues,alleta,'k--',label='Threshold')
    ax[1].set_xlabel('time (sec)')
    ax[1].set_ylabel('x')
    ax[1].legend()
    if np.max(alleta)>0:
        ax[1].set_ylim([np.min(allx),np.max(alleta)+0.5])
    else:
        ax[1].set_ylim([np.min(alleta)-0.5,np.max(allx)+0.5])


#%% functions for plotting
def gridplots(ax,paramvalues, grid_r0values, grid_E_results, grid_T_results,
              label='',lsize=12,smooth=0,smoothdegree=3,matchoptcolor=False):
    tpx=paramvalues
    for tpnum in range(grid_r0values.shape[0]):
            
        for Ttrnum in range(grid_r0values.shape[1]):
            for Enum in np.arange(grid_r0values.shape[2]):
    #            ax0=axE[Enum,Ttrnum]
    #            ax1=axT[Enum,Ttrnum]
                ax0=ax[2-Enum,Ttrnum]
                ax1=ax[2-Enum,Ttrnum+3]
                Eres=grid_E_results[tpnum][Ttrnum][Enum]
                Tres=grid_T_results[tpnum][Ttrnum][Enum]
                if Enum==0:
                    toplim = 2
                else:
                    toplim = Eres[0][-1]*1.7
                toplim = [2,4,8][Enum]
                maketwoplots(tpx,label,Eres,Tres,ax0,ax1,lsize=lsize,ytopE=toplim,ytopT=15,colornum=tpnum*3,smooth=smooth,smoothdegree=smoothdegree,matchoptcolor=matchoptcolor)
                # formatting:
                ax0.set_title('')
                ax1.set_title('')
                if (Ttrnum>0) | (Enum>0):
                    ax0.legend().set_visible(False)
                    ax1.legend().set_visible(False)
                if Ttrnum>0:
                    ax0.set_ylabel('')
                    ax1.set_ylabel('')    
                    ax0.set_yticks([])
                    ax1.set_yticks([])                    
                if Enum>0:
                    ax0.set_xlabel('')
                    ax1.set_xlabel('')
        ax[0,1].set_title('Energy',fontsize=1.3*lsize)
        ax[0,4].set_title('Patch residence times',fontsize=1.3*lsize)                   
    
def combinedplots(paramvalues, grid_r0values, grid_E_results, grid_T_results, grid_prts, grid_traveltimes, saveplot=False, savedir='', savebasename=''):
    tpx=paramvalues
    for tpnum in range(len(tau_patch_values)):
        f,ax=plt.subplots(1,2,sharex=True)
        f.set_size_inches(15,6)
        plt.suptitle('tau_patch='+str(tau_patch_values[tpnum]))        
        for Ttrnum in range(len(Ttravelvalues)):
            for Enum in range(len(Evalues)):
    #            ax0=axE[Enum,Ttrnum]
    #            ax1=axT[Enum,Ttrnum]
                if  True or (Ttrnum==1):
                    ax0=ax[0]
                    ax1=ax[1]
                    Eres=np.array(grid_E_results[tpnum][Ttrnum][Enum])
                    Tres=np.array(grid_T_results[tpnum][Ttrnum][Enum])
                    prtres=grid_prts[tpnum][Ttrnum][Enum]
                    travelres=grid_traveltimes[tpnum][Ttrnum][Enum]
                    maketwoplots(tpx,label,Eres,Tres,ax0,ax1,lsize=8,ytopE=6,ytopT=20,colornum=Enum)
                    ax0.set_title(np.round(grid_r0values[tpnum][Ttrnum][Enum],1))
                    ax0.set_ylim(bottom=-0.5)
        if saveplot:
            plt.savefig(savedir+'combinedplots-'+savebasename+'-tau_patch='+str(tau_patch_values[tpnum])+'.png')
        plt.show()