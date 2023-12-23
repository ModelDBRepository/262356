#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:17:35 2020

@author: tiziano
"""
import matplotlib
import pylab as pl
import numpy as np
import grid_utils.plotlib as pp
import grid_utils.simlib as sl
import amp_paper_2d_main as apm

from recamp_2pop import RecAmp2PopAttractor,RecAmp2PopLearn
#from recamp_2pop import find_bump_peak_idxs

#%% TEST BUMP DETECTION ALGORITHM

sim=RecAmp2PopAttractor(apm.def_recamp_attractor_params)
sim.post_init()
sim.stimulus_bump_sigma=0.2
sim.get_stimulus_bump()

det_stim_bumps_idxs2d=np.unravel_index(sim.det_stim_bumps_idxs, (sim.n_e,)*2)
phase_idxs2d=np.unravel_index(range(sim.N_e), (sim.n_e,)*2)

pl.figure(figsize=(10,3))
pl.subplots_adjust(left=0.2,right=0.95,wspace=0.4)
pl.suptitle('Test of bump detection algorithm')

pl.subplot(131)
pl.gca().set_aspect('equal', adjustable='box')
pl.plot(phase_idxs2d[0],det_stim_bumps_idxs2d[0],'.k')
pl.xlabel('given stim x')
pl.ylabel('detected stim x')
pp.custom_axes()

pl.subplot(132)
pl.gca().set_aspect('equal', adjustable='box')
pl.plot(phase_idxs2d[1],det_stim_bumps_idxs2d[1],'.k')
pl.xlabel('given stim y')
pl.ylabel('detected stim y')
pp.custom_axes()

pl.subplot(133)
pl.gca().set_aspect('equal', adjustable='box')
pl.xlabel('detected stim x')
pl.ylabel('detected stim y')
pp.custom_axes()

H, xedges, yedges = np.histogram2d(det_stim_bumps_idxs2d[0], det_stim_bumps_idxs2d[1], bins=(range(31), range(31)))
pl.pcolormesh(H.T,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=100,cmap='viridis_r')
pp.colorbar(fixed_ticks=[1,10,100])
    

shift_idx=np.ravel_multi_index((15,15), (sim.n_e,)*2)

pl.figure(figsize=(9,3))
pl.subplot(131,aspect='equal')
poly=pp.plot_on_rhombus(sim.gp.R_T,1,0, sim.N_e,sim.gp.phases,
                              sim.stimulus_bumps[:,shift_idx],plot_axes=False,plot_rhombus=True,
                              plot_cbar=True,vmin=0)
pl.title('stimulus bump')

#%% TEST BUMP DETECTION ALGORITHM



weights_data_path=apm.batch_signal_weight_learn.get_path_by_pars((1.0,))
#%%
sim=RecAmp2PopAttractor(apm.def_recamp_attractor_params)
sim.post_init()
#sim.load_weights_from_data_path(weights_data_path)

# use homogeneous inhibitory connectivity
#sim.W[:sim.N_e,sim.N_e:]=-sim.W_tot_ei/sim.N_i
#sim.W[sim.N_e:,:sim.N_e]=sim.W_tot_ie/sim.N_e
#sim.W[sim.N_e:,sim.N_e:]=-sim.W_tot_ii/sim.N_i

   


#W2=sim.W.copy()
#W2a=W2[:sim.N_e,:sim.N_e].reshape(30,30,30,30)
#W2a=np.swapaxes(W2a,2,3)
#W2[:sim.N_e,:sim.N_e]=W2a.reshape(sim.N_e,sim.N_e)

pl.figure()
pl.subplot(121,aspect='equal')
pl.pcolormesh(sim.W)

sim.plot_recurrent_connectivity()

#%%

W2=sim.W[:sim.N_e,:sim.N_e].reshape(30,30,30,30)

W2_summed=W2.sum(axis=3).sum(axis=2)
pl.figure()
pl.subplot(121,aspect='equal')

#pl.pcolormesh(W2[10,10,:,:])
pl.pcolormesh(W2_summed.T)
pl.title(sim.W_tot_ee)
pl.subplot(122)
pl.plot(W2_summed.ravel()-sim.W_tot_ee)


#%%

sim=RecAmp2PopAttractor(apm.def_recamp_attractor_params)
sim.post_init()
#sim.load_weights_from_data_path(RecAmp2PopLearn(apm.def_recamp_learn_params).data_path)
sim.time_stimulus_on=4.
sim.time_stimulus_off=4.
sim.run_attractor_sims()



#%%

sim.get_attractor_fields()

stim_on =False

if stim_on:
  out_bumps_evo_idx2d=sim.out_bumps_stim_on_evo_idx2d
  time=np.linspace(0,sim.time_stimulus_on,sim.recdyn_num_snaps)
  field_len_evo=sim.field_len_stim_on_evo
  
else:
  out_bumps_evo_idx2d=sim.out_bumps_stim_off_evo_idx2d
  time=np.linspace(0,sim.time_stimulus_off,sim.recdyn_num_snaps)
  field_len_evo=sim.field_len_stim_off_evo


pl.figure()
pl.plot(time,field_len_evo,lw=1.5)
pl.ylabel('Field length')
pl.xlabel('Time [s]')
pp.custom_axes()

fig,axes = pl.subplots(2,5,figsize=(15,8))
pl.subplots_adjust(left=0.05,right=0.95,hspace=0.2)

  
for snap_idx in range(10):
    
    idxs2d=out_bumps_evo_idx2d[snap_idx,:,:]
  
    #ax=fig.add_subplot(gs[0 if seed_idx<5 else 1,seed_idx],aspect='equal')
    pl.subplot(axes.ravel()[snap_idx],aspect='equal')
    pl.gca().set_aspect('equal', adjustable='box')

    H, xedges, yedges = np.histogram2d(idxs2d[0,:], idxs2d[1,:], bins=(range(31), range(31)))   
    img=pl.pcolormesh(H,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=100,cmap='viridis_r')
    pl.title('time=%.2f\nfield len=%.2f'%(time[snap_idx],field_len_evo[snap_idx]))
  
#%% ## PLOT FEED STIMULUS AND OUTPUT BUMPS

sim.load_attractor_outputs()

shift_idx=np.ravel_multi_index((15,15), (sim.n_e,)*2)

pl.figure(figsize=(9,3))
pl.subplot(131,aspect='equal')
poly=pp.plot_on_rhombus(sim.gp.R_T,1,0, sim.N_e,sim.gp.phases,
                              sim.stimulus_bumps[:,shift_idx],plot_axes=False,plot_rhombus=True,
                              plot_cbar=True,vmin=0)
pl.title('stimulus bump')

pl.subplot(132,aspect='equal')
poly=pp.plot_on_rhombus(sim.gp.R_T,1,0, sim.N_e,sim.gp.phases,
                              sim.out_bumps_stim_on[0:sim.N_e,shift_idx],plot_axes=False,plot_rhombus=True,
                              plot_cbar=True,vmin=0)
pl.title('output bump with cue')

pl.subplot(133,aspect='equal')
poly=pp.plot_on_rhombus(sim.gp.R_T,1,0, sim.N_e,sim.gp.phases,
                              sim.out_bumps_stim_off[0:sim.N_e,shift_idx],plot_axes=False,plot_rhombus=True,
                              plot_cbar=True,vmin=0)
pl.title('output bump without cue')
 
#%%

pl.figure()
pl.hist(sim.out_bumps_stim_on[0:sim.N_e,shift_idx],bins=100)    

#%% LOAD ATTRACTOR LANDSCAPE DATA WITH DIFFERENT SEEDS (SAME SIGNAL WEIGHT)
    
batch= apm.batch_attractor_weight_seeds
batch.post_init()
sims=batch.sims
sims[9]=RecAmp2PopAttractor(sl.map_merge(apm.def_recamp_attractor_params,{'use_learned_recurrent_weights':False}))

for sim in sims: 
  sim.load_attractor_outputs()  
  sim.get_attractor_fields()

  
#%% LOAD ATTRACTOR LANDSCAPE DATA WITH SIGNAL WEIGHTS (SAME SEED)
    
# extracts an array from a dataframe grouping by one column
get_array_from_df= lambda  df,group_col,sel_col:  np.array(df.groupby([group_col])[sel_col].apply(lambda x: x.values.tolist()).tolist())


batch= apm.batch_attractor_weight_signal
batch.post_init()
batch.merge(['field_len_stim_on','field_len_stim_off'])

conn_batch_sig=apm.batch_signal_weight_learn
conn_batch_sig.post_init()
sort_idxs=np.argsort(conn_batch_sig.hashes)
inv_sort_idxs=np.argsort(sort_idxs)



# collect amp indexes grouped by connectivity and sort them accorting to the signal weight during learning
field_len_stim_on=get_array_from_df(batch.df,'recurrent_weights_path','field_len_stim_on')
field_len_stim_off=get_array_from_df(batch.df,'recurrent_weights_path','field_len_stim_off')

pl.figure()
pp.custom_axes()


sims=np.array(batch.sims)[inv_sort_idxs]

sims=sims[1:30:3]
  
sims[9]=RecAmp2PopAttractor(sl.map_merge(apm.def_recamp_attractor_params,{'use_learned_recurrent_weights':False}))

for sim in sims: 
  sim.load_attractor_outputs()  
  sim.get_attractor_fields()
  
pl.axhline(sims[9].field_len_stim_off,color='C1',ls='-',lw=3, label='hardwired (with cue)')
pl.axhline(sims[9].field_len_stim_on,color='C0',ls='-',lw=3,label='hardwired (no cue)')

pl.plot(apm.signal_weight_ran ,field_len_stim_on[inv_sort_idxs],'.-',label='learned (with cue)',ms=12)
pl.plot(apm.signal_weight_ran ,field_len_stim_off[inv_sort_idxs],'.-',label='learned (no cue)',ms=12)
pl.xlabel('Input tuning strength $\\beta$ during learning',fontsize=14)
pl.ylabel('Mean vector length',fontsize=14)

pl.legend(fontsize=14)

#%% PLOT INPUT/OUTPUT BUMP LOCATIONS X_in,X_out, Y_in,Y_out

stim_on=True
  
fig=pl.figure(figsize=(15,5))

gs = pl.GridSpec(2,10,hspace=0.3,wspace=0.3,bottom=0.15,left=0.04,right=0.99)

if stim_on:
  pl.suptitle('Network output with cued stimulus bump')
else:
  pl.suptitle('Network output with uniform input (after cueing)')
  
for dim_idx in range(2):
  for seed_idx in range(10):
    
    if stim_on:
      idxs2d=sims[seed_idx].out_bumps_stim_on_idx2d
    else:
      idxs2d=sims[seed_idx].out_bumps_stim_off_idx2d

  
    ax=fig.add_subplot(gs[dim_idx,seed_idx],aspect='equal')
    
    pl.plot(det_stim_bumps_idxs2d[dim_idx],idxs2d[dim_idx,:],'.',ms=4)
    pl.plot([0,30],[0,30],'-r')
    pl.xlim(0,30)
    pl.ylim(0,30)

    pl.xticks((0,30))
    pl.yticks((0,30))
    pp.custom_axes()
    
    if seed_idx<9:
      pl.title('Learned %d'%(seed_idx+1))
    else:
      pl.title('Hardwired')
      
    if seed_idx==0:
        dim='x' if dim_idx==0 else 'y'
      
        pl.xlabel('input bump %s'%dim)
        pl.ylabel('output bump %s'%dim)
    
    if seed_idx>0:
      pl.gca().axes.yaxis.set_ticklabels([])


#%% PLOT OUTPUT ATTRACTORS

stim_on=True
quiver=False

fig,axes = pl.subplots(2,5,figsize=(15,8))
pl.subplots_adjust(left=0.05,right=0.95,hspace=0.2)
if stim_on:
  pl.suptitle('Network output with cued stimulus bump')
else:
  pl.suptitle('Network output with uniform input (after cueing)')
  
for seed_idx in range(10):
    
    curr_sim=sims[seed_idx]  
    if stim_on:
      idxs2d=curr_sim.out_bumps_stim_on_idx2d
      dX,dY=curr_sim.dX_stim_on,curr_sim.dY_stim_on
      field_len=curr_sim.field_len_stim_on
      
    else:
      idxs2d=curr_sim.out_bumps_stim_off_idx2d
      dX,dY=curr_sim.dX_stim_off,curr_sim.dY_stim_off
      field_len=curr_sim.field_len_stim_off

  
    #ax=fig.add_subplot(gs[0 if seed_idx<5 else 1,seed_idx],aspect='equal')
    pl.subplot(axes.ravel()[seed_idx],aspect='equal')
    pl.gca().set_aspect('equal', adjustable='box')

    H, xedges, yedges = np.histogram2d(idxs2d[0,:], idxs2d[1,:], bins=(range(31), range(31)))
    if quiver is False:
      img=pl.pcolormesh(H,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=100,cmap='viridis_r')
    else:
      img=pl.pcolormesh(H,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=100,cmap='viridis_r')
      pl.quiver(curr_sim.Y_in+0.5,curr_sim.X_in+0.5,dY,dX,headwidth=14)
      
      
    pl.xlim(0,30)
    pl.ylim(0,30)

    pl.xticks((0,30))
    pl.yticks((0,30))
    pp.custom_axes()
    
    if seed_idx<9:
      pl.title('Learned %d\n mean vector len =%.2f'%(seed_idx+1,field_len))
    else:
      pl.title('Hardwired\n mean vector len =%.2f'%field_len)
    
    if seed_idx==5:
      
        pl.xlabel('output bump x')
        pl.ylabel('output bump y')

   
cbar_ax = fig.add_axes([0.96, 0.14, 0.01, 0.2])
fig.colorbar(img, cax=cbar_ax)


#%% ### TO DELETE: COPY FILES FROM THE SERVER 
import os
  
for signal_weight in apm.signal_weight_ran:
    
    sim_conn=RecAmp2PopLearn(sl.map_merge(apm.def_recamp_learn_params,{'signal_weight':signal_weight}))
    
    if not os.path.exists(sim_conn.data_path):
        remote_path=sim_conn.data_path.replace('/home/tiziano/grid_amp','/groups/kempter/dalbis')
        cmd='scp dalbis@gate.biologie.hu-berlin.de:%s %s'%(remote_path,sim_conn.data_path)
        print(cmd)
        ret=os.system(cmd)
        print(ret)