#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:02:23 2019

@author: dalbis
"""
import pylab as pl
import numpy as np
import grid_utils.plotlib as pp
import grid_utils.gridlib as gl
import grid_utils.simlib as sl
from recamp_2pop import RecAmp2PopSteady,RecAmp2PopLearn,RecAmp2PopAttractor
import amp_paper_2d_main as apm

#%%
#% ### STAIGHT TRAJECTORY WITHOUT THETA MODULATION

sim=RecAmp2PopSteady(apm.def_recamp_steady_params)
sim.post_init()
sim.compute_and_save_steady_output()
sim.use_speed_input=False
sim.load_steady_outputs()
sim.load_weights_from_data_path(sim.recurrent_weights_path)

walk_time=8.
num_snaps=int(walk_time*2000)
init_p=[0.2,-1.]
init_theta=np.pi/2
theta_sigma=0.
track_cell_idx=3
sim.tau=0.01
theta_freq=8
sim.dt=0.0005

sim.run_recurrent_dynamics_with_walk(  walk_time,
                                       num_snaps,
                                       theta_sigma,
                                       init_p=init_p,
                                       init_theta=init_theta,
                                       interpolate_inputs=True,
                                       track_cell_evo=True,
                                       track_cell_idx=track_cell_idx,
                                       use_theta_modulation=False,
                                      )

no_theta_hh=sim.cell_hh_evo
no_theta_rec_input=sim.cell_rec_input_evo
no_theta_total=sim.cell_rr_evo
no_theta_rec_input_e=sim.cell_rec_input_from_e_evo
no_theta_rec_input_i=sim.cell_rec_input_from_i_evo


#%% ### STAIGHT TRAJECTORY WITH THETA MODULATION

sim=RecAmp2PopSteady(apm.def_recamp_steady_params)
sim.post_init()
sim.compute_and_save_steady_output()
sim.use_speed_input=False
sim.load_steady_outputs()
sim.load_weights_from_data_path(sim.recurrent_weights_path)

walk_time=8.
num_snaps=int(walk_time*2000)
init_p=[0.2,-1.]
init_theta=np.pi/2
theta_sigma=0.
track_cell_idx=3
sim.tau=0.01
theta_freq=8
sim.dt=0.0005

sim.run_recurrent_dynamics_with_walk(  walk_time,
                                       num_snaps,
                                       theta_sigma,
                                       init_p=init_p,
                                       init_theta=init_theta,
                                       interpolate_inputs=True,
                                       track_cell_evo=True,
                                       track_cell_idx=track_cell_idx,
                                       force_walk=False,
                                       use_theta_modulation=True,
                                       theta_freq=theta_freq,
                                       position_dt=sim.dt)

theta_hh=sim.cell_hh_evo
theta_rec_input=sim.cell_rec_input_evo
theta_total=sim.cell_rr_evo
theta_rec_input_e=sim.cell_rec_input_from_e_evo
theta_rec_input_i=sim.cell_rec_input_from_i_evo


#%% ### PLOT TEMPORAL INPUT/OUT WITH AND WITHOUT THETA

input_color='m'
rec_color='gray'
inhib_color='dodgerblue'
time_out_color='limegreen'
lw=1.5


pl.figure(figsize=(5,3))
pl.subplots_adjust(left=0.15,right=0.95,hspace=0.7,wspace=0.,bottom=0.2)
time=np.arange(num_snaps)*sim.delta_snap*sim.dt

pl.subplot(211)
pl.plot(time,no_theta_hh,color=input_color,lw=lw,label='feed-forward')
pl.plot(time,no_theta_rec_input_e,color=rec_color,lw=lw,label='recurrent')
pl.plot(time,-no_theta_rec_input_i,color=inhib_color,lw=lw,label='recurrent')
pl.plot(time,no_theta_total,color=time_out_color,lw=lw,label='output rate')
pp.custom_axes()
pl.xlabel('Time [s]')
pl.ylabel('Firing rate [1/s]')
pl.ylim([-5,50])
pl.xlim([-0.1,8])
pl.yticks([0,25,50])

pl.subplot(212)
pl.plot(time,theta_rec_input_e,color=rec_color,lw=lw,label='recurrent')
pl.plot(time,-theta_rec_input_i,color=inhib_color,lw=lw,label='recurrent')
pl.plot(time,theta_total,color=time_out_color,lw=lw,label='output rate')
pl.plot(time,theta_hh,color=input_color,lw=lw,label='feed-forward')

pl.ylim([-5,50])
pl.xlim([-0.1,8])
pl.gca().set_frame_on(False)

pl.gca().axes.get_yaxis().set_visible(False)
fname = 'fig_teporal_raw' 
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])



#%% #### RUN WITH RANDOM WALK ON THE ENTIRE ARENA WITH THETA MODULATION
# We could run several simulations with theta modulation and increasing network filtering time constants
# to show in which range of time constant amplification works

sim=RecAmp2PopSteady(sl.map_merge(apm.def_recamp_steady_params))
sim.post_init()
sim.use_speed_input=False
sim.load_weights_from_data_path(sim.recurrent_weights_path)
sim.load_steady_outputs()

walk_time=800
num_snaps=200
theta_sigma=0.7
track_cell_idx=0
sim.tau=0.01
theta_freq=8

sim.run_recurrent_dynamics_with_walk(  walk_time,
                                       num_snaps,
                                       theta_sigma,
                                       interpolate_inputs=True,
                                       force_walk=True,
                                       use_theta_modulation=True,
                                       theta_freq=theta_freq,
                                       sweep=True)



# compute excitatory tuning indexes

time_tuning_out=gl.comp_grid_tuning_index(sim.L,sim.nx,sim.r_e_walk_map.T)


#%% #### PLOT GRID TUNING INDEXES

input_color='m'
rec_color='gray'
inhib_color='dodgerblue'
time_out_color='limegreen'
lw=1.5


pl.figure(figsize=(3,2.3))
pl.subplots_adjust(left=0.3,bottom=0.26,right=0.95)

bins = pl.histogram_bin_edges(time_tuning_out,bins=50,range=[0,1])
#logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

sim.load_steady_outputs()
pl.hist(sim.grid_tuning_in,bins=bins,color=input_color,histtype='stepfilled',weights=np.ones_like(sim.grid_tuning_in)/float(len(sim.grid_tuning_in)),alpha=1)
pl.hist(sim.grid_tuning_out,bins=bins,color='k',histtype='stepfilled',weights=np.ones_like(sim.grid_tuning_out)/float(len(sim.grid_tuning_out)),alpha=1)
pl.hist(time_tuning_out,bins=bins,color=time_out_color,histtype='stepfilled',weights=np.ones_like(time_tuning_out)/float(len(time_tuning_out)),alpha=1)

pl.hist(sim.grid_tuning_out,bins=bins,color='k',histtype='step',weights=np.ones_like(sim.grid_tuning_out)/float(len(sim.grid_tuning_out)),alpha=1)

pp.custom_axes()
pl.ylim(0,0.3)  
pl.xlim(0,0.7)
pl.xlabel('Grid tuning index')
pl.ylabel('Fraction of cells')


fname = 'fig_temporal_tuning_hists'

pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])

    
#%% TEST ATTRACTOR DYNAMICS: STAIGHT TRAJECTORY AND SWITCH OFF FEEDFORWARD TUNING

walk_time=16. 
num_snaps=int(walk_time*50)
init_p=[0.0,-1.]
init_theta=np.pi/2
theta_sigma=0.
phases=gl.get_phases(30,1.,0.)
track_cell_idx=gl.get_pos_idx([0.,0.],phases)

theta_freq=10
switch_off_time=4.1
switch_on_time=10.9


sims=[]
for inputs_seed in range(10):
  
  sim_conn=RecAmp2PopLearn(sl.map_merge(apm.def_recamp_learn_params,{'inputs_seed':inputs_seed}))
  sim=RecAmp2PopSteady(sl.map_merge(apm.def_recamp_steady_params,{
                                                                    'inputs_seed':inputs_seed,
                                                                    'recurrent_weights_path':sim_conn.data_path,
                                                                    'use_speed_input':False
                                                                 }))
  sim.post_init()
  sim.load_weights_from_data_path(sim.recurrent_weights_path)
  
  sim.run_recurrent_dynamics_with_walk(  walk_time,
                                         num_snaps,
                                         theta_sigma,
                                         init_p=init_p,
                                         init_theta=np.pi/2,
                                         use_tuning_switch=True,
                                         switch_off_feedforward=True,
                                         feed_forward_off_value=5.,
                                         rec_gain_with_no_feedforward=1.,
                                         switch_off_times=[switch_off_time],
                                         switch_on_times=[switch_on_time],
                                         interpolate_inputs=True,
                                         track_cell_evo=True,
                                         track_cell_idx=track_cell_idx,
                                         force_walk=False,
                                         periodic_walk=True,
                                         use_theta_modulation=False,
                                         track_bump_evo=True,
                                         r_max=100.,
                                         synaptic_filter=False,
                                         walk_speed=0.25
                                        )
  sims.append(sim)
  

#%% SELECTING ONE EXAMPLE SIMULATION
sim=None  
sel_sim_idx=6
sel_sim=sims[sel_sim_idx]


#%% ### SINGLE-CELL PLOT WITH SWITCHING OFF FEED-FORWARD INPUTS (ATTRACTOR DYNAMICS)
  
walk_time=16. 
num_snaps=int(walk_time*50)
#4  # one example simulation out of the 10 realized network

input_color='m'
rec_color='gray'
inhib_color='dodgerblue'
time_out_color='limegreen'
lw=1.5

pl.figure(figsize=(8,3))
pl.subplots_adjust(left=0.125,right=0.99,hspace=0.7,wspace=0.,bottom=0.4)
time=np.arange(num_snaps)*sel_sim.delta_snap*sel_sim.dt


pl.subplot(111)
pl.plot(time,sel_sim.cell_hh_evo,color=input_color,lw=lw,label='feed-forward')
pl.plot(time,sel_sim.cell_rec_input_from_e_evo,color=rec_color,lw=lw,label='recurrent')
pl.plot(time,-sel_sim.cell_rec_input_from_i_evo,color=inhib_color,lw=lw,label='recurrent')
pl.plot(time,sel_sim.cell_rr_evo,color=time_out_color,lw=lw,label='output rate')
pp.custom_axes()
pl.xlabel('Time [s]')
pl.ylabel('Firing rate [1/s]')
pl.xlim([-0.1,walk_time+0.1])
pl.xticks(np.arange(17))
pl.yticks([0,10,20,30,40,50])
pl.axvline(switch_off_time,color='k',ls=pp.linestyles['densely dotted'],lw=1.5)
pl.axvline(switch_on_time,color='k',ls=pp.linestyles['densely dotted'],lw=1.5)


# mark switching off time
ax=pl.gca()
for idx in range(17):
  pl.plot(idx,48,'vk',ms=5)
fname = 'fig_attractor_one_cell_raw' 
#pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'],transparent=True)


#%% ### COMPUTE BUMP SPEED FOR WALK SIMUATIONS
speed_num_dt=1
filter_len=30
num_speed_snaps=num_snaps/speed_num_dt
speed_dt=sel_sim.dt*speed_num_dt
num_speed_steps=int(walk_time/speed_dt)        
speed_delta_snap=num_speed_steps/num_speed_snaps
speed_snap_time=speed_delta_snap*speed_dt
speed_time=np.arange(num_speed_snaps)*speed_snap_time
  
for sim in sims:
  det_phase_idx=np.ravel_multi_index(sim.bump_peak_evo[:,0].astype(int), (sim.n_e,sim.n_e))
  old_det_phase=sim.gp.phases[det_phase_idx,:]
  

  # time=np.arange(sim.recdyn_num_snaps)*snap_time

  bump_speed_raw=np.zeros(num_speed_snaps)
  
  for time_idx in range(1,num_speed_snaps):
   
    det_phase_idx=np.ravel_multi_index(sim.bump_peak_evo[:,time_idx*speed_num_dt].astype(int), (sim.n_e,sim.n_e))
    det_phase=sim.gp.phases[det_phase_idx,:]
    dist=gl.get_single_phase_periodic_dist(det_phase,old_det_phase,sim.gp.u1,sim.gp.u2)
    bump_speed_raw[time_idx]=dist/speed_snap_time
    old_det_phase=det_phase
    
  sim.bump_move_speed_evo=np.convolve(bump_speed_raw, np.ones((filter_len,))/filter_len, mode='full')[:num_speed_snaps]
  

#%% ### BUMP LOCATION AND AMPLITUDE WITH SWITCHING OFF FEED-FORWARD INPUTS (ATTRACTOR DYNAMICS)

pl.rc('font',size=10)

fig=pl.figure(figsize=(8,3))
time=np.arange(num_snaps)*sel_sim.delta_snap*sel_sim.dt
switch_off_time_idx=np.argmin(abs(time-switch_off_time))
switch_on_time_idx=np.argmin(abs(time-switch_on_time))

gs = pl.GridSpec(4,1,hspace=0.5,wspace=0.1,bottom=0.15,left=0.125,right=0.99,top=0.98)
 

ax=fig.add_subplot(gs[0,0])

for idx,curr_sim in enumerate(sims):
  pl.plot(time,curr_sim.bump_peak_evo[0,:]/30.,color='salmon')

pl.plot(time,sel_sim.bump_peak_evo[0,:]/30.,color='black',lw=1.5)  
pp.custom_axes()
pl.yticks([0,1])
pl.ylabel('Bump\nphase\n(vert.)',fontsize=10)
pl.xlim([-0.1,walk_time+0.1])
pl.gca().axes.get_xaxis().set_visible(False)
ax.spines['bottom'].set_color('none')
pl.axvline(switch_off_time,color='k',ls=pp.linestyles['densely dotted'],lw=1.5)
pl.axvline(switch_on_time,color='k',ls=pp.linestyles['densely dotted'],lw=1.5)


ax=fig.add_subplot(gs[1,0])

for curr_sim in sims:
  pl.plot(time,curr_sim.bump_peak_evo[1,:]/30.,color='salmon')

pl.plot(time,sel_sim.bump_peak_evo[1,:]/30.,color='black',lw=1.5)  
pp.custom_axes()
pl.yticks([0,1])
pl.ylabel('\nBump\nphase\n(horiz.)',fontsize=10)
pl.xlim([-0.1,walk_time+0.1])
pl.axvline(switch_off_time,color='k',ls=pp.linestyles['densely dotted'],lw=1.5)
pl.axvline(switch_on_time,color='k',ls=pp.linestyles['densely dotted'],lw=1.5)
pl.gca().axes.get_xaxis().set_visible(False)
ax.spines['bottom'].set_color('none')


ax=fig.add_subplot(gs[2,0])
for curr_sim in sims:
  pl.plot(speed_time[filter_len:],curr_sim.bump_move_speed_evo[filter_len:],'salmon')

pl.plot(speed_time[filter_len:],sel_sim.bump_move_speed_evo[filter_len:],color='black',lw=1.5)            
pp.custom_axes()
#pl.ylim(0,50)
pl.xlim([-0.1,walk_time+0.1])
pl.xticks(np.arange(17))
pl.ylabel('Bump\nspeed\n[1/s]',fontsize=10)
pl.ylim(-0.1,0.7)
pl.xlim([-0.1,walk_time+0.1])
pl.axvline(switch_off_time,color='k',ls=pp.linestyles['densely dotted'],lw=1.5)
pl.axvline(switch_on_time,color='k',ls=pp.linestyles['densely dotted'],lw=1.5)
pl.gca().axes.get_xaxis().set_visible(False)
ax.spines['bottom'].set_color('none')



ax=fig.add_subplot(gs[3,0])
for curr_sim in sims:
  pl.plot(time,curr_sim.bump_evo.max(axis=0),'salmon')

pl.plot(time,sel_sim.bump_evo.max(axis=0),color='black',lw=1.5)            
pp.custom_axes()
pl.ylim(0,50)
pl.xlim([-0.1,walk_time+0.1])
pl.xticks(np.arange(17))
pl.ylabel('Bump\npeak rate\n[1/s]',fontsize=10)
pl.xlim([-0.1,walk_time+0.1])
pl.axvline(switch_off_time,color='k',ls=pp.linestyles['densely dotted'],lw=1.5)
pl.axvline(switch_on_time,color='k',ls=pp.linestyles['densely dotted'],lw=1.5)
pl.xticks(np.arange(17))
pl.xlabel('Time [s]',fontsize=10)


#fname = 'fig_attractor_track_bump_pos' 
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'],transparent=True)

#%% ### BUMP OVER TIME WITH SWITCHING OFF FEED-FORWARD INPUTS (ATTRACTOR DYNAMICS)

pl.rc('font',size=10)

vmax_ff=15
vmax_out=30

pl.figure(figsize=(8,2))

plot_idx=1
snaps_to_plot=np.arange(0,800,num_snaps/16)

snaps_to_plot=list(snaps_to_plot)+[799]
pl.subplots_adjust(left=0.08,right=0.99,wspace=0.0)
for snap_idx in snaps_to_plot:

  
  pl.subplot(2,len(snaps_to_plot),plot_idx,aspect='equal')  
  poly=pp.plot_on_rhombus(sel_sim.gp.R_T,sel_sim.gp.grid_T,0,sel_sim.N_e,sel_sim.gp.phases,
                                sel_sim.bump_hh_evo[:,snap_idx],plot_axes=False,plot_rhombus=True,
                                plot_cbar=False,vmin=0,vmax=vmax_ff)

  #pl.plot(0,0,'ok',mfc='none',mew=1.1)
  pl.title('%.0f s'%(snap_idx*sel_sim.delta_snap*sel_sim.dt),fontsize=10)
  
  pl.subplot(2,len(snaps_to_plot),plot_idx+len(snaps_to_plot),aspect='equal')  
  poly=pp.plot_on_rhombus(sel_sim.gp.R_T,sel_sim.gp.grid_T,0,sel_sim.N_e,sel_sim.gp.phases,
                                sel_sim.bump_evo[:,snap_idx],plot_axes=False,plot_rhombus=True,
                                plot_cbar=False,vmin=0,vmax=vmax_out)

  plot_idx+=1
  #pl.plot(0,0,'ok',mfc='none',mew=1.1)
  
  
  det_phase_idx=np.ravel_multi_index(sel_sim.bump_peak_evo[:,snap_idx].astype(int), (sel_sim.n_e,sel_sim.n_e))
  det_phase=sel_sim.gp.phases[det_phase_idx,:] 
  #pl.plot(det_phase[0],det_phase[1],'.r')
  
  
fname = 'fig_attractor_bumps_raw' 
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'],transparent=True)

#%% ### GET ATTRACTOR SIMS
stim_on=False
batch=apm.batch_attractor_weight_seeds
batch.post_init()

attractor_sims=[]

for inputs_seed in range(10):
  sim_conn=RecAmp2PopLearn(sl.map_merge(apm.def_recamp_learn_params,{'inputs_seed':inputs_seed}))  
  sim=RecAmp2PopAttractor(sl.map_merge(batch.batch_default_map,
                                       {
                                         'recurrent_weights_path':sim_conn.data_path,
                                       }))
  #sim.run_and_save_attractor()
  sim.post_init(get_inputs=False)
  sim.load_attractor_outputs()
  sim.get_attractor_fields()
  attractor_sims.append(sim)



  
#%% ###  PLOT ATTRACTOR LANDSCAPES

# show the selected showcase netweork in the first position
sims_idxs=np.arange(10)
sims_idxs[0]=sel_sim_idx
sims_idxs[sel_sim_idx]=0

fig=pl.figure(figsize=(8,2))
pl.subplots_adjust(left=0.1,right=1,wspace=0.00,hspace=0.)

for plot_idx, sim_idx in enumerate(sims_idxs,1):
  
  walk_sim=sims[sim_idx]
  att_sim=attractor_sims[sim_idx] 
  
  
  pl.subplot(1,10,plot_idx,aspect='equal')  
  ax=pl.gca()
  ax.patch.set_facecolor('None')

    
  poly=att_sim.plot_landscape(stim_on,time_idx=None,rhombus=True,plot=True,attractor_plots=True,quiver=False
                              ,rhombus_color='black' if sim_idx==sel_sim_idx else 'salmon',rhombus_lw=1.5)



  det_phase_idx=np.ravel_multi_index(walk_sim.bump_peak_evo[:,switch_off_time_idx+1].astype(int), (walk_sim.n_e,walk_sim.n_e))
  det_phase=walk_sim.gp.phases[det_phase_idx,:] 
  pl.plot(det_phase[0],det_phase[1],'*y',ms=10,mec='k',mfc='gold',mew=0.5)

  
  det_phase_idx=np.ravel_multi_index(walk_sim.bump_peak_evo[:,switch_on_time_idx-1].astype(int), (walk_sim.n_e,walk_sim.n_e))
  det_phase=walk_sim.gp.phases[det_phase_idx,:] 
  pl.plot(det_phase[0],det_phase[1],'or',ms=7,mfc='none',mew=1)  
  pl.title('Net. %d'%(plot_idx),fontsize=10)

ax=fig.add_axes([0.05
                 ,0.3,0.01,0.3])
cbar=pp.colorbar(obj=poly,cax=ax,fixed_ticks=[0,1])
cbar.ax.tick_params(labelsize=10) 
  
fname = 'fig_attractor_landscapes_raw' 
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'],transparent=True)

#%% ### COMPUTE AVERAGE SPEED OF BUMP MOVEMENT    
all_avg_dists=np.zeros((10,batch.batch_default_map['recdyn_num_snaps']))

for inputs_seed in range(10):
  sim=attractor_sims[inputs_seed]
  old_det_phases=sim.get_detected_phases(sim.out_bumps_stim_off_evo_idx2d,0)

  num_steps=int(sim.time_stimulus_off/sim.dt)        
  delta_snap=num_steps/sim.recdyn_num_snaps
  snap_time=delta_snap*sim.dt  
  time=np.arange(sim.recdyn_num_snaps)*snap_time

  all_dists=np.zeros((sim.recdyn_num_snaps,sim.N_e))
  
  for time_idx in range(sim.recdyn_num_snaps):
    det_phases=sim.get_detected_phases(sim.out_bumps_stim_off_evo_idx2d,time_idx)
    dists=gl.get_periodic_dist_on_rhombus(sim.n_e,det_phases,old_det_phases,sim.gp.u1,sim.gp.u2)
    all_dists[time_idx,:]=dists
    old_det_phases=det_phases
  
  all_avg_dists[inputs_seed,:]=all_dists.mean(axis=1)
  
  
#%% ### PLOT AVERAGE SPEED OF BUMP MOVEMENT    

pl.figure(figsize=(4,2))
pl.subplots_adjust(left=0.2,right=0.99,wspace=0.00,hspace=0.,bottom=0.3)

pl.plot(time[1:],all_avg_dists[:,1:].T/snap_time,color='salmon',lw=1.5);
pl.plot(time[1:],all_avg_dists[sel_sim_idx,1:].T/snap_time,color='k',lw=1.5);

#pl.gca().set_xscale('log')
pl.xlabel('Time after cue removal [s]')
pl.ylabel('Bump speed [1/s]')
pp.custom_axes()

fname = 'fig_attractor_speed_raw' 
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'],transparent=True)