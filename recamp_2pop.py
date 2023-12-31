# -*- coding: utf-8 -*-


import numpy as np
import os
import datetime,time
import pylab as pl

import grid_utils.plotlib as pp
import grid_utils.gridlib as gl
import grid_utils.simlib as sl

from grid_utils.random_walk import RandomWalk
from grid_utils.spatial_inputs import SpatialInputs
from grid_utils.spatial_inputs import rhombus_mises



    
    
def dir_vect(theta):
  """
  Returns a 2-d vector given an angle theta
  """
  return np.array([np.cos(theta),np.sin(theta)])
  
def find_bump_peak_idxs(map1d,unravel=True,**kwargs):
  """
  Utility function that returns the 2D coordinates of an activity bump
  """
  from scipy.ndimage import gaussian_filter
    
  map_side=int(np.sqrt(len(map1d)))
  map2d=map1d.reshape(map_side,map_side)
  map2d_smoothed=gaussian_filter(map2d, sigma=3,mode='wrap') 
  max_idx=map2d_smoothed.argmax() 

  if unravel:
    return np.unravel_index(max_idx, map2d.shape)
  else:
    return max_idx

  

def get_bumps_idxs_all_shifts(r,N_e):

    bump_idsx2d=np.zeros((2,N_e))
    
    # compute the coordinates of the output bump of each stimulus bump shift
    for shift_idx in range(N_e):
      bx,by=find_bump_peak_idxs(r[:N_e,shift_idx],unravel=True)
      bump_idsx2d[0,shift_idx]=bx
      bump_idsx2d[1,shift_idx]=by   
      
    return bump_idsx2d


def get_activation_fun(activation,r_max):
  """
  Neuronal activation function
  """
  
  clip = lambda(x) : x.clip(min=0,max=r_max)
  tanh = lambda(x) : np.clip(np.tanh(x/float(r_max))*r_max,1e-16,r_max)

  if activation == 'linear':
    return lambda x : x
  if activation == 'clip':
    return clip
  elif activation == 'tanh':
    return tanh
  else:
    raise Exception('Invalid activation function')
      

  

class __RecAmp2Pop(object):
  """
  This class implements the basic methods for simulating the amplification model in 2D with 2 populations of 
  neuron (excitatory and inhibitory). This class is private and abstract and is the common building block
  upon which the classes RecAmp2PopLearn and RecAmp2PopSteady are written.
  The class RecAmp2PopLearn deals with the learning of the recurrent connectivity.
  The class RecAmp2PopSteady deals with the simulation of the output steady-state patterns
  """
  
      
  def __init__(self):
    pass
    
    
  def get_inputs(self,force_gen=False,comp_gridness_score=False,comp_tuning_index=True):
    """
    Read inputs from disk and loads into the object
    """
    #print self.paramMap
    
    
    self.inputs=SpatialInputs(sl.map_merge(self.paramMap,{'n':self.n_e}),
                               force_gen=force_gen,
                               comp_gridness_score=comp_gridness_score,
                               comp_tuning_index=comp_tuning_index)    
    
    
    self.inputs_flat=self.inputs.inputs_flat
      
    self.h_e=self.inputs_flat.T        
    self.h_i=np.zeros((self.N_i,self.NX))
    self.h=np.vstack([self.h_e,self.h_i])
    
    


    
  def get_hardwired_speed_weights(self):
    """
    Generate hardwired speed-weight connectivity matrix
    """
   
    phase_shift=self.speed_phase_shift
    
    # row 1 has the weights of speed cells to grid cell 1
    self.W_speed_east=np.zeros_like(self.W_ee)  
    self.W_speed_west=np.zeros_like(self.W_ee)  
    self.W_speed_north=np.zeros_like(self.W_ee)  
    self.W_speed_south=np.zeros_like(self.W_ee)  

    if self.use_eight_directions is True:
      self.W_speed_north_east=np.zeros_like(self.W_ee)  
      self.W_speed_north_west=np.zeros_like(self.W_ee)  
      self.W_speed_south_east=np.zeros_like(self.W_ee)  
      self.W_speed_south_west=np.zeros_like(self.W_ee)  


    for phase_idx,phase in enumerate(self.gp.phases):
      shifted_north_phase_idx=gl.get_pos_idx(phase+phase_shift*dir_vect(np.pi/2.),self.gp.phases)
      shifted_south_phase_idx=gl.get_pos_idx(phase+phase_shift*dir_vect(-np.pi/2.),self.gp.phases)
      shifted_east_phase_idx=gl.get_pos_idx(phase+phase_shift*dir_vect(0),self.gp.phases)
      shifted_west_phase_idx=gl.get_pos_idx(phase+phase_shift*dir_vect(-np.pi),self.gp.phases)

      self.W_speed_north[phase_idx,:]=self.W_ee[shifted_north_phase_idx,:]
      self.W_speed_south[phase_idx,:]=self.W_ee[shifted_south_phase_idx,:]
      self.W_speed_east[phase_idx,:]=self.W_ee[shifted_east_phase_idx,:]
      self.W_speed_west[phase_idx,:]=self.W_ee[shifted_west_phase_idx,:]  
      
      if self.use_eight_directions is True:
        shifted_north_east_phase_idx=gl.get_pos_idx(phase+phase_shift*dir_vect(np.pi/4),self.gp.phases)
        shifted_north_west_phase_idx=gl.get_pos_idx(phase+phase_shift*dir_vect(np.pi*3/4),self.gp.phases)
        shifted_south_east_phase_idx=gl.get_pos_idx(phase+phase_shift*dir_vect(-np.pi/4),self.gp.phases)
        shifted_south_west_phase_idx=gl.get_pos_idx(phase+phase_shift*dir_vect(-np.pi*3/4),self.gp.phases)
        
        self.W_speed_north_east[phase_idx,:]=self.W_ee[shifted_north_east_phase_idx,:]
        self.W_speed_north_west[phase_idx,:]=self.W_ee[shifted_north_west_phase_idx,:]
        self.W_speed_south_east[phase_idx,:]=self.W_ee[shifted_south_east_phase_idx,:]
        self.W_speed_south_west[phase_idx,:]=self.W_ee[shifted_south_west_phase_idx,:]
            
            
    
  def switch_to_tuned_inputs(self):
    """
    Switches on feed-forward spatial tuning (during the simulation)
    """
    
    self.h_e=self.inputs_flat.T
    self.h=np.vstack([self.h_e,self.h_i])


  def switch_to_untuned_inputs(self):
    """
    Switches off feed-forward spatial tuning (during the simulation)
    """

    self.h_e=self.inputs.noise_flat.T
    self.h=np.vstack([self.h_e,self.h_i])


  def switch_to_no_feedforward_inputs(self):
    """
    Switches off feed-forward spatial tuning (during the simulation)
    """

    self.h_e=np.ones_like(self.inputs.noise_flat.T)*self.feed_forward_off_value
    self.h=np.vstack([self.h_e,self.h_i])

    
    
  def get_tuned_excitatory_weights(self,binary=True):
    """
    Computes a tuned excitatory connectivity matrix 
    It sets to W_max_e the weights of the num_conns_ee connections with the smallest phase difference
    If fixed_connectivity_tuning is set to a value smaller than 1, only the specified fraction of connections will be tuned
    """
    
    self.W_ee=np.zeros((self.N_e,self.N_e))
    
    if not hasattr(self,'fixed_connectivity_tuning'):
      self.fixed_connectivity_tuning=1
    
    num_tuned_conns=int(np.floor(self.fixed_connectivity_tuning*self.num_conns_ee))
    num_untuned_conns=self.num_conns_ee-num_tuned_conns
      
    for i in xrange(self.N_e):
      ref_phase=self.gp.phases[i,:]
      dists=gl.get_periodic_dist_on_rhombus(self.n_e,ref_phase,self.gp.phases,self.gp.u1,self.gp.u2)

      if binary is True:
        sorted_idxs=np.argsort(dists)      
        tuned_idxs=sorted_idxs[:self.num_conns_ee]
        np.random.shuffle(tuned_idxs)

        all_idxs=np.arange(self.N_e)
        np.random.shuffle(all_idxs)

        self.W_ee[i,tuned_idxs[0:num_tuned_conns]]=self.W_max_ee
        self.W_ee[i,all_idxs[:num_untuned_conns]]=self.W_max_ee
      else:
        
        # initialize bump activity to zero
        bump=np.zeros(self.N_e)
      
        # To account for the periodicity of the phase space we need to add up 9 bumps that are shifted
        # of +- one period in each dimension (i.e., 3 horizontal shifts + 3 vertical shifts)
        for xp in (-1,0,1):
          for yp in (-1,0,1):
            shift=xp*self.gp.u1+yp*self.gp.u2
            bump+=rhombus_mises(self.gp.phases-ref_phase[np.newaxis,:]+shift,0.1)
        bump=bump/bump.sum()*self.W_tot_ee
        self.W_ee[i,:]=bump
        
        #self.W_ee[i,:]=expit(2*(1-dists/dists.max()))
        #self.W_ee[i,:]/=self.W_ee[i,:].sum()
        #self.W_ee[i,:]*=self.W_tot_ee
      
    self.W[:self.N_e,:self.N_e]=self.W_ee      


  def get_random_inhibitory_weights(self):
    """
    Compute random inhibitory connectivity matrix
    """
    
    self.W_ei=np.zeros((self.N_e,self.N_i))
    self.W_ie=np.zeros((self.N_i,self.N_e)) 
    self. W_ii=np.zeros((self.N_i,self.N_i))

    
    # connections to the excitatory neurons  
    for row_idx in range(self.N_e):
        
      # from ihibitory
      all_idxs_ei=np.arange(self.N_i)
      np.random.shuffle(all_idxs_ei)
      self.W_ei[row_idx,all_idxs_ei[0:self.num_conns_ei]]=self.W_max_ei  
        
    # connections to inhibitory neurons
    for row_idx in range(self.N_i):
    
      # from exitatory      
      all_idxs_ie=np.arange(self.N_e)
      np.random.shuffle(all_idxs_ie)
      self.W_ie[row_idx,all_idxs_ie[0:self.num_conns_ie]]=self.W_max_ie
      
      # from inhibitory
      all_idxs_ii=np.arange(self.N_i)
      np.random.shuffle(all_idxs_ii)
      self.W_ii[row_idx,all_idxs_ii[0:self.num_conns_ii]]=self.W_max_ii
     
     
    self.W[:self.N_e,self.N_e:]=self.W_ei
    self.W[self.N_e:,:self.N_e]=self.W_ie
    self.W[self.N_e:,self.N_e:]=self.W_ii
  

  def post_init(self,force_gen_inputs=False,comp_gridness_score=False,comp_tuning_index=True,get_inputs=True):
    
    # set the seed 
    np.random.seed(self.seed)    

    # switches between local recurrent inhibition (n_i>0) or feed-forward inhibition (r0<0)
    assert(self.n_i==0 or self.r0==0)
    assert(self.n_i>0 or self.r0<0)
       
    #print sl.params_to_str(self.paramMap)


    self.N_e=self.n_e**2                                      # total number of excitatory neurons
    self.N_i=self.n_i**2                                      # total number of inhibitory neurons
    self.N=self.N_e+self.N_i                                  # total number of neurons
    self.NX=self.nx**2                                        # total number of space samples   

    
    self.num_conns_ee=int(np.floor(self.N_e*self.frac_conns_ee))    
    
    
    if self.n_i>0:
      
      self.num_conns_ie=int(np.floor(self.N_e*self.frac_conns_ie))    
      self.num_conns_ei=int(np.floor(self.N_i*self.frac_conns_ei))    
      self.num_conns_ii=int(np.floor(self.N_i*self.frac_conns_ii))    
          
    # mean input/output weight for the excitatory neurons
    self.W_av_star=float(self.W_tot_ee)/self.N_e

    # maximal connection strength for excitatory/inhibitory neurons
    self.W_max_ee=float(self.W_tot_ee)/self.num_conns_ee
        
    
    if self.n_i>0:
      self.W_max_ie=float(self.W_tot_ie)/self.num_conns_ie

      self.W_max_ei=-float(self.W_tot_ei)/self.num_conns_ei
      self.W_max_ii=-float(self.W_tot_ii)/self.num_conns_ii
           
    # get grid properties
    self.gp=gl.GridProps(self.n_e,self.grid_T,self.grid_angle)
    
    # compute recurrent connectivity matrix (W_ee can be overwritten by learning or loaded from disk)              
    self.W=np.zeros((self.N,self.N))
    self.get_tuned_excitatory_weights()
    if self.N_i>0:
      self.get_random_inhibitory_weights()    
    self.zero_phase_idx = gl.get_pos_idx([0.,0.],self.gp.phases)

    # get inputs
    if get_inputs:
      self.get_inputs(force_gen_inputs,comp_gridness_score,comp_tuning_index)

    
    
    
  def run_recurrent_dynamics(self,
                             record_mean_max=True,
                             record_rec_input_evo=False,
                             activation='tanh',
                             r_max=100,
                             init_r=None,
                             custom_h=None):
    """
    The output activity is computed for each pixel independently without modeling
    the random walk of the virtual rat explicitely
    """
    
    print '\nRunning recurrent dynamics'

    activation_fun=get_activation_fun(activation,r_max)
 
    # initialization of the feed-forward input
    if custom_h is None:
      h=self.h
    else:
      assert(custom_h.shape[0]==self.N_e+self.N_i)
      h=custom_h

    # initialization of the output rates
    if init_r is None:
      r=np.zeros_like(h)
    else:
      assert(init_r.shape[0]==self.N_e+self.N_i)
      r=init_r
    
    # check that shapes matches 
    assert(h.shape==r.shape)
    
    # number of spatial locations
    num_pos=h.shape[1]
    
    num_steps=int(self.recdyn_time/self.dt)

    if record_mean_max is True:        
      self.rec_input_mean_vect=np.zeros((self.N,num_steps))
      self.rec_input_max_vect=np.zeros((self.N,num_steps))
      self.r_mean_vect=np.zeros((self.N,num_steps))
      self.r_max_vect=np.zeros((self.N,num_steps))

    self.r_evo=np.zeros((self.N,num_pos,self.recdyn_num_snaps))
    
    if record_rec_input_evo:
      self.rec_input_evo=np.zeros((self.N,num_pos,self.recdyn_num_snaps))
        
    delta_snap=num_steps/self.recdyn_num_snaps
    
    snap_idx=0
    
    rec_input=np.zeros_like(r)
    start_clock=time.time()
        
    for t in range(num_steps):
      #print '%d/%d'%(t,num_steps)
      
      if np.remainder(t,delta_snap)==0 and snap_idx<self.recdyn_num_snaps:
        
        sl.print_progress(snap_idx,self.recdyn_num_snaps,start_clock=start_clock,step=1)

        self.r_evo[:,:,snap_idx]=r
        #print('r_min: ',r.min())
        #print('r_max: ',r.max())
        
        if record_rec_input_evo:
          self.rec_input_evo[:,:,snap_idx]=rec_input

        snap_idx+=1

      if record_mean_max:
        self.rec_input_mean_vect[:,t]=np.mean(rec_input,axis=1)
        self.rec_input_max_vect[:,t]=np.max(rec_input,axis=1)
        self.r_mean_vect[:,t]=np.mean(r,axis=1)
        self.r_max_vect[:,t]=np.max(r,axis=1)
        
      # recurrent input        
      rec_input=np.dot(self.W,r)
      
      # total input, add feed-forward inhibition if recurrent inhibition is not explicitely modeled
      tot_input=h+rec_input            
      if self.N_i==0:
        tot_input+=self.r0
        
      tot_activation = activation_fun(tot_input)
        
      r=r+(self.dt/self.tau)*(-r+tot_activation)
      

    self.r=r
    
  def compute_steady_scores(self,comp_inhibitory_scores=True,force_input_scores=False):
    
    # excitatory scores
    R_e=self.r[0:self.N_e,:].T
    R_e=np.reshape(R_e,(self.nx,self.nx,self.N_e))
    self.re_scores,re_spacings=gl.gridness_evo(R_e[:,:,:],self.L/self.nx,num_steps=10)

    if comp_inhibitory_scores is True:
      # inhibitory scores
      R_i=self.r[self.N_e:,:].T
      R_i=np.reshape(R_i,(self.nx,self.nx,self.N_i))
      self.ri_scores,ri_spacings=gl.gridness_evo(R_i[:,:,:],self.L/self.nx,num_steps=10)
    
    # input scores
    if not hasattr(self.inputs,'in_scores') or force_input_scores:
      print 'Computing input scores'
      self.inputs.gen_data(False,comp_gridness_score=True)
      
    self.he_scores=self.inputs.in_scores

  def save_steady_scores(self):
    """
    Updates data files by adding gridness scores
    """
        
    assert (hasattr(self,'re_scores') and hasattr(self,'ri_scores') and hasattr(self,'he_scores')  )
    data=np.load(self.data_path,allow_pickle=True)
    dataMap=dict(data.items())
    scores_attrs=['re_scores','ri_scores','he_scores']
    
    for scores_attr in scores_attrs:
      
      assert(hasattr(self,scores_attr)),'%s is not a field'%scores_attr
      dataMap[scores_attr]=getattr(self,scores_attr)

    np.savez(self.data_path,**dataMap)


    
  def load_steady_scores(self):
    """
    Loads gridness scores. Generates and save them if not present in the data file.
    """

    data=np.load(self.data_path,allow_pickle=True)
    scores_attrs=['re_scores','ri_scores','he_scores']
    
    if 're_scores' not in data.keys():
      self.compute_steady_scores()
      self.save_steady_scores()
      data=np.load(self.data_path,allow_pickle=True)   
      
    for scores_attr in scores_attrs:
      
      assert(scores_attr in data.keys())
      setattr(self,scores_attr,data[scores_attr])
        
                
    
  def update_speed_weights_step(self):
    """
    Update step to learn speed weights
    """
    
    weights_list = [self.W_speed_east, self.W_speed_west,self.W_speed_north,self.W_speed_south]
    speed_input_list = [self.speed_inputs_east,self.speed_inputs_west,
                       self.speed_inputs_north,self.speed_inputs_south]
    
    if self.use_eight_directions is True:
      weights_list+=[self.W_speed_north_east,
                     self.W_speed_north_west,self.W_speed_south_east,self.W_speed_south_west]
      
      speed_input_list+=[self.speed_inputs_north_east,self.speed_inputs_north_west,                                  
                         self.speed_inputs_south_east,self.speed_inputs_south_west]

    
    for weights,speed_input in zip(weights_list,speed_input_list):
            
            
      weight_update=speed_input*(self.rr[:self.N_e]-self.input_mean)*(self.rr_e_trace.T-self.input_mean)
      weights+=self.learn_rate_speed_weights*weight_update


      # normalize to fixed mean of incoming and outgoing weights
      weights-=(weights.mean(axis=1)-self.W_av_star)[:,np.newaxis]
      weights-=(weights.mean(axis=0)-self.W_av_star)[np.newaxis,:]
            
      # clip weights      
      np.clip(weights,0,self.W_max_e,out=weights)
  


  def update_speed_input_step(self,curr_v):
    
    
    """
    Update step for speed inputs (also used to learn speed weights)
    """
    
    # update speed inputs                
    self.speed_inputs_east*=0
    self.speed_inputs_west*=0
    self.speed_inputs_north*=0
    self.speed_inputs_south*=0

    if self.use_eight_directions is True:  
      self.speed_inputs_north_east*=0
      self.speed_inputs_north_west*=0
      self.speed_inputs_south_east*=0
      self.speed_inputs_south_west*=0
    
    #speed_values=self.rr[:self.N_e,0] 
    speed_values=np.ones((self.N_e,1))

    if curr_v[0]>0:
      
      # north-east
      if  self.use_eight_directions is True and curr_v[1]>0:
        self.speed_inputs_north_east=speed_values                   
        
      # south-east  
      elif self.use_eight_directions is True and curr_v[1]<0:
        self.speed_inputs_south_east=speed_values
        
      #east  
      else:
        self.speed_inputs_east=speed_values


    elif curr_v[0]<0:

      # north-west                    
      if self.use_eight_directions is True and  curr_v[1]>0:
        self.speed_inputs_north_west=speed_values

      # south-west                    
      elif self.use_eight_directions is True and curr_v[1]<0:
        self.speed_inputs_south_west=speed_values
        
       # west 
      else:
        self.speed_inputs_west=speed_values

    else:  
      # north
      if curr_v[1]>0:
        self.speed_inputs_north=speed_values

      # south
      elif curr_v[1]<0:
        self.speed_inputs_south=speed_values
    



  def update_total_speed_input_step(self,curr_v):
    """
    Update step to compute the total speed input to add to the recurrent dynamics
    """
      
    tot_speed_input_east=np.dot(self.W_speed_east,self.speed_inputs_east)/self.N_e
    tot_speed_input_west=np.dot(self.W_speed_west,self.speed_inputs_west)/self.N_e
    tot_speed_input_north=np.dot(self.W_speed_north,self.speed_inputs_north)/self.N_e
    tot_speed_input_south=np.dot(self.W_speed_south,self.speed_inputs_south)/self.N_e

    self.tot_speed_input_all_padded[:self.N_e,0]=\
    tot_speed_input_east+tot_speed_input_west+\
    tot_speed_input_north+tot_speed_input_south
    
    if self.use_eight_directions is True:
      tot_speed_input_north_east=np.dot(self.W_speed_north_east,
                                        self.speed_inputs_north_east)/self.N_e
      tot_speed_input_north_west=np.dot(self.W_speed_north_west,
                                        self.speed_inputs_north_west)/self.N_e
      tot_speed_input_south_east=np.dot(self.W_speed_south_east,
                                        self.speed_inputs_south_east)/self.N_e
      tot_speed_input_south_west=np.dot(self.W_speed_south_west,
                                        self.speed_inputs_south_west)/self.N_e
    
      self.tot_speed_input_all_padded[:self.N_e,0]+=\
      tot_speed_input_north_east+tot_speed_input_north_west+\
      tot_speed_input_south_east+tot_speed_input_south_west
      
    else:
      
      # diagonal move with four directions
      if abs(curr_v[0])>0 and abs(curr_v[1])>0:
        self.tot_speed_input_all_padded[:self.N_e,0]*=.5




  def update_recurrent_weights_step(self):
    """
    Update step to learn recurrent weights
    """
   
    # update weights: hebbian term
    self.delta_Wee=self.learn_rate*(self.rr[0:self.N_e]-self.input_mean)*\
    (self.rr[0:self.N_e].T-self.input_mean)
        
    self.W_ee+=self.dt*self.delta_Wee

    # update weights: normalize to fixed mean of incoming and outgoing weights
    self.W_ee-=(self.W_ee.mean(axis=1)-self.W_av_star)[:,np.newaxis]
    self.W_ee-=(self.W_ee.mean(axis=0)-self.W_av_star)[np.newaxis,:]
            
    # clip weights      
    self.W_ee=np.clip(self.W_ee,0,self.W_max_ee)
    
    # update excitatory weights in the big weight matrix
    self.W[:self.N_e,:self.N_e]=self.W_ee
        

    
  def run_recurrent_dynamics_with_walk(self,
                                       walk_time,
                                       num_snaps,
                                       theta_sigma,
                                       learn_recurrent_weights=False,
                                       learn_speed_weights=False,
                                       track_bump_evo=False,
                                       track_cell_evo=False,
                                       track_cell_idx=0,
                                       
                                       
                                       run_in_circle=False,
                                       sweep=False,
                                       fixed_position=False,
                                         
                                       use_recurrent_input=True,
                                       use_theta_modulation=False,
                                       theta_freq=10.,
                                       
                                       use_tuning_switch=False,
                                       switch_off_feedforward=False,
                                       feed_forward_off_value=0.,
                                       rec_gain_with_no_feedforward=1.,
                                       switch_off_times=[],
                                       switch_on_times=[],
                                       tuning_time=1.,
                                       evo_idxs=[],
                                       
                                       force_walk=False,
                                       periodic_walk=False,
                                       init_p=np.array([0.,0.]),
                                       init_theta=0.0,
                                       
                                       interpolate_inputs=False,
                                       
                                       activation='tanh',
                                       r_max=100.,
                                       
                                       position_dt=None,
                                       
                                       synaptic_filter=False,
                                       tau_synaptic=0.2,
                                       
                                       walk_speed=None
                                       
                                       ):
                                         

    # we cannot learn both recurrent weights and speed weights at the same time         
    assert(learn_recurrent_weights is False or  learn_speed_weights  is False)
                        
    # at most one of these flag can be true
    assert ((int(run_in_circle)+int(sweep)+int(fixed_position)<2))
            
    self.run_in_circle=run_in_circle
    self.sweep=sweep
    self.fixed_position=fixed_position
    self.rec_gain_with_no_feedforward=rec_gain_with_no_feedforward
    
    self.synaptic_filter=synaptic_filter
    self.tau_synaptic=tau_synaptic
    
    self.feed_forward_off_value=feed_forward_off_value
    self.walk_speed = walk_speed if walk_speed is not None else self.speed
    
    # copy initial weights in case they are rescaled to compensate the absence of feed-forward inputs
    self.Wee_nogain=self.W_ee.copy()
    self.W_nogain=self.W.copy()
    
    
        
    # initialize swithing times (in case we are turning off feed-forward input or their tuning)
    if len(switch_on_times)>0:
      curr_switch_on_time=switch_on_times.pop(0)
    else:
      curr_switch_on_time=None
 
    if len(switch_off_times)>0:
      curr_switch_off_time=switch_off_times.pop(0)
    else:
      curr_switch_off_time=None
         
    # activation function
    activation_fun=get_activation_fun(activation,r_max)


    ### ============= LEARNING RECURRENT WEIGHTS ===============================

    if learn_recurrent_weights is True:
      
      # we cannot learn recurrent weights with speed input
      assert(self.use_speed_input is False)
      
      print 'Learning recurrent weights with random walk'    

      
      # initialize connections to the excitatory neurons  
      self.W_ee0=np.zeros((self.N_e,self.N_e))
      
      # initial weights are random
      if self.start_with_zero_connectivity is False:
        
        print 'Initializing weights at random to the upper bound'
        for row_idx in xrange(self.N_e):
          idxs=np.arange(self.N_e)
          np.random.shuffle(idxs)
          self.W_ee0[row_idx,idxs[0:self.num_conns_ee]]=self.W_max_ee
          
      # initial weights are set to zero    
      else:
        print 'Initializing weights to zero'
          
      # initializations
      self.learn_snap_idx=0
      self.learn_walk_step_idx=0
      self.W_ee=self.W_ee0
      self.W[:self.N_e,:self.N_e]=self.W_ee
      self.rr=np.zeros((self.N,1))      

      # weight evolution vectors
      if len(evo_idxs) == 0 :
        self.Wee_evo=np.zeros((self.N_e,num_snaps))
      else:
        self.Wee_evo=np.zeros((len(evo_idxs),self.N_e,num_snaps))
        
      self.mean_rr_evo=np.zeros(num_snaps)

    

    ### ============= LEARNING SPEED WEIGHTS ===================================
    
    elif learn_speed_weights is True:

      # we cannot learn speed weights with speed input
      assert(self.use_speed_input is False)

      print 'Learning speed weights with random walk'    
 
      self.W_speed_east_evo=np.zeros((self.N_e,num_snaps))
      
      # target mean weight for input and output connections        
      self.W_av_star=(np.float(self.W_max_e)*self.num_conns_ee)/self.N_e
 
      # initialize speed weights to zero
     
      # row 1 has the weights of speed cells to grid cell 1
      self.W_speed_east=np.zeros_like(self.W_ee)  
      self.W_speed_west=np.zeros_like(self.W_ee)  
      self.W_speed_north=np.zeros_like(self.W_ee)  
      self.W_speed_south=np.zeros_like(self.W_ee)  
    
      if self.use_eight_directions is True:
        self.W_speed_north_east=np.zeros_like(self.W_ee)  
        self.W_speed_north_west=np.zeros_like(self.W_ee)  
        self.W_speed_south_east=np.zeros_like(self.W_ee)  
        self.W_speed_south_west=np.zeros_like(self.W_ee) 
        
        
              

    ### ============= RUN DYNAMICS WITHOUT LEARNING ============================      
                        
    else:
      print 'Recurrent dynamics with random walk'    
      
      
    print 'use_speed_input: %s'%self.use_speed_input
    

    self.num_walk_steps = int(walk_time/self.dt)    

    # rate at which we shall update the position (there is the option to interpolate inputs between updates)
    if position_dt is None:
      self.position_dt=self.L/self.nx/self.speed
    else:
      self.position_dt=position_dt
      
    self.pos_dt_scale=int(self.position_dt/self.dt)      

    self.walk=RandomWalk(sl.map_merge(self.paramMap,{  'walk_time':walk_time,
                                                       'position_dt':self.position_dt,
                                                       'theta_sigma':theta_sigma,
                                                       'sweep':sweep,
                                                       'init_p':init_p,
                                                       'init_theta':init_theta,
                                                       'periodic_walk':periodic_walk,
                                                       'speed':self.walk_speed,
                                                       }),
                           force=force_walk,
                           #init_p=init_p,
                           #init_theta=init_theta,
                           )

           
    self.delta_snap = int(np.floor(float(self.num_walk_steps)/(num_snaps)))
    assert(self.delta_snap>0) 
    
    print 'pos_dt_scale: %d'%self.pos_dt_scale
    print 'delta_snap: %d'%self.delta_snap
        
    self.start_clock=time.time()
    self.startTime=datetime.datetime.fromtimestamp(time.time())
    self.startTimeStr=self.startTime.strftime('%Y-%m-%d %H:%M:%S')
    
    # initializations
    self.snap_idx=0
    self.walk_step_idx=0
    self.rr=np.zeros((self.N,1))    

        
    self.start_clock=time.time()

    self.r_e_walk_map=np.zeros((self.N_e,self.NX))

   
    self.visits_map=np.zeros(self.NX)        


    if self.use_speed_input or learn_speed_weights:
      self.speed_inputs_east=np.zeros(self.N_e)
      self.speed_inputs_west=np.zeros(self.N_e)
      self.speed_inputs_north=np.zeros(self.N_e)
      self.speed_inputs_south=np.zeros(self.N_e)
  
      if self.use_eight_directions is True:
        
        self.speed_inputs_north_east=np.zeros(self.N_e)
        self.speed_inputs_north_west=np.zeros(self.N_e)
        self.speed_inputs_south_east=np.zeros(self.N_e)
        self.speed_inputs_south_west=np.zeros(self.N_e)

      self.tot_speed_input_all_padded=np.zeros((self.N,1))
  

  
    if track_bump_evo is True:
      self.bump_peak_evo=np.zeros((2,num_snaps))
      self.bump_hh_peak_evo=np.zeros((2,num_snaps))
            
      self.bump_evo=np.zeros((self.N_e,num_snaps))
      self.bump_hh_evo=np.zeros((self.N_e,num_snaps))
      self.bump_rec_evo=np.zeros((self.N_e,num_snaps))
      self.bump_speed_evo=np.zeros((self.N_e,num_snaps))

    if track_cell_evo is True:
      
      self.cell_rr_evo=np.zeros(num_snaps)
      self.cell_hh_evo=np.zeros(num_snaps)
      self.cell_rec_input_evo=np.zeros(num_snaps)
      self.cell_rec_input_from_e_evo=np.zeros(num_snaps)
      self.cell_rec_input_from_i_evo=np.zeros(num_snaps)

    
    
    pos_idx=-1
    curr_p=self.walk.pos[pos_idx]        

    self.hh_e=self.h_e[:,pos_idx]
    self.hh_i=self.h_i[:,pos_idx]
        
    
    # feed-forward input vector
    self.hh=np.zeros((self.N_e+self.N_i,1))    
    self.next_hh=np.zeros((self.N_e+self.N_i,1))    
    
    tot_input=np.zeros_like(self.hh)
    filtered_tot_input=np.zeros_like(self.hh)
    
    
    # run the simulation
    for step_idx in xrange(self.num_walk_steps):

      #print 'step_idx: %d'%step_idx

      if self.fixed_position is False:      
        
      # ==== start of updating rat position ===============================

        if np.remainder(step_idx,self.pos_dt_scale)==0:
                   
          #print 'updating position, interpolate_inputs=%d'%interpolate_inputs
          
          # if we are at the end of the walk we start again
          if self.walk_step_idx>=self.walk.walk_steps:
            self.walk_step_idx=0
  
          # read inputs at this walk step       
          new_pos_idx= self.walk.pidx_vect[self.walk_step_idx]
          
          # the position has really changed from the last walk step
          # note that the position could still be the same because the rat moved less
          # than the discretization step used for space, that is, L/nx. 
          # on straight trajectories position shall update every L/nx/speed 
          
          if not (new_pos_idx == pos_idx):
                      
            pos_idx=new_pos_idx
            
            new_p=self.walk.pos[pos_idx]
    
            # with speed input or learning speed weights we need to update current direction
            if self.use_speed_input is True or learn_speed_weights is True:
    
              if step_idx>0:
                dp=new_p-curr_p
                
                # if we are changing position update current direction
                if not (dp[0]==0. and dp[1]==0):
                  curr_v=dp
    
                  self.update_speed_input_step(curr_v)
                              
                  # compute total weighted speed inputs to add to the recurrent dynamics
                  if self.use_speed_input is True:
                    self.update_total_speed_input_step(curr_v)
    
                  # update speed weights
                  if learn_speed_weights is True:
                    self.update_speed_weights_step() 
                    
            curr_p=new_p
    
            # update the input at this position        
            self.hh[:self.N_e,0]=self.h_e[:,pos_idx]
            self.hh[self.N_e:,0]=self.h_i[:,pos_idx]
  
            # get the inputs at the next different position for interpolation
            if interpolate_inputs:
              
              j=1
              while(j<1000):
                            
                # get inputs at the next different position (for interpolation)
                next_walk_step_idx=self.walk_step_idx+j
                if next_walk_step_idx>=self.walk.walk_steps:
                  next_walk_step_idx=0
                  
                next_pos_idx=self.walk.pidx_vect[next_walk_step_idx]
                
                if not(next_pos_idx == pos_idx):
                  break
                
                j+=1
              
              # we found the next position
              if j<1000:
                self.next_hh[:self.N_e,0]=self.h_e[:,next_pos_idx]
                self.next_hh[self.N_e:,0]=self.h_i[:,next_pos_idx]
                
                # the next different position is j walk steps away
                hh_increment=(self.next_hh-self.hh)/(self.pos_dt_scale*j)
                
              # we did not find it, no increment  
              else:
                hh_increment=np.zeros_like(self.hh)
            
          else:
            
            if interpolate_inputs:
              ### the position was the same -> interpolate
              self.hh+=hh_increment
  
                  
          self.walk_step_idx+=1 
          
        # ==== End of update of rat poistion =================================
          
        else:
        
          if interpolate_inputs:
            ### time has passed but position not updated -> interpolate
            self.hh+=hh_increment
          
        
      # tuning switch
      if use_tuning_switch is True:# and tuning_switch_count<len(switch_times):
                      
        if curr_switch_on_time is not None and step_idx*self.dt>=curr_switch_on_time:
            print 'Switching back to tuned inputs, time=%.3f'%(step_idx*self.dt)
            self.switch_to_tuned_inputs()
            self.hh_e=self.h_e[:,pos_idx]
            self.hh_i=self.h_i[:,pos_idx]
            self.hh=np.vstack((self.hh_e[:,np.newaxis],self.hh_i[:,np.newaxis]))            

            # replug the normal recurrent weights
            self.W_ee=self.Wee_nogain
            self.W[:self.N_e,:self.N_e]=self.W_ee
  
            if len(switch_on_times)>0:
              curr_switch_on_time=switch_on_times.pop(0)
            else:
              curr_switch_on_time=None
            
   
        if curr_switch_off_time is not None and step_idx*self.dt>=curr_switch_off_time:
           
          if switch_off_feedforward is True:
            print 'Switching off feed-forward inputs, time=%.3f'%(step_idx*self.dt)
            self.switch_to_no_feedforward_inputs()
            # add a gain factor to the excitatory recurrent weights
            self.W_ee=self.Wee_nogain*rec_gain_with_no_feedforward
            self.W[:self.N_e,:self.N_e]=self.W_ee
            
          else:
            print 'Switching to untuned inputs, time=%.3f'%(step_idx*self.dt)
            self.switch_to_untuned_inputs()
          
          self.hh_e=self.h_e[:,pos_idx]
          self.hh_i=self.h_i[:,pos_idx]
          self.hh=np.vstack((self.hh_e[:,np.newaxis],self.hh_i[:,np.newaxis]))
          
          if len(switch_off_times)>0:
            curr_switch_off_time=switch_off_times.pop(0)
          else:
            curr_switch_off_time=None
          

        
      # reset total input
      tot_input*=0      
        
      # get feedforward input
      self.ff_input=self.hh
      
      # add feed-forward input
      tot_input+=self.ff_input        

      # add theta modulation (of the feed-forward inputs) if necessary
      if use_theta_modulation is True:
        theta_signal=0.5*(np.cos(2*np.pi*step_idx*self.dt*theta_freq)+1) 
        tot_input*=theta_signal

      # compute recurrent input
      self.rec_input=np.dot(self.W,self.rr)
      
      # recurrent input from excitatory and inhibitory cells
      if track_cell_evo is True:  
        self.rec_input_from_e=np.dot(self.W[:,:self.N_e],self.rr[:self.N_e,:])
        self.rec_input_from_i=np.dot(self.W[:,self.N_e:],self.rr[self.N_e:,:])
      
      # add  recurrent and speed input
      if self.use_speed_input is True:        
        tot_input+=(1-self.speed_input_scale)*self.rec_input+\
                            self.speed_input_scale*self.tot_speed_input_all_padded
                            
      # add recurrent input only                     
      elif use_recurrent_input is True:
          tot_input+=self.rec_input
        
      # add feed-forward inhibition if needed  (in case of no recurrent inhibition)
      if self.N_i==0:
        tot_input+=self.r0
        
#      # add theta modulation (of the total input) if necessary
#      if use_theta_modulation is True:
#        theta_signal=0.5*(np.cos(2*np.pi*step_idx*self.dt*theta_freq)+1) 
#        tot_input*=theta_signal        
        
      if self.synaptic_filter is True: 
        filtered_tot_input+=(self.dt/self.tau_synaptic)*(-filtered_tot_input+tot_input)
      else:
        filtered_tot_input=tot_input
        
      # compute firing-rate output  
      self.rr+=(self.dt/self.tau)*(-self.rr+activation_fun(filtered_tot_input))

      # record output and position      
      self.r_e_walk_map[:,pos_idx]+=self.rr[:self.N_e,0]        
      self.visits_map[pos_idx]+=1
       
      
      # update recurrent weights if needed
      if learn_recurrent_weights is True:
        self.update_recurrent_weights_step() 
        
    
      # progress 
      if np.remainder(step_idx,self.delta_snap)==0:  
        
        sl.print_progress(self.snap_idx,num_snaps,self.start_clock)
        
        # track recurrent weights evolution        
        if learn_recurrent_weights is True:
          
          if len(evo_idxs)==0:
            self.Wee_evo[:,self.snap_idx]=self.W_ee[self.zero_phase_idx,:]
          else:
            for i,evo_cell_idx in enumerate(evo_idxs):
              self.Wee_evo[i,:,self.snap_idx]=self.W_ee[evo_cell_idx,:]
            
          self.mean_rr_evo[self.snap_idx]=self.rr.mean()
          
        if learn_speed_weights is True:
          self.W_speed_east_evo[:,self.snap_idx]=self.W_speed_east[self.zero_phase_idx,:]

        
        # track bump evolution
        if track_bump_evo is True and self.snap_idx<num_snaps:
          
          self.bump_evo[:,self.snap_idx]=self.rr[:self.N_e,0]
          self.bump_hh_evo[:,self.snap_idx]=self.hh[:self.N_e,0]
          self.bump_rec_evo[:,self.snap_idx]=self.rec_input[:self.N_e,0]
          
          if self.use_speed_input is True:
            self.bump_speed_evo[:,self.snap_idx]=self.tot_speed_input_all_padded[:self.N_e,0]
                  
          bump_peak_xy=find_bump_peak_idxs(self.rr[:self.N_e,0])
          bump_hh_peak_xy=find_bump_peak_idxs(self.hh[:self.N_e,0])
          
          if bump_peak_xy is not None:
            self.bump_peak_evo[0,self.snap_idx]=bump_peak_xy[0]
            self.bump_peak_evo[1,self.snap_idx]=bump_peak_xy[1]

          if bump_hh_peak_xy is not None:
            self.bump_hh_peak_evo[0,self.snap_idx]=bump_hh_peak_xy[0]
            self.bump_hh_peak_evo[1,self.snap_idx]=bump_hh_peak_xy[1]

        # track single cell rate evolution
        if track_cell_evo is True and self.snap_idx<num_snaps:
          self.cell_rr_evo[self.snap_idx]=self.rr[track_cell_idx,0]
          self.cell_hh_evo[self.snap_idx]=self.ff_input[track_cell_idx,0]
          self.cell_rec_input_evo[self.snap_idx]=self.rec_input[track_cell_idx,0]
          
          self.cell_rec_input_from_e_evo[self.snap_idx]=self.rec_input_from_e[track_cell_idx,0]
          self.cell_rec_input_from_i_evo[self.snap_idx]=self.rec_input_from_i[track_cell_idx,0]
        
        self.snap_idx+=1

    self.visits_map[self.visits_map==0]=1
    
    self.r_e_walk_map/=self.visits_map

    # logging simulation end
    self.endTime=datetime.datetime.fromtimestamp(time.time())
    self.endTimeStr=self.endTime.strftime('%Y-%m-%d %H:%M:%S')
    self.elapsedTime =time.time()-self.start_clock

    print 'Simulation ends: %s'%self.endTimeStr
    print 'Elapsed time: %s\n' %sl.format_elapsed_time(self.elapsedTime)
    


    
  
  def load_weights_from_data_path(self,weights_data_path):
    if os.path.exists(weights_data_path):
        print 'Loading recurrent weights: %s'%weights_data_path
        data=np.load(weights_data_path,allow_pickle=True)
        
        self.Wee_evo=data['Wee_evo']
        self.mean_rr_evo=data['mean_rr_evo']
        self.W_ee=data['W_ee']
        self.W_ee0=data['W_ee0']
        self.W_av_star=data['W_av_star']                
        self.W[:self.N_e,:self.N_e]=self.W_ee
          
        if 'conn_tuning_index' in data.keys():
          self.conn_tuning_index=data['conn_tuning_index']
          self.conn_trans_index=data['conn_trans_index']
          
          
    else:
      raise Exception('Data do not exist: %s'%weights_data_path)
      
    
#  def comp_inhibitory_tuning_index(self,verbose=False)  :
#    self.grid_tuning_out_inhib=gl.comp_grid_tuning_index(self.L,self.nx,(self.r[self.n_e**2:,:]).T,verbose=verbose)
    
  def comp_amplification_index(self):
    """
    Compute Acell ANoise and amplification index (measure based on power spectra in space)
    """
        
    self.grid_tuning_in=self.inputs.grid_tuning_in
    self.grid_tuning_out=gl.comp_grid_tuning_index(self.L,self.nx,(self.r[0:self.n_e**2,:]).T)    
    self.grid_tuning_out_inhib=gl.comp_grid_tuning_index(self.L,self.nx,(self.r[self.n_e**2:,:]).T)

    self.grid_amp_index=self.grid_tuning_out/self.grid_tuning_in
        
    
  def comp_output_spectra(self):

    """
    Compute output power spectra
    """
    assert(hasattr(self,'r'))
    
    self.nx=int(self.nx)
    
    r_mat=self.r.T.reshape(self.nx,self.nx,self.N)

    in_allfreqs = np.fft.fftshift(np.fft.fftfreq(self.nx,d=self.L/self.nx))
    
    self.freqs=in_allfreqs[self.nx/2:]
    
    r_dft_flat=np.fft.fftshift(np.fft.fft2(r_mat,axes=[0,1]),axes=[0,1])*(self.L/self.nx)**2

    r_pw=abs(r_dft_flat)**2    
    r_pw_profiles=gl.dft2d_profiles(r_pw)
    
    self.re_pw_profile=np.mean(r_pw_profiles,axis=0)
    self.he_pw_profile=self.inputs.in_mean_pw_profile
    
    
  def plot_recurrent_connectivity(self):
            
    pp.plot_recurrent_weights(self.W_ee,self.gp,vmax=self.W_ee.max())
    tuning_index= gl.get_recurrent_matrix_tuning_index(self.W_ee,self.gp)
    
    tot_in_weight=self.W_ee.sum(axis=1).mean()
    
    print 'Total input weight %.3f'%tot_in_weight 
    print 'Maximal weight %.3f'%self.W_ee.max()
    print 'Connnectivity tuning index: %.3f'%tuning_index


  def plot_recurrent_dynamics(self,cell_idx=0,snap_idxs=[1,5,10,5,19]):

    pl.rc('font',size=14)
    
    time=np.arange(0,self.recdyn_time,self.dt)
    

    pl.figure(figsize=(10,5))
    pl.subplots_adjust(bottom=0.2,hspace=0.4)
    
    pl.subplot(211)
    pl.plot(time,self.r_mean_vect[cell_idx,:],lw=2,label='Output')
    pl.plot(time,self.rec_input_mean_vect[cell_idx,:],lw=2,label='Rec. Input')
    pp.custom_axes()
    pl.ylabel('Mean rate [spike/s]')
    pl.legend()
    
    pl.subplot(212)
    pl.plot(time,self.r_max_vect[cell_idx,:],lw=2,label='Output')
    pl.plot(time,self.rec_input_max_vect[cell_idx,:],lw=2,label='Rec. Input')
    pp.custom_axes()
    pl.ylabel('Max rate [spike/s]')
    pl.xlabel('Time [s]')
    
    
    pl.figure(figsize=(12,5))
    pl.subplots_adjust(bottom=0.2,hspace=0.4)
    
    idx=1
    for snap_idx in snap_idxs:
      pl.subplot(2,len(snap_idxs),idx,aspect='equal')
      r_map=self.r_evo[cell_idx,:,snap_idx].reshape(self.nx,self.nx).T
      pl.pcolormesh(r_map,vmin=0,rasterized=True)
      pl.title('%.2f s'%(snap_idx*(self.recdyn_time/self.recdyn_num_snaps)),fontsize=12)
      pp.noframe()
      
      if snap_idx==snap_idxs[0]:
        pl.ylabel('Output')
        
      idx+=1
      
    for snap_idx in snap_idxs:
      pl.subplot(2,len(snap_idxs),idx,aspect='equal')
      r_map=self.rec_input_evo[cell_idx,:,snap_idx].reshape(self.nx,self.nx).T
      pl.pcolormesh(r_map,vmin=0,rasterized=True)
      pp.noframe()
      
      if snap_idx==snap_idxs[0]:
        pl.ylabel('Rec. Input')

      idx+=1
      
  def plot_steady_scores(self):
    import pylab as pl
    import plotlib as pp
    
    pl.figure(figsize=(3.5,2.8))
    pl.subplots_adjust(left=0.25,bottom=0.25)
    pl.hist(self.he_scores,bins=40,range=[-0.5,2],color='gray',histtype='stepfilled',weights=np.ones_like(self.he_scores)/float(len(self.he_scores)),alpha=1)
    pl.hist(self.re_scores,bins=40,range=[-0.5,2],color='black',histtype='stepfilled',weights=np.ones_like(self.re_scores)/float(len(self.re_scores)),alpha=1)
    pl.hist(self.he_scores,bins=40,range=[-0.5,2],color='gray',histtype='step',weights=np.ones_like(self.he_scores)/float(len(self.he_scores)),alpha=1,lw=2)
    
    pp.custom_axes()
    pl.ylim(1e-3,0.7)
    
    ax=pl.gca()
    ax.set_yscale('log')
    
    pl.xlabel('Gridness score')
    pl.ylabel('Fraction of cells')
    
    print 'Mean input gridness score: %.2f'%np.mean(self.he_scores)
    print 'Mean output gridness score: %.2f'%np.mean(self.re_scores)
    pl.title('%.2f'%np.mean(self.re_scores))      

class RecAmp2PopLearn(__RecAmp2Pop):
  """
  This class inherits from __RecAmp2Pop and implements the learning of the recurrent excitatory connections.
  The results of these simulations are saved in the results subfolder 'recamp_2pop' and have prefix 'RecAmp2PopLearn_'.
  """
  
  results_path=os.path.join(sl.get_results_path(),'recamp_2pop')

  
  def __init__(self,paramMap):
    
        # set parameter values from input map
    for param,value in paramMap.items():
      setattr(self,param,value)
      
    self.paramMap=paramMap         
    
    self.str_id=sl.gen_string_id(self.paramMap)
    self.hash_id=sl.gen_hash_id(self.str_id)
    
    self.data_path=os.path.join(RecAmp2PopLearn.results_path,'RecAmp2PopLearn_'+self.hash_id+'_data.npz')    
    self.params_path=os.path.join(RecAmp2PopLearn.results_path,'RecAmp2PopLearn_'+self.hash_id+'_log.txt')
    
    
  def save_learned_recurrent_weights(self):    
    
    # save variables
    toSaveMap={'paramMap':self.paramMap,
    'Wee_evo':self.Wee_evo,'mean_rr_evo':self.mean_rr_evo,
    'W':self.W,'W_ee':self.W_ee,'W_ee0':self.W_ee0,'W_av_star':self.W_av_star,
    'conn_tuning_index':self.conn_tuning_index,'conn_trans_index':self.conn_trans_index}
                     
    # save
    sl.ensureParentDir(self.data_path)
    np.savez(self.data_path,**toSaveMap)
    print 'Recurrent weights saved in: %s\n'%self.data_path

    
  def load_learned_recurrent_weights(self):
    """
    """
  
    self.load_weights_from_data_path(self.data_path)


    
  def learn_recurrent_weights(self,force=False):
    
    if force or not os.path.exists(self.data_path):
             
      self.post_init()      
      self.run_recurrent_dynamics_with_walk(self.learn_walk_time,
                                            self.learn_num_snaps,
                                            self.theta_sigma,
                                            learn_recurrent_weights=True,
                                            use_recurrent_input=self.learn_with_recurrent_input)
    
      
      self.gp=gl.GridProps(self.n_e,self.grid_T,self.grid_angle)
      self.conn_tuning_index= gl.get_recurrent_matrix_tuning_index(self.W_ee,self.gp)
      self.conn_trans_index=gl.get_trans_index(self.W_ee)
      
      self.save_learned_recurrent_weights()
      
    else:
      print 'Data already present: %s'%self.data_path    
    

class RecAmp2PopSteady(__RecAmp2Pop):
  """
  This class inherits from __RecAmp2Pop and implements the simulation of the steady-state outputs.
  The results of these simulations are saved in the results subfolder 'recamp_2pop' and have prefix 'RecAmp2PopSteady_'.
  """

  results_path=os.path.join(sl.get_results_path(),'recamp_2pop')


  def __init__(self,paramMap):
    
    # set parameter values from input map
    for param,value in paramMap.items():
      setattr(self,param,value)
      
    self.paramMap=paramMap         
    
    self.str_id=sl.gen_string_id(self.paramMap)
    self.hash_id=sl.gen_hash_id(self.str_id)
    
    self.data_path=os.path.join(RecAmp2PopSteady.results_path,'RecAmp2PopSteady_'+self.hash_id+'_data.npz')    
    self.paramsPath=os.path.join(RecAmp2PopSteady.results_path,'RecAmp2PopSteady_'+self.hash_id+'_log.txt')
    

  def save_steady_outputs(self):
    

    
    # variables to be saved
    toSaveMap={'paramMap':self.paramMap,
               'r':self.r,                   # steady state output patterns 
               
               'grid_tuning_in':self.grid_tuning_in,
               'grid_tuning_out':self.grid_tuning_out,             
               'grid_tuning_out_inhib':self.grid_tuning_out_inhib,             
               'grid_amp_index':self.grid_amp_index
            
               }   
                 
    # save      
    sl.ensureParentDir(self.data_path)
    np.savez(self.data_path,**toSaveMap)
    print 'Steady-state ouput saved in: %s\n'%self.data_path
    
      
  def load_steady_outputs(self):

    if os.path.exists(self.data_path):
        print 'Loading steady state output: %s'%self.data_path
        data=np.load(self.data_path,allow_pickle=True)
        self.r=data['r']
        
        self.grid_tuning_in=data['grid_tuning_in']
        self.grid_tuning_out=data['grid_tuning_out']        
        self.grid_amp_index=data['grid_amp_index']
        
        if 'grid_tuning_out_inhib' in data.keys():
          self.grid_tuning_out_inhib=data['grid_tuning_out_inhib']
                         
    else:
      print 'Steady state output does not exist: %s'%self.data_path
        

  def recompute_and_save_amplification_index(self):    
      self.post_init()    
      self.load_steady_outputs()      
      self.comp_amplification_index()
      self.save_steady_outputs()

  
  def compute_and_save_steady_output(self,force=False):
    """
    Computes steady-state output of the network and saves the results to disk

    Parameters
    ----------
    force : boolean, optional
      Force the recomputation of the outputs even if they are already present on disk.
      The default is False.

    Returns
    -------
    None. 

    """
    
    if force or not os.path.exists(self.data_path):

            
      self.post_init()      
      
      # we need to load already learned weights
      if self.use_learned_recurrent_weights is True:
        assert(self.recurrent_weights_path is not None)
        self.load_weights_from_data_path(self.recurrent_weights_path)
        
      self.run_recurrent_dynamics(record_mean_max=False)
      self.comp_amplification_index()      
      self.save_steady_outputs()
      
    else:
      print 'Data already present: %s'%self.data_path
      


  

  def plot_example_outputs(self,cell_idxs=[1,2,3,4,5]):
    
        
    import pylab as pl
    import grid_utils.plotlib as pp
      
    dx=self.L/self.nx
    xran=np.arange(self.nx)*self.L/self.nx-self.L/2-dx/2.
       
    vmax=None
      
    
    pl.figure(figsize=(10,3))
    pl.subplots_adjust(wspace=0.2,top=0.8)
    for idx,cell_idx in enumerate(cell_idxs):
      pl.subplot(1,5,idx+1,aspect='equal')
      r_map=self.r[cell_idx,:].reshape(self.nx,self.nx).T
  
      
    
      pl.pcolormesh(xran,xran,r_map,rasterized=True,vmin=0,vmax=vmax)   
      pl.title('%.1f  %.2f'%(r_map.max(),self.grid_tuning_out[cell_idx]),fontsize=11)
      
  
      pp.noframe()
  
class RecAmp2PopAttractor(__RecAmp2Pop):
  """
  This class inherits from __RecAmp2Pop and implements the simulations to probe attractor dynamics
  The results of these simulations are saved in the results subfolder 'recamp_2pop' and have prefix 'RecAmp2PopSteady_'.
  """

  results_path=os.path.join(sl.get_results_path(),'recamp_2pop')


  def __init__(self,paramMap):
    #self.as_super = super(RecAmp2PopAttractor, self)
    
    # set parameter values from input map
    for param,value in paramMap.items():
      setattr(self,param,value)
      
    self.paramMap=paramMap         
    
    self.str_id=sl.gen_string_id(self.paramMap)
    self.hash_id=sl.gen_hash_id(self.str_id)
    
    self.data_path=os.path.join(RecAmp2PopSteady.results_path,'RecAmp2PopAttractor_'+self.hash_id+'_data.npz')    
    self.paramsPath=os.path.join(RecAmp2PopSteady.results_path,'RecAmp2PopAttractor_'+self.hash_id+'_log.txt')
    
  # def post_init(self):
  #   #return super(RecAmp2PopAttractor, self).post_init(get_inputs=False)    
  #   return self.as_super.post_init(get_inputs=False)    

  def get_stimulus_bump(self):
    """
    Generates an artificial activity bumps in phase space to be presented as feed-forward input to the network.
    Such a feed-forward input stimulus is used to probe the attractor landscape of the newtwork, i.e.
    does the network follow the imposed bump location? Where does the bump move when such input is removed?
    
    We produce N_e such bumps, where N_e is the total number of excitatory neurons in the network.
    Each bump is centered at the preferred phase of each excitatory neuron.
    
    Parameters
    ----------
    r_max : float, optional
      Peak rate of the bump. The default is 10.
    sigma : float, optional
      With of the bump. The default is 0.2.

    Returns
    -------
    None. But it stores in the object the fields following fields:
      stimulus_bumps: 
    """
   
    re_stimulus_bumps=np.zeros((self.N_e,self.N_e))
    
    # loop across the preferred phases of all neurons
    for idx,phase_shift in enumerate(self.gp.phases[:,:]):
      
      # initialize bump activity to zero
      bump=np.zeros(self.N_e)
      
      # To account for the periodicity of the phase space we need to add up 9 bumps that are shifted
      # of +- one period in each dimension (i.e., 3 horizontal shifts + 3 vertiacl shifts)
      for xp in (-1,0,1):
        for yp in (-1,0,1):
          shift=xp*self.gp.u1+yp*self.gp.u2
          bump+=rhombus_mises(self.gp.phases-phase_shift[np.newaxis,:]+shift,self.stimulus_bump_sigma)
            
      # save the generated bump for the current preferred phase      
      re_stimulus_bumps[idx,:]=bump/bump.mean()*self.input_mean

    self.stimulus_bumps=np.zeros((self.N_e+self.N_i,self.N_e))
    self.stimulus_bumps[:self.N_e,:]=re_stimulus_bumps
        
    
  def run_single_attractor_sim(self,init_r,time,stimulus_on):
    """
    Runs the recurrent dynamics in two different scenarios:
      1. With an external stimulus (stimuls_on=True) starting from a zero intial condition
      2. Without an external stimuls (stimulus_on=False) starting from a non-zero initial condition
    In case 1. the feed-forward input is generated using the method `get_stimulus_bump`. And the rates 
    of all neurons are initialized at zero
    
    In case 2. the feed-forward input is off (set to a constant value ) and the neurons are initialized
    with the rates provided by the input argmunet `init_r`.

    Parameters
    ----------
    init_r : Numpy Array of dimensions: N_e+N_i
      Initialization values for the rates of all neurons
      
    time : float
      Time length of the simulation
    stimulus_on : bool
      Indicates if an external feed-forward input needs to be provided.
      If False we require to initialize the rates to non zero-values

    Returns
    -------
    out_bumps: Numpy array of dimensions: N_e x N_e
      Rates of the excitatory neurons for all the N_e shifts of the stimulus bump
      
    out_bump_idsx2d: Numpy a of dimensions: 2 x N_e
      Coordinates of the output bump of all the N_e shifts of the similus bump
    """
  
    # if we need a feed-forward stimulus we need to generate it first
    if stimulus_on is True:
      self.get_stimulus_bump()
      assert(init_r is None)
      
    # otherwise we should make sure that the rates have been previously intialized with a stimulus  
    else:
      assert(init_r is not None)
      
    # set the simulation time to the required value
    self.recdyn_time=time
    
    # run recurrent dynamics with stimulus on
    if stimulus_on :
      print('Running with stimulus on for %.1f s'%time)
      self.run_recurrent_dynamics(
      record_mean_max=False,
      custom_h=self.stimulus_bumps,
      init_r=np.zeros_like(self.stimulus_bumps),
      )

    # run recurrent dynamics with stimulus off (stimulus was presented previously)      
    else:
      print('Running with stimulus off for %.1f s'%time)
      self.run_recurrent_dynamics(
        record_mean_max=False,
        custom_h=np.ones_like(init_r)*self.flat_h_rate,
        init_r=init_r,
        )
      
    # save the output rates at the and of the simulation  
    out_bumps=self.r

    # get bump idxs at the end of the simulation    
    out_bump_idx2d=get_bumps_idxs_all_shifts(self.r,self.N_e)
        
    out_bump_evo_idx2d=[]
    for snap_idx in range(self.recdyn_num_snaps):
      out_bump_evo_idx2d.append(get_bumps_idxs_all_shifts(np.squeeze(self.r_evo[:,:,snap_idx]),self.N_e)) 
    out_bump_evo_idx2d=np.array(out_bump_evo_idx2d)
      
    return out_bumps,out_bump_idx2d,out_bump_evo_idx2d

    

  def run_attractor_sims(self):
    
    self.out_bumps_stim_on,self.out_bumps_stim_on_idx2d,self.out_bumps_stim_on_evo_idx2d=self.run_single_attractor_sim(None,self.time_stimulus_on,True)
    self.out_bumps_stim_off,self.out_bumps_stim_off_idx2d,self.out_bumps_stim_off_evo_idx2d=self.run_single_attractor_sim(self.out_bumps_stim_on,self.time_stimulus_off,False)


  def save_attractor_outputs(self):
    
    
    # variables to be saved
    toSaveMap={'paramMap':self.paramMap,
               'out_bumps_stim_on':self.out_bumps_stim_on,
               'out_bumps_stim_on_idx2d':self.out_bumps_stim_on_idx2d,
               'out_bumps_stim_on_evo_idx2d':self.out_bumps_stim_on_evo_idx2d,
               
               'out_bumps_stim_off':self.out_bumps_stim_off,
               'out_bumps_stim_off_idx2d':self.out_bumps_stim_off_idx2d,
               'out_bumps_stim_off_evo_idx2d':self.out_bumps_stim_off_evo_idx2d,
                              
               'field_len_stim_on':self.field_len_stim_on,
               'field_len_stim_off':self.field_len_stim_off,
               'field_len_stim_on_evo':self.field_len_stim_on_evo,
               'field_len_stim_off_evo':self.field_len_stim_off_evo,         
                              
               'stim_off_speed_evo':self.stim_off_speed_evo,
               'stim_off_avg_last_second_speed':self.stim_off_avg_last_second_speed,
               'stim_off_avg_path_dist_evo':self.stim_off_avg_path_dist_evo,
    
               'stim_off_end_dists':self.stim_off_end_dists,
               'stim_off_avg_end_dist':self.stim_off_avg_end_dist,
                
               'all_end_phases':self.all_end_phases,
               'num_attractors':self.num_attractors,
                }
    
                 
    # save      
    sl.ensureParentDir(self.data_path)
    np.savez(self.data_path,**toSaveMap)
    print 'Attractor ouput saved in: %s\n'%self.data_path
    
      
  def load_attractor_outputs(self):

    if os.path.exists(self.data_path):
        print 'Loading attractor output: %s'%self.data_path
        data=np.load(self.data_path,allow_pickle=True)
        
        self.out_bumps_stim_on=data['out_bumps_stim_on']
        self.out_bumps_stim_on_idx2d=data['out_bumps_stim_on_idx2d']
        self.out_bumps_stim_on_evo_idx2d=data['out_bumps_stim_on_evo_idx2d']
                
        self.out_bumps_stim_off=data['out_bumps_stim_off']
        self.out_bumps_stim_off_idx2d=data['out_bumps_stim_off_idx2d']
        self.out_bumps_stim_off_evo_idx2d=data['out_bumps_stim_off_evo_idx2d']
        
        self.field_len_stim_on=data['field_len_stim_on']
        self.field_len_stim_off=data['field_len_stim_off']
                
        self.field_len_stim_on_evo=data['field_len_stim_on_evo']
        self.field_len_stim_off_evo=data['field_len_stim_off_evo']
        
        self.stim_off_speed_evo=data['stim_off_speed_evo']
        self.stim_off_avg_last_second_speed=data['stim_off_avg_last_second_speed']
        self.stim_off_avg_path_dist_evo=data['stim_off_avg_path_dist_evo']
    
        self.stim_off_end_dists=data['stim_off_end_dists']
        self.stim_off_avg_end_dist=data['stim_off_avg_end_dist']
                
        self.all_end_phases=data['all_end_phases']
        self.num_attractors=data['num_attractors']
        
                         
    else:
      print 'Attractor output does not exist: %s'%self.data_path
      

  def get_attractor_fields(self):
    
    self.X_in,self.Y_in=[C/float(self.n_e) for C in np.mgrid[0:self.n_e,0:self.n_e]]
    self.X_out_stim_on=self.out_bumps_stim_on_idx2d[0,:].reshape(self.n_e,self.n_e)/float(self.n_e)
    self.Y_out_stim_on=self.out_bumps_stim_on_idx2d[1,:].reshape(self.n_e,self.n_e)/float(self.n_e)

    self.X_out_stim_off=self.out_bumps_stim_off_idx2d[0,:].reshape(self.n_e,self.n_e)/float(self.n_e)
    self.Y_out_stim_off=self.out_bumps_stim_off_idx2d[1,:].reshape(self.n_e,self.n_e)/float(self.n_e)
    
    self.dX_stim_on,self.dY_stim_on,self.field_len_stim_on=gl.get_vector_field(self.X_in,self.Y_in
                                                                               ,self.X_out_stim_on,self.Y_out_stim_on)
    
    self.dX_stim_off,self.dY_stim_off,self.field_len_stim_off=gl.get_vector_field(self.X_in,self.Y_in
                                                                               ,self.X_out_stim_off,self.Y_out_stim_off)
    
    self.field_len_stim_off_evo=np.zeros(self.recdyn_num_snaps)
    self.field_len_stim_on_evo=np.zeros(self.recdyn_num_snaps)
    
    for snap_idx in range(self.recdyn_num_snaps):
      curr_X_out_stim_off=self.out_bumps_stim_off_evo_idx2d[snap_idx,0,:].reshape(self.n_e,self.n_e)/float(self.n_e)
      curr_Y_out_stim_off=self.out_bumps_stim_off_evo_idx2d[snap_idx,1,:].reshape(self.n_e,self.n_e)/float(self.n_e)

      curr_X_out_stim_on=self.out_bumps_stim_on_evo_idx2d[snap_idx,0,:].reshape(self.n_e,self.n_e)/float(self.n_e)
      curr_Y_out_stim_on=self.out_bumps_stim_on_evo_idx2d[snap_idx,1,:].reshape(self.n_e,self.n_e)/float(self.n_e)
      
      _,_,field_len_stim_off=gl.get_vector_field(self.X_in,self.Y_in,curr_X_out_stim_off,curr_Y_out_stim_off)
      _,_,field_len_stim_on=gl.get_vector_field(self.X_in,self.Y_in,curr_X_out_stim_on,curr_Y_out_stim_on)
      
      self.field_len_stim_off_evo[snap_idx]=field_len_stim_off
      self.field_len_stim_on_evo[snap_idx]=field_len_stim_on
  
  
  def run_and_save_attractor(self,force=False):
    if force or not os.path.exists(self.data_path):

      self.post_init()      
      
      # we need to load already learned weights
      if self.use_learned_recurrent_weights is True:
        assert(self.recurrent_weights_path is not None)
        self.load_weights_from_data_path(self.recurrent_weights_path)
        if hasattr(self,'W_ee_scale_factor') and not self.W_ee_scale_factor ==1:
          self.W_ee*=self.W_ee_scale_factor
          self.W[:self.N_e,:self.N_e]=self.W_ee
      else:
        if hasattr(self,'smooth_conn_bump') and self.smooth_conn_bump is True:
          self.get_tuned_excitatory_weights(binary=False)
        else:
          self.get_tuned_excitatory_weights(binary=True)

      self.run_attractor_sims()
      self.get_attractor_fields()
      self.get_bump_speed()
      self.get_final_attractors()
      self.save_attractor_outputs()      
      
    else:
      print 'Data already present: %s'%self.data_path
    

  def get_detected_phases(self,idx2d,time_idx):
    """
    Converts detected bump-peak indexes in phases
    idx2d: indexes to convert
    time_idx: time index
    """

    det_phases=np.zeros_like(self.gp.phases)
    for idx in range(len(self.gp.phases)):
      if time_idx is None:
        indexes=idx2d[:,idx].astype(int)
      else:
        indexes=idx2d[time_idx,:,idx].astype(int)
      det_phase_idx=np.ravel_multi_index(indexes, (self.n_e,self.n_e))
      det_phases[idx,:]=self.gp.phases[det_phase_idx,:]
    return det_phases
  
  
  def get_bump_speed(self):
    """
    Compute the distance travelled and the speed of the bump in the untuned condition

    Returns
    -------
    None.

    """
    
    old_det_phases=self.get_detected_phases(self.out_bumps_stim_off_evo_idx2d,0)
  
    num_steps=int(self.time_stimulus_off/self.dt)        
    delta_snap=num_steps/self.recdyn_num_snaps
    snap_time=delta_snap*self.dt  
    one_second_idx=int(1./snap_time)
    assert(one_second_idx>0)
    #print one_second_idx
    
    stim_off_dists_evo=np.zeros((self.recdyn_num_snaps,self.N_e))
    
    for time_idx in range(self.recdyn_num_snaps):
      det_phases=self.get_detected_phases(self.out_bumps_stim_off_evo_idx2d,time_idx)
      dists=gl.get_periodic_dist_on_rhombus(self.n_e,det_phases,old_det_phases,self.gp.u1,self.gp.u2)
      stim_off_dists_evo[time_idx,:]=dists
      old_det_phases=det_phases
    
    self.stim_off_speed_evo=stim_off_dists_evo/snap_time
    self.stim_off_avg_last_second_speed=self.stim_off_speed_evo[-one_second_idx:,:].mean(axis=0).mean()
    self.stim_off_avg_path_dist_evo=np.cumsum(stim_off_dists_evo.mean(axis=1))
    
    
   
  def get_final_attractors(self):
    start_det_phases=self.get_detected_phases(self.out_bumps_stim_off_evo_idx2d,0)
    end_phases=self.get_detected_phases(self.out_bumps_stim_off_idx2d,None)
    
    self.stim_off_end_dists=gl.get_periodic_dist_on_rhombus(self.n_e,end_phases,start_det_phases,self.gp.u1,self.gp.u2)
    self.stim_off_avg_end_dist=self.stim_off_end_dists.mean()
    
    # get unique end phases
    self.all_end_phases=set([])
    for phase in end_phases:
      self.all_end_phases.add(tuple(phase))
    self.num_attractors=len(self.all_end_phases)
    
    

  def plot_landscape(self,stim_on,time_idx=None,quiver=True,rhombus=True,plot=True,attractor_plots=False,rhombus_color='k',rhombus_lw=1):
    """
    Plots the energy landscape of the attractor network
    stim_on: bool, stimulus on or off
    time_idx: time index
    """
    import matplotlib
    
    if time_idx is None:
      if stim_on:
        idxs2d=self.out_bumps_stim_on_idx2d
        dX,dY=self.dX_stim_on,self.dY_stim_on
        field_len=self.field_len_stim_on

      else:
        idxs2d=self.out_bumps_stim_off_idx2d
        dX,dY=self.dX_stim_off,self.dY_stim_off
        field_len=self.field_len_stim_off
    else:
      if stim_on:
        idxs2d=self.out_bumps_stim_on_evo_idx2d[time_idx,:,:]
      else:
        idxs2d=self.out_bumps_stim_off_evo_idx2d[time_idx,:,:]

      X_out=idxs2d[0,:].reshape(self.n_e,self.n_e)/float(self.n_e)
      Y_out=idxs2d[1,:].reshape(self.n_e,self.n_e)/float(self.n_e)
      dX,dY,field_len=gl.get_vector_field(self.X_in,self.Y_in,X_out,Y_out)

    pl.gca().set_aspect('equal', adjustable='box')

    H, xedges, yedges = np.histogram2d(idxs2d[0,:], idxs2d[1,:], bins=(range(self.n_e+1), range(self.n_e+1)))

    end_phases=self.get_detected_phases(idxs2d,None)
    Q=None

    if plot is True:
      if rhombus is False:
        if quiver is False:
          img=pl.pcolormesh(self.Y_in,self.X_in,H,norm=matplotlib.colors.LogNorm(vmin=1,vmax=100),cmap='viridis_r')
          pp.colorbar(fixed_ticks=[1,10,100])
        else:
          img=pl.pcolormesh(self.Y_in,self.X_in,H,norm=matplotlib.colors.LogNorm(vmin=1,vmax=100),cmap='viridis_r')
          pp.colorbar(fixed_ticks=[1,10,100])
          Q=pl.quiver(self.Y_in+.5/self.n_e,self.X_in+.5/self.n_e,dY,dX,headwidth=14)

        pl.xlim(0,1)
        pl.ylim(0,1)

        pp.custom_axes()
      else:
        if attractor_plots:
          dists=np.sqrt(dX**2+dY**2).ravel()
          dists/=dists.max()
          img=pp.plot_on_rhombus(self.gp.R_T,1,0, self.N_e,self.gp.phases,
                                     dists,plot_axes=False,plot_rhombus=True,
                                     plot_cbar=False,cmap='Purples',rhombus_color=rhombus_color,rhombus_lw=rhombus_lw)     
          
          # get unique end phases
          all_end_phases=set([])
          for phase in end_phases:
            all_end_phases.add(tuple(phase))
          
          # plot only the unique end phases
          for phase in all_end_phases:
            pl.plot(phase[0],phase[1],'.k',ms=8)#,mec='tab:orange',mfc='tab:orange')
            
        else:
          img=pp.plot_on_rhombus(self.gp.R_T,1,0, self.N_e,self.gp.phases,
                                H.ravel(),plot_axes=False,plot_rhombus=True,
                                plot_cbar=False,norm=matplotlib.colors.LogNorm(vmin=1,vmax=100),cmap='viridis_r')
          if quiver:
            Q=pl.quiver(self.gp.phases[:,1],self.gp.phases[:,0],dY,dX,pivot='mid',headwidth=10)


      pl.xlabel('Bump location (x)')
      pl.ylabel('Bump location (y)')

    if plot is True:
      if attractor_plots:
        return img
      elif Q is not None:
        return img,Q    
    else:
      return H,dY,dX if quiver else H
    
    