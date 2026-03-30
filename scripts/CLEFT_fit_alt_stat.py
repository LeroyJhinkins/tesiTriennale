#!/usr/bin/python

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"]= "1"
os.environ["OPENBLAS_MAIN_FREE"]= "1"

import numpy as np
from numpy.linalg import inv
import sys
import time
sys.path.append("libs/comet-emu")
sys.path.append("libs/comet-emu/comet")
from PTEmu_LPT_new import PTEmu as PTEmu_LPT # type: ignore
from scipy.stats import norm
from datetime import datetime

from nautilus import Sampler, Prior

#------------------------------
# Pre-processing functions
#-----------------------------

class run_chain():
    def __init__(self, what_model, smin, de_model='lambda', what_stat='multipoles', mu_edges=[0., 0.5], pi_max=60., multi_bins=[1], what_cosmo='ELM', what_data='measured'):
        self.smin = smin
        self.de_model = de_model

        self.what_cosmo = what_cosmo
        # Defining the range for which we want to fit the correlation function
        self.smax = 200
        # This is for defining how many bins we have in xi
        self.ds = 5
        self.ns = int((self.smax-self.smin)/self.ds)

        self.pool = 3

        # This determines how many redshifts should be fitted simulateneously
        self.z_mean_array = [1.0, 1.2, 1.4, 1.65]
        self.multi_bins = multi_bins
        self.multi_z = [self.z_mean_array[z-1] for z in self.multi_bins]
        self.num_z = len(self.multi_z)
        self.multi_z_string = "_" + "_".join(str(num) for num in self.multi_z)

        self.what_model = what_model
        self.what_stat = what_stat
        self.what_data = what_data
        if self.what_stat == 'wedges':
            self.mu_edges = mu_edges
        elif self.what_stat == 'proj':
            self.pi_max = pi_max

        # Folder structure
        if self.de_model == None:
            self.output_dir = f'chains/{self.what_model}_template/'
        else:
            self.output_dir = f'chains/{self.what_model}_{self.de_model}/'
        self.cov_dir = [f'outData/z{num}_rebin/' for num in self.multi_bins]
        self.data_dir = [f'outData/z{num}_rebin/' for num in self.multi_bins]
        self.log_dir = 'logs/'

        self.full_set_up()

        # The filename of the chain
        self.modefit = f'{self.output_dir}chain_{self.what_model}-z{self.multi_z_string}-smin{self.smin}-de_model_{self.de_model}-cov_it_2-what_stat_{self.what_stat}'

    def full_set_up(self):
        print('Setting up data and covarince...', end='')
        self.set_up_data_cov()
        print('Done')

        print('Setting up emulator...', end='')
        self.set_up_EMU()
        print('Done')

        print('Setting up sampler...', end='')
        self.set_up_sampler()
        print('Done')

    def set_up_data_cov(self):
        

        if self.what_stat == 'multipoles':
            self.icov = np.zeros((3*self.ns, 3*self.ns, self.num_z))
            self.data_to_fit = np.zeros((3*self.ns, self.num_z))
        elif self.what_stat == 'wedges':
            how_many_wedges = len(self.mu_edges) - 1
            self.icov = np.zeros((how_many_wedges*self.ns, how_many_wedges*self.ns, self.num_z))
            self.data_to_fit = np.zeros((how_many_wedges*self.ns, self.num_z))

        for counter in range(self.num_z):
            #file_data = f'{self.data_dir}AbacusSummit_base_c000_multipoles_correlation_function_galaxies_z{self.multi_z[counter]}_axis_2_AM_asym_HOD_mean_rebinned_{self.ds}hMpc.txt'
            #file_covar = f'{self.cov_dir}covariance_z_{self.multi_z[counter]}_v2_Deltar_{self.ds}hMpc.npy'
            file_data = f'{self.data_dir[counter]}mean_{self.what_stat}_{self.what_data}.npy'
            file_covar = f'{self.cov_dir[counter]}cov_{self.what_stat}_{self.what_data}.npy'
            self.s, data_proxy =  self._data_cutting(file_data, mode='ELM') # s will be the same, regardless of redshift
            self.icov[:,:,counter] = self._cov_cutting_abacus(file_covar, mode='ELM')
            self.data_to_fit[:,counter] = data_proxy


    def set_up_EMU(self):
        self.CModel = PTEmu_LPT(model=self.what_model, use_Mpc=False)

        if self.what_cosmo == "Abacus":
            self.fid_cosmo = {'wc':0.1200*np.ones(self.num_z) ,'wb':0.02237*np.ones(self.num_z) ,'ns':0.9649*np.ones(self.num_z), 'h':0.6736*np.ones(self.num_z), 'As':2.0830*np.ones(self.num_z), 'w0':-1.*np.ones(self.num_z), 'wa':0.*np.ones(self.num_z), 'z':self.multi_z}
        elif self.what_cosmo == "ELM":
            self.fid_cosmo = {'wc':(0.32-0.049)*0.67**2*np.ones(self.num_z) ,'wb':0.049*0.67**2*np.ones(self.num_z) ,'ns':0.96*np.ones(self.num_z), 'h':0.67*np.ones(self.num_z), 'As':2.11065*np.ones(self.num_z), 'w0':-1.*np.ones(self.num_z), 'wa':0.*np.ones(self.num_z), 'z':self.multi_z}

        self.CModel.define_fiducial_cosmology(params_fid = self.fid_cosmo)

    def set_up_sampler(self):
        self.prior = Prior()

        self.n_live = 3000 # This needs to be changed to 5000

        if self.de_model != None:
            self.prior.add_parameter(f'h', dist=(0.6, 0.8))
            self.prior.add_parameter(f'As', dist=(1.2, 3.))
            self.prior.add_parameter(f'wc', dist=(0.085, 0.155))

        for counter, z in enumerate(self.multi_z):
            if self.de_model == None:
                self.prior.add_parameter(f'f_z{counter+1}', dist=(0.5, 1.05))
                self.prior.add_parameter(f's12_z{counter+1}', dist=(0.2, 1.0))
                self.prior.add_parameter(f'q_lo_z{counter+1}', dist=(0.9, 1.1))
                self.prior.add_parameter(f'q_tr_z{counter+1}', dist=(0.9, 1.1))

            self.prior.add_parameter(f'b1_z{counter+1}', dist=(-0.5, 2.5))
            if self.what_model == 'CLEFT':
                self.prior.add_parameter(f'b2_z{counter+1}', dist=(-10.0, 10.0))
                self.prior.add_parameter(f'bs_z{counter+1}', dist=(-20.0, 20.0))
                self.prior.add_parameter(f'a_xi_z{counter+1}', dist=(-100., 200.))
                self.prior.add_parameter(f'a_v_z{counter+1}', dist=(-100., 200.))
                self.prior.add_parameter(f'a_s_z{counter+1}', dist=(-100., 200.))
            elif self.what_model == 'CLPT':
                self.prior.add_parameter(f'b2_z{counter+1}', dist=(-10.0, 10.0))
                self.prior.add_parameter(f'sig_v_z{counter+1}', dist=(0., 100.0))
            elif self.what_model == 'ZA':
                self.prior.add_parameter(f'sig_v_z{counter+1}', dist=(0., 100.0))

        self.sampler = Sampler(self.prior, self.likelihood, n_live=self.n_live, pool=self.pool, vectorized=False)

    def run_sampler(self):
        start = time.perf_counter()
        self.sampler.run(verbose=True)
        end = time.perf_counter()
        self.total_time = end-start
        print(f"pool and n_live: {self.pool, self.n_live}")

        # Extract chain
        self.points, self.log_w, self.log_l = self.sampler.posterior() # type: ignore

    def store_and_log(self):
        chain_shape = np.shape(self.points) # type: ignore

        # Logging
        now = datetime.now()
        time_string = now.strftime("%Y-%m-%d %H:%M:%S")
        with open(f"{self.log_dir}logfile.txt", "a") as myfile:
            myfile.write(f"Time end={time_string}; path+name={self.modefit}; multi_z={self.multi_z}; pool={self.pool}; n_live={self.n_live}; final num_samp_points={chain_shape[0]}; num_fit_par={chain_shape[1]}; runtime in seconds={self.total_time}; runtime in hours={self.total_time/3600.}; CPUs per iteration={self.total_time*self.pool/chain_shape[0]}; what_stat={self.what_stat}\n")

        # Storing
        final_chain = np.concatenate((np.array([np.exp(self.log_w)]).T, np.array([self.log_l]).T, self.points), axis=1) # type: ignore
        print(np.shape(final_chain))
        np.savetxt(f"{self.modefit}_nlive{self.n_live}_pool_{self.pool}_newcode_{self.what_data}.txt", final_chain)

    def full_run(self):
        self.run_sampler()
        self.store_and_log()


    ############################################################################
    # Helper functions
    def _cov_cutting_abacus(self, cov_name, mode='ELM'):
        if self.ds == 5:
            if mode == 'ELM':
                s_raw = np.linspace(2.5, 197.5, 40) # These are the 5Mpc/h bins the Cov is in
            else:
                s_raw = np.linspace(7.5, 197.5, 39) # These are the 5Mpc/h bins the Cov is in
            if self.what_stat == 'multipoles':
                num_tot = 3 * len(s_raw[(s_raw>self.smin) & (s_raw<self.smax)])
            elif self.what_stat == 'wedges':
                num_tot = len(s_raw[(s_raw>self.smin) & (s_raw<self.smax)])
        elif self.ds == 4:
            if mode == 'ELM':
                raise Exception("This binning is not supported for ELM measurements")
            s_raw = np.linspace(6., 198., 49) # These are the 5Mpc/h bins the Cov is in
            num_tot = 3 * len(s_raw[(s_raw>self.smin) & (s_raw<self.smax)])
            if self.what_stat == 'multipoles':
                num_tot = 3 * len(s_raw[(s_raw>self.smin) & (s_raw<self.smax)])
            elif self.what_stat == 'wedges':
                num_tot = len(s_raw[(s_raw>self.smin) & (s_raw<self.smax)])
        else:
            raise Exception("No known binning")

        if 'txt' in cov_name:
            data_cov = np.loadtxt(cov_name)
        else:
            data_cov = np.load(cov_name)

        if self.what_stat == 'multipoles':
            s_3l = np.hstack((s_raw, s_raw, s_raw)) # There are three multipoles
            s_3l_2D_xx, s_3l_2D_yy = np.meshgrid(s_3l, s_3l, indexing='ij') # This gives us the two matrices containing the respective s_bins of the covariances
            cut_cov = data_cov[((s_3l_2D_xx > self.smin) & (s_3l_2D_xx < self.smax)) & ((s_3l_2D_yy > self.smin) & (s_3l_2D_yy < self.smax))].reshape(num_tot, num_tot) / 5.06617801 # Divide by 25 because we fit the mean of 25 realisations
            icov_ret = inv(cut_cov)
        elif self.what_stat == 'wedges':
            s_3l = s_raw # Currently supports only one wedge 
            s_3l_2D_xx, s_3l_2D_yy = np.meshgrid(s_3l, s_3l, indexing='ij') # This gives us the two matrices containing the respective s_bins of the covariances
            cut_cov = data_cov[((s_3l_2D_xx > self.smin) & (s_3l_2D_xx < self.smax)) & ((s_3l_2D_yy > self.smin) & (s_3l_2D_yy < self.smax))].reshape(num_tot, num_tot) / 5.06617801 # Divide by 1000. because we fit the mean of 1000. realisations

        icov_ret = inv(cut_cov)
        return icov_ret

    def _data_cutting(self, data_name, mode='ELM'):
        if 'txt' in data_name:
            data = np.loadtxt(data_name).T
        else:
            data = np.load(data_name).T

        if self.what_stat == 'multipoles':
            if mode == "ELM":
                s_ret = np.load(f"{self.data_dir[0]}s_unique_reb.npy")
                ns_tmp = len(s_ret)
                xi0 = data[:ns_tmp]
                xi2 = data[ns_tmp:2*ns_tmp]
                xi4 = data[2*ns_tmp:]
                xi0 = xi0[(s_ret>self.smin) & (s_ret<self.smax)]
                xi2 = xi2[(s_ret>self.smin) & (s_ret<self.smax)]
                xi4 = xi4[(s_ret>self.smin) & (s_ret<self.smax)]
                data_ret = np.array([xi0,xi2,xi4]).flatten()
                s_ret = s_ret[(s_ret>self.smin) & (s_ret<self.smax)]
            else:
                s_ret = data[0]
                xi0 = data[1][(s_ret>self.smin) & (s_ret<self.smax)]
                xi2 = data[2][(s_ret>self.smin) & (s_ret<self.smax)]
                xi4 = data[3][(s_ret>self.smin) & (s_ret<self.smax)]
                data_ret = np.array([xi0,xi2,xi4]).flatten()
                s_ret = s_ret[(s_ret>self.smin) & (s_ret<self.smax)]

        elif self.what_stat == 'wedges':
            if mode == "ELM":
                s_ret = np.load(f"{self.data_dir[0]}s_unique_reb.npy")
                ns_tmp = len(s_ret)
                xi0 = data[:ns_tmp]
                xi0 = xi0[(s_ret>self.smin) & (s_ret<self.smax)]
                data_ret = xi0
                s_ret = s_ret[(s_ret>self.smin) & (s_ret<self.smax)]


        return s_ret, data_ret

    ############################################################################
    # The likelihood-function 
    # The classic likelihood no AM, taking the outpute Pell as they are
    def likelihood(self, params_dict):

        theta_dict = {}

        # These parameters are always there
        theta_dict['z'] = np.array(self.multi_z)
        theta_dict['wb'] = self.fid_cosmo['wb']*np.ones(self.num_z)
        theta_dict['ns'] = self.fid_cosmo['ns']*np.ones(self.num_z)
        theta_dict['b1'] = np.array([params_dict[f'b1_z{i+1}'] for i in range(self.num_z)])

        if self.de_model == None:
            theta_dict['h'] = self.fid_cosmo['h']*np.ones(self.num_z)
            theta_dict['wc'] = self.fid_cosmo['wc']*np.ones(self.num_z)
            theta_dict['As'] = self.fid_cosmo['As']*np.ones(self.num_z)

            theta_dict['f'] = np.array([params_dict[f'f_z{i+1}'] for i in range(self.num_z)])
            theta_dict['s12'] = np.array([params_dict[f's12_z{i+1}'] for i in range(self.num_z)])
            theta_dict['q_lo'] = np.array([params_dict[f'q_lo_z{i+1}'] for i in range(self.num_z)])
            theta_dict['q_tr'] = np.array([params_dict[f'q_tr_z{i+1}'] for i in range(self.num_z)])
        else:
            theta_dict['h'] = params_dict[f'h'] * np.ones(self.num_z)
            theta_dict['As'] = params_dict[f'As'] * np.ones(self.num_z)
            theta_dict['wc'] = params_dict[f'wc'] * np.ones(self.num_z)

        if self.what_model == 'CLEFT':
            theta_dict['a_xi'] = np.array([params_dict[f'a_xi_z{i+1}'] for i in range(self.num_z)])
            theta_dict['a_v'] = np.array([params_dict[f'a_v_z{i+1}'] for i in range(self.num_z)])
            theta_dict['a_s'] = np.array([params_dict[f'a_s_z{i+1}'] for i in range(self.num_z)])
            theta_dict['b2'] = np.array([params_dict[f'b2_z{i+1}'] for i in range(self.num_z)])
            theta_dict['bs'] = np.array([params_dict[f'bs_z{i+1}'] for i in range(self.num_z)])
            theta_dict['sig_v'] = np.zeros(self.num_z)
        elif self.what_model == 'CLPT':
            theta_dict['bs'] = np.zeros(self.num_z)
            theta_dict['a_xi'] = np.zeros(self.num_z)
            theta_dict['a_v'] = np.zeros(self.num_z)
            theta_dict['a_s'] = np.zeros(self.num_z)
            theta_dict['b2'] = np.array([params_dict[f'b2_z{i+1}'] for i in range(self.num_z)])
            theta_dict['sig_v'] = np.array([params_dict[f'sig_v_z{i+1}'] for i in range(self.num_z)])
        elif self.what_model == 'ZA':
            theta_dict['a_xi'] = np.zeros(self.num_z)
            theta_dict['a_v'] = np.zeros(self.num_z)
            theta_dict['a_s'] = np.zeros(self.num_z)
            theta_dict['b2'] = np.zeros(self.num_z)
            theta_dict['bs'] = np.zeros(self.num_z)
            theta_dict['sig_v'] = np.array([params_dict[f'sig_v_z{i+1}'] for i in range(self.num_z)])
        theta_dict['bn2'] = np.zeros(self.num_z)
        theta_dict['a_vp'] = np.zeros(self.num_z)
        theta_dict['beta_s'] = np.zeros(self.num_z)

        if self.what_stat == 'multipoles':
            return self.CModel.xiell_chi2(self.s, theta_dict, de_model=self.de_model, N_vec=self.num_z, inv_cov=self.icov, data_vec=self.data_to_fit)
        elif self.what_stat == 'wedges':
            return self.CModel.xiwedges_chi2(self.s, theta_dict, de_model=self.de_model, N_vec=self.num_z, inv_cov=self.icov, data_vec=self.data_to_fit, mu_edges=self.mu_edges)
        elif self.what_stat == 'proj':
            return self.CModel.xiproj_chi2(self.s, theta_dict, de_model=self.de_model, N_vec=self.num_z, inv_cov=self.icov, data_vec=self.data_to_fit, pi_max=self.pi_max)

    def test(self):
        tick = time.perf_counter()
        print(self.likelihood({'h':0.67, 'As':2.1, 'wc':0.12, 'b1_z1':0.7, 'b2_z1':0.1, 'sig_v_z1':0.5}))
        print(self.likelihood({'h':0.67, 'As':2.1, 'wc':0.12, 'b1_z1':0.7, 'b2_z1':0.1, 'sig_v_z1':1.5}))
        #print(self.likelihood({'f_z1':0.4, 's12_z1':0.8, 'q_lo_z1':1.05, 'q_tr_z1':0.98, 'b1_z1':0.5, 'b2_z1':0.1, 'bs_z1':0.01, 'a_xi_z1':1.5, 'a_v_z1':2.3, 'a_s_z1':3.5, 'sig_v_z1':0., 's12_z2':0.8, 'q_lo_z2':1.05, 'q_tr_z2':0.98, 'b1_z2':0.5, 'b2_z2':0.1, 'bs_z2':0.01, 'a_xi_z2':1.5, 'a_v_z2':2.3, 'sig_v_z2':0., 'f_z2':0.4, 'a_s_z2':3.5}))
        tock = time.perf_counter()
        print(tock-tick)


if __name__ == "__main__":
    cases = [[20, None], [25, None], [30,None], [35,None], [40,None], [45,None], [50,None]]
    cases = [[20, 'lambda'], [25, 'lambda'], [30,'lambda'], [35,'lambda'], [40,'lambda'], [45,'lambda'], [50,'lambda']]

    model_arg = sys.argv[1]
    case_num = int(sys.argv[2])
    what_statistic = sys.argv[3]
    data_type = sys.argv[4]

    raw_bins = sys.argv[5]
    redshift_bins = [int(num) for num in raw_bins.split(',')]

    the_chain = run_chain(model_arg, cases[case_num][0], cases[case_num][1], what_stat=what_statistic, multi_bins=redshift_bins, what_data=data_type)

    #the_chain.test()
    the_chain.full_run()



