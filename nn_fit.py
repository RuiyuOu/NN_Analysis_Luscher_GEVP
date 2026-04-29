#!/usr/bin/env python3

import sys
import os
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(linewidth=180)
import opt_einsum
import scipy as sp
from scipy.linalg import eigh
from scipy.linalg import sqrtm
from scipy.linalg import inv
from numpy.linalg import cond

from os import path

import tqdm
from colorama import Fore

import gvar as gv
import lsqfit

import nn_parameters as parameters
import bs_utils

import json_priors_utils as jsonpriors

def block_data(data, bl):
    ''' data shape is [Ncfg, others]
        bl = block length in configs
    '''
    ncfg = data.shape[0]
    dims = data.shape[1:]
    if ncfg % bl == 0:
        nb = ncfg // bl
    else:
        nb = ncfg // bl + 1
    corr_bl = np.zeros((nb,)+ dims, dtype=data.dtype)
    for b in range(nb-1):
        corr_bl[b] = data[b*bl:(b+1)*bl].mean(axis=0)
    corr_bl[nb-1] = data[(nb-1)*bl:].mean(axis=0)

    return corr_bl

class Fit:
    def __init__(self, params=None):
        print('numpy:',      np.__version__)
        print('opt_einsum:', opt_einsum.__version__)
        print('scipy:',      sp.__version__)
        print('matplotlib:', sys.modules[plt.__package__].__version__)
        print('h5py:',       h5.__version__)
        print('gvar:',       gv.__version__)
        print('lsqfit:',     lsqfit.__version__)
        if params is None:
            self.params = parameters.params()
        else:
            self.params = params

        if 'block' in self.params:
            self.block = self.params['block']
        else:
            self.block = 1

        if 't_norm' in self.params:
            self.t_norm = self.params['t_norm']
        else:
            self.t_norm = self.params['t0']

        ''' We solve either with 'gevp' or 'evp'
            evp:  Fisrt make G(t) = C(t0)**(-1/2) * C(t) * C(t0)**(-1/2)
                  then solve eigenvalue problem
                  eig = scipy.linalg.eigh(Gt)
            gevp: Solve the GEVP problem with 
                  eig = scipy.linalg.eigh(Ctd, Ct0)
        '''
        self.gevp = self.params['gevp']

        self.d_sets   = self.params["masterkey"]
        self.save_fit = self.params["save"]

        self.plot = Plot(self.params)
        self.func = Functions(self.params)

        # are we using restricted set of configs?
        if 'cfgs' in self.params:
            if len(self.params['cfgs']) > 1:
                ci = int(self.params['cfgs'][0])
                cf = int(self.params['cfgs'][1])
            else:
                ci = 0
                cf = int(self.params['cfgs'][0])
            if len(self.params['cfgs']) > 2:
                dc = int(self.params['cfgs'][2])
            else:
                dc = 1
            self.cfgs = range(ci,cf,dc)
            
        self.version = self.params['version']
        if self.version not in ['agnostic', 'conspire']:
            sys.exit('version must be in [ agnostic, conspire]')

        self.nstates = self.params['nstates']
        try:
            self.r_n_el = self.params['fit_choices'][key[0]]['r_n_el']
        except:
            self.r_n_el = self.params["r_n_el"]
        if self.params['version'] == 'agnostic':
            try:
                self.r_n_inel = self.params['fit_choices'][key[0]]['r_n_inel']
            except:
                self.r_n_inel = self.params['r_n_inel']

        if 'deuteron' in self.params["fpath"]["nn"].split('/')[-1]:
            nn = 'deuteron'
        elif 'dineutron' in self.params["fpath"]["nn"].split('/')[-1]:
            nn = 'dineutron'
        else:
            sys.exit('unkown nn data:', self.params["fpath"]["nn"].split('/')[-1])
        filename = f"NN_{nn}_tnorm{self.t_norm}_t0-td_{self.params['t0']}-{self.params['td']}"
        filename = f"{filename}_N_n{self.nstates}"
        filename = f"{filename}_t_{self.params['trange']['N'][0]}-{self.params['trange']['N'][1]}"
        filename = f"{filename}_NN_{self.params['version']}"
        if self.params['version'] == 'agnostic':
            filename = f"{filename}_n{self.params['r_n_inel']}_e{self.r_n_el}"
        elif self.params['version'] == 'conspire':
            if self.params['gs_conspire']:
                filename = f"{filename}_gs"
            filename = f"{filename}_e{self.r_n_el}"

        filename = f"{filename}_t_{self.params['trange']['R'][0]}-{self.params['trange']['R'][1]}"
        filename = f"{filename}_ratio_{self.params['ratio']}"

        # are we using restricted set of configs?
        if 'cfgs' in self.params:
            cfgs = f"cfgs{self.cfgs[0]}-{self.cfgs[-1]}x{self.cfgs[1]-self.cfgs[0]}"
            filename = f"{filename}_{cfgs}"

        # did we block the data?
        if self.block != 1:
            filename = f"{filename}_block{self.block}"

        # bs prior gs or all
        if self.params['bootstrap']:
            bs_prior = self.params['bs_prior']
            filename = f"{filename}_bsPrior-{bs_prior}"
            if 'old_bs' in self.params:
                self.old_bs = self.params['old_bs']
            else:
                self.old_bs = False
        # SVD study?
        if self.params['svd_study']:
            filename = f"{filename}_svdcut"
            if 'svdcut' in self.params:
                svd = str(self.params['svdcut'])
                filename = f"{filename}{svd}"
            else:
                filename = f"{filename}Opt"

        self.filename = f"{filename}.pickle"
        if self.params['bootstrap']:
            self.boot0_file = self.filename.replace('_bsPrior-'+bs_prior,"")

        self.data, self.irrep_dim = self.gevp_correlators()
        self.ratio_denom = self.get_ratio_combinations_Eff()
        self.ratio = self.params['ratio']
        #import IPython; IPython.embed()

        self.autopriors = self.params["autopriors"]

    def get_all_levels(self):
        d_sets = list(self.d_sets)
        new_dsets = []
        irreps = []
        for subset in d_sets:
            nd = (subset[0][0], subset[0][1])
            if nd not in irreps:
                irreps.append(nd)
        for irrep in irreps:
            for level in range(self.irrep_dim[irrep]):
                new_dsets.append([(irrep[0],irrep[1],level)])

        self.irreps   = irreps
        self.d_sets   = new_dsets
        self.save_fit = False

    def restore_masterkey(self):
        self.d_sets   = self.params["masterkey"]
        self.save_fit = self.params["save"]

    def compute_Zjn(self):
        ZjnSq_irrep = dict()
        for irrep in self.irreps:
            ''' The overlap factors of the original operators can be determined by
                Z_j^n = (C*V)_jn * A_n
                where A_n are the ground state factors for the n'th principle correlator
                D_n(t) = A_N**2 exp(-E0 t) * (1 + e.s.)
                The C*V factor is different for the EVP and GEVP methods
                EVP: eigh( Ct0m12 * Ctd * Ct0m12)
                    CV = sqrt(Ct0) * eVecs
                GEVP: eigh( Ctd, Ct0)
                    NOTE: for GEVP, the eVecs are not normalized but are solutions of
                          Ctd * eVec = eVal * Ct0 * eVec
                    CV = Ct0 * eVecs
            '''
            if self.gevp == 'evp':
                CV = opt_einsum.contract('jk,kn->jn', sqrtm(self.Ct0[irrep]), self.eVecs[irrep])
            elif self.gevp == 'gevp':
                CV = opt_einsum.contract('jk,kn->jn', self.Ct0[irrep], self.eVecs[irrep])
            ZjnSq = dict()
            for op_j in range(self.irrep_dim[irrep]):
                ZjSq = []
                for level in range(self.irrep_dim[irrep]):
                    n1,n2  = self.ratio_denom[(irrep[0], irrep[1], level)]
                    Z0_key = (((irrep[0], irrep[1], level), 'R', (n1,n2)), 'z0')
                    Z0     = self.posterior[Z0_key].mean
                    Zjn    = CV[op_j,level] * Z0
                    ZjSq.append(abs(Zjn)**2)
                ZjnSq[op_j] = np.array(ZjSq) / np.array(ZjSq).sum()
            ZjnSq_irrep[irrep] = ZjnSq
        self.ZjnSq = ZjnSq_irrep

        if self.params['Zjn_values']:

            IrrepPath = f"{self.params['masterkey'][0][0][0]}{self.params['masterkey'][0][0][1]}{self.params['masterkey'][0][0][2]}"

            outdir = f"result/{IrrepPath}"
            os.makedirs(outdir, exist_ok=True)

            with h5.File(f"result/{IrrepPath}/{self.params['Zjn_values']}",'w') as f5:
                for irrep in ZjnSq_irrep:
                    key = f"{irrep[0]}_{irrep[1]}"
                    for level in ZjnSq_irrep[irrep]:
                        f5.create_dataset(f"{key}/Zjn/{level}",data=ZjnSq_irrep[irrep][level])

    def read_Zjn(self):
        ZjnSq_irrep = dict()
        E_sort      = dict()
        
        IrrepPath = f"{self.params['masterkey'][0][0][0]}{self.params['masterkey'][0][0][1]}{self.params['masterkey'][0][0][2]}"

        print('reading Zjn values')
        print(f"result/{IrrepPath}/{self.params['Zjn_values']}")
        with h5.File(f"result/{IrrepPath}/{self.params['Zjn_values']}",'r') as f5:
            #for irrep in f5.keys():
                #key = tuple(irrep.split('_'))
            for key in self.irreps:
                irrep = key[0]+'_'+key[1]
                ZjnSq_irrep[key] = dict()
                try:
                    for level in f5[f"{irrep}/Zjn"].keys():
                        ZjnSq_irrep[key][int(level)] = f5[f"{irrep}/Zjn/{level}"][()]
                except:
                    print()
                    print(f"your {f"result/{IrrepPath}/{self.params['Zjn_values']}"} is missing irreps")
                    sys.exit(f"remake {f"result/{IrrepPath}/{self.params['Zjn_values']}"} with all irreps")
                E_sort[irrep] = f5[f"{irrep}/E_sort"][()]
        self.ZjnSq = ZjnSq_irrep
        self.E_sort = E_sort

    def report_ZjnSq(self):
        nn_ops = self.get_nn_operators()
        for irrep in self.ZjnSq:
            print('\n',irrep)
            print('%4s  %35s  %s' %('op_j', 'op_ID', 'level - max(Zjn)'))

            for op_j in range(self.irrep_dim[irrep]):
                opt_id = self.ZjnSq[irrep][op_j].argmax()
                opt_op = self.nn_ops[irrep][opt_id]
                op_lbl = self.nn_ops[irrep][op_j]

                op_lbl = nn_ops[irrep][op_lbl]['label']
                lbl    = f"{irrep[0]}_{irrep[1]}_O{op_j}"
                print('%4d  %35s  %d' %(op_j, op_lbl, opt_id))
                plt.figure(lbl, figsize=(7,5.5))
                ax = plt.axes([0.12,0.12,0.87,0.87])
                for level in range(self.irrep_dim[irrep]):
                    ax.bar(level, self.ZjnSq[irrep][op_j][level])
                ax.set_ylabel(r'$|Z_{j=%d}^{(n)}|^2$' %op_j, fontsize=20)
                ax.set_xlabel(r'$n^{th}$-level', fontsize=20)
                ax.text(0.5, 0.9, r'%s(%s): $O_{%d}$' %(irrep[1],irrep[0],op_j),
                        verticalalignment='bottom', horizontalalignment='center',
                        transform=ax.transAxes,
                        bbox={'boxstyle':'round', 'facecolor':'None'},
                        fontsize=20)
                ax.set_ylim([0,1])
                plt.savefig(f"figures/{irrep[0]}_{irrep[1]}_Z_{op_j}n.pdf", transparent=True)
            # order levels
            E_tot = []
            for level in range(self.irrep_dim[irrep]):
                if any([(irrep[0], irrep[1], level) in k for k in self.d_sets]):
                    n1, n2 = self.ratio_denom[(irrep[0], irrep[1], level)]
                    E1     = self.posterior[(((irrep[0], irrep[1], level), 'N', n1), 'e0')]
                    E2     = self.posterior[(((irrep[0], irrep[1], level), 'N', n2), 'e0')]
                    dE     = self.posterior[(((irrep[0], irrep[1], level), 'R', (n1, n2)), 'e0')]
                    Z0     = self.posterior[(((irrep[0], irrep[1], level), 'R', (n1, n2)), 'z0')]
                    print(irrep, '%2d' %level, E1+E2+dE, E1, E2, dE,Z0)
                    E_tot.append((E1+E2+dE).mean)
            E_tot = np.array(E_tot)
            i_sort = np.argsort(E_tot)
            print(i_sort)

            IrrepPath = f"{self.params['masterkey'][0][0][0]}{self.params['masterkey'][0][0][1]}{self.params['masterkey'][0][0][2]}"

            with h5.File(f"result/{IrrepPath}/{self.params['Zjn_values']}",'a') as f5:
                key = f"{irrep[0]}_{irrep[1]}"
                f5.create_dataset(f"{key}/E_sort",data=i_sort)

    def nucleon_data(self):
        """ Reads nucleon data from h5.
        Constructs same momentum lists.
        Average correlators of same momentum.
        Construct dictionary of momentum^2 with data of shape [ncfg, tsep].
        """
        fpath = self.params["fpath"]["nucleon"]
        file = h5.File(fpath, "r")
        correlators = file.keys()
        avglist = dict()
        dcorr = dict()
        for correlator in correlators:
            if 'cfgs' in dir(self):
                dcorr[correlator] = file[f"{correlator}/data"][()][self.cfgs].real
            else:
                dcorr[correlator] = file[f"{correlator}/data"][()].real
            irrep = correlator.split("_")[1]
            mom = correlator.split("_")[0][1:].replace("m", "")
            mom2 = f"{sum([int(i) ** 2 for i in mom])}"
            if mom2 == "5":
                mom2 = f"{mom2}{irrep}"
            if mom2 in avglist:
                avglist[mom2].append(correlator)
            else:
                avglist[mom2] = [correlator]
        if self.params["debug"]:
            """Plot effective mass for each correlator"""
            for d in avglist:
                for c in avglist[d]:
                    self.plot.simple_plot(dcorr, c)
                    break
                break
        data = dict()
        for mom2 in avglist:
            data[mom2] = np.average([dcorr[corr] for corr in avglist[mom2]], axis=0)
        if self.block != 1:
            new_data = dict()
            for k in data:
                new_data[k] = block_data(data[k], self.block)
            data = new_data
        if self.params["debug"]:
            """Plot effective mass for averaged correlator"""
            for mom2 in data:
                self.plot.simple_plot(data, mom2)
        return data

    def nn_data(self):
        if 'make_Hermitian' in self.params:
            self.make_Hermitian = self.params['make_Hermitian']
        else:
            self.make_Hermitian = True

        fpath = self.params["fpath"]["nn"]
        data = dict()
        file = h5.File(fpath, "r")
        correlators = file.keys()
        for correlator in correlators:
            try:# data packed via Drew
                psq,irrep = correlator.split("_")
                mom2 = psq.split("PSQ")[1]
            except:# old way where Andre/Jason took Drew's data and repacked it
                irrep,psq = correlator.split("_")
                mom2 = psq.split("Psq")[1]
            tag   = (mom2, irrep)
            if 'cfgs' in dir(self):
                corr = file[f"{correlator}/data"][()][self.cfgs]
            else:
                corr = file[f"{correlator}/data"][()]
            if self.make_Hermitian and len(corr.shape) == 4:
                # restore Hermiticity of NN data
                corr_full = np.zeros_like(corr)
                for i in range(corr.shape[1]):
                    for j in range(corr.shape[2]):
                        re = corr[0,i,j,self.params['t0']].real
                        im = corr[0,i,j,self.params['t0']].imag
                        if re != 0 or im != 0:
                            corr_full[:,i,j,:] = corr[:,i,j,:]
                        else:
                            corr_full[:,i,j,:] = np.conjugate(corr[:,j,i,:])
                corr = 0.5*(corr_full + np.conjugate(np.einsum('cijt->cjit', corr_full)))
            # if nn data in shape (Ncfg, Nt), we need to recast it to (Ncfg, 1,1, Nt)
            # for this code to work properly
            if len(corr.shape) == 2:
                corr_full = np.zeros([corr.shape[0],1,1,corr.shape[1]], dtype=float)
                corr_full[:,0,0,:] = corr.real
                corr = corr_full

            # block data
            if self.block != 1:
                corr = block_data(corr, self.block)

            # normalize data
            # C_ij -> C_ij / sqrt(C_ii(t_norm) C_jj(t_norm))
            if 't_norm' in self.params:
                t_norm = self.params['t_norm']
            else:
                t_norm = self.params['t0']
            #'''
            if len(corr.shape) == 4 and t_norm != 'None':
                C_norm = np.diagonal(corr.mean(axis=0)[:,:,t_norm]).real
                corr_full = np.zeros_like(corr)
                for i in range(corr.shape[1]):
                    for j in range(corr.shape[2]):
                        corr_full[:,i,j,:] = corr[:,i,j,:] / np.sqrt(C_norm[i]*C_norm[j])
                corr = corr_full

            data[tag] = corr

        return data

    def get_equivalent_momenta(self, file):
        correlators = file.keys()
        avglist = dict()
        dcorr = dict()
        for correlator in correlators:
            dcorr[correlator] = file[f"{correlator}/data"][()]
            irrep = correlator.split("_")[1]
            mom = correlator.split("_")[0][1:].replace("m", "")
            mom2 = sum([int(i) ** 2 for i in mom])
            tag = (mom2, irrep)
            if tag in avglist:
                avglist[tag].append(correlator)
            else:
                avglist[tag] = [correlator]
        return dcorr, avglist

    def n_operators(self,operators):
        n_ops = {}
        for k in operators:
            iso,b1,b2 = k.split()[0].split('_')
            if 'CG_' in k.split()[2]:
                CG    = k.split()[2].split('_')[1]
                shift = 1
            else:
                CG    = '0'
                shift = 0
            psq1   = int(k.split()[2+shift].split('PSQ=')[1])
            psq2   = int(k.split()[5+shift].split('PSQ=')[1])
            irrep1 = k.split()[3+shift]
            irrep2 = k.split()[6+shift]
            op_key = {}
            op_key['label'] = r'%s-%s(%d) %s-%s(%d): CG %s' %(b1,irrep1,psq1,b2,irrep2,psq2,CG)
            op_key['B1'] = {'state':b1, 'PSQ':psq1, 'irrep':irrep1}
            op_key['B2'] = {'state':b2, 'PSQ':psq2, 'irrep':irrep2}
            n_ops[k] = op_key

        return n_ops

    def get_nn_operators(self):
        fpath = self.params["fpath"]["nn"]
        with h5.File(fpath, 'r') as f5:
            nn_ops  = dict()
            n_irrep = dict()
            correlators = f5.keys()
            for correlator in correlators:
                try:
                    psq,irrep    = correlator.split('_')
                    mom2         = psq.split('PSQ')[1]
                except:
                    irrep,psq = correlator.split("_")
                    mom2 = psq.split("Psq")[1]
                tag          = (mom2, irrep)
                nn_ops[tag]  = f5[correlator].attrs['op_list']
                n_irrep[tag] = self.n_operators(nn_ops[tag])
        self.nn_ops = nn_ops

        return n_irrep

    def get_ratio_combinations_Zjn(self):
        ''' Use fitted overlap factors to decide optimal reference N N states
        '''
        print('getting NN reference states by Z_jn overlaps')
        nn_ops = self.get_nn_operators()
        ratio_denom = {}
        for irrep in self.ZjnSq:
            n_op_j = {k:[] for k in range(self.irrep_dim[irrep])}
            for op_j in range(self.irrep_dim[irrep]):
                opt_id = self.ZjnSq[irrep][op_j].argmax()
                opp    = self.nn_ops[irrep][op_j]
                B1     = nn_ops[irrep][opp]['B1']
                B2     = nn_ops[irrep][opp]['B2']
                e1     = '%d' %B1['PSQ']
                e2     = '%d' %B2['PSQ']
                if B1['PSQ'] > 4:
                    e1 += B1['irrep']
                if B2['PSQ'] > 4:
                    e2 += B2['irrep']
                tag = [e1,e2]
                if tag not in n_op_j[opt_id]:
                    n_op_j[opt_id].append(tag)
            for k in n_op_j:
                if len(n_op_j[k]) > 0:
                    ratio_denom[(irrep[0], irrep[1], k)] = n_op_j[k][0]
        return ratio_denom

    def get_ratio_combinations_Eff(self):
        ''' This function compares the energy of all choices of two single nucleon
            operators that overlap with the state, and finds the one with an energy
            closest to the NN energy to decide what set of single nucleon operators
            to pair with the NN correlator.
            It uses the effective mass of the single nucleons at the autotime chosen
            by the user to estimate the energy.
        '''
        print('getting NN reference states by Eff Mass')
        autotime = self.params["autotime"]
        irrep = self.get_nn_operators()
        nonint_lvls = dict()
        for tag in irrep:
            tag_lst = []
            meff_list = []
            for element in irrep[tag]:
                B1 = irrep[tag][element]['B1']
                B2 = irrep[tag][element]['B2']
                e1 = '%d' %B1['PSQ']
                e2 = '%d' %B2['PSQ']
                if B1['PSQ'] > 4:
                    e1 += B1['irrep']
                if B2['PSQ'] > 4:
                    e2 += B2['irrep']
                x, meff1 = self.func.meff(self.data[e1])
                meff1 = meff1[x.index(autotime)].mean
                x, meff2 = self.func.meff(self.data[e2])
                meff2 = meff2[x.index(autotime)].mean
                meff_list.append(meff1 + meff2)
                tag_lst.append([e1,e2])
            nonint_lvls[tag] = {"meff": np.array(meff_list), "irrep": tag_lst}
        ratio_denom = dict()

        for tag in self.data:
            if tag in ["0", "1", "2", "3", "4", "5F1", "5F2"]:
                continue
            x, meff = self.func.meff(self.data[tag])
            meff = meff[x.index(autotime)].mean
            idx = np.abs(nonint_lvls[(tag[0], tag[1])]["meff"] - meff).argmin()
            ratio_denom[tag] = nonint_lvls[(tag[0], tag[1])]["irrep"][idx]
        
        return ratio_denom

    def make_bootstrap_list(self):
        try:
            ncfgs = len(self.nucleon_data()[0])
            bslist = pd.read_csv(f"./data/bslist_{ncfgs}.csv", sep=";", header=0)
        except:
            from random import randint

            nbs = 5000
            fpath = self.params["fpath"]["nucleon"]
            file = h5.File(fpath, "r")
            correlators = list(file.keys())
            ncfgs = len(file[f"{correlators[0]}/data"][()])
            draws = []
            for idx in range(nbs):
                idraw = []
                for draw in range(ncfgs):
                    idraw.append(randint(0, ncfgs - 1))
                draws.append(idraw)
            output = "nbs;draws\n"
            output += f"0;{str(list(range(ncfgs))).replace(' ', '')}\n"
            output += "\n".join(
                [
                    f"{idx + 1};{str(draw).replace(' ', '')}"
                    for idx, draw in enumerate(draws)
                ]
            )
            with open(f"./data/bslist_{ncfgs}.csv", "w") as file:
                file.write(output)
            bslist = pd.read_csv(f"./data/bslist_{ncfgs}.csv", sep=";", header=0)
        return bslist

    def gevp_correlators(self):

        def get_gevp_rotation(data, verbose=True):

            t0 = self.params["t0"]
            td = self.params["td"]
            eVecs_irrep = dict()
            Ct0_irrep   = dict()
            for key in data:
                if len(np.shape(data[key])) == 4:
                    Ct  = np.average(data[key], axis=0)
                    try:
                        Ct0 = Ct[:,:, t0]
                        Ctd = Ct[:,:, td]
                        if self.gevp == 'evp':
                            Gt  = inv(sqrtm(Ct0)) @ Ctd @ inv(sqrtm(Ct0))
                            eval, evec = eigh(Gt)
                        elif self.gevp == 'gevp':
                            eval, evec = eigh(Ctd, Ct0)
                        eVecs_irrep[key] = np.fliplr(evec)
                        #eVecs_irrep[key] = evec
                        Ct0_irrep[key]  = Ct0
                        if verbose:
                            C_shape = Ct0.shape
                            print(f"\n{key} {C_shape}, Success, condition numbers:")
                            print("  cond(Ct0) = %.3f" %cond(Ct0))
                            if self.gevp == 'evp':
                                print("  cond(Gt)  = %.3f" %cond(Gt))
                            elif self.gevp == 'gevp':
                                print("  cond(Ctd)  = %.3f" %cond(Ctd))

                    except Exception as e:
                        print(e)
                        print(f"{key} Fail, condition numbers:")
                        print(cond(Ct[:, :, td]), cond(Ct[:, :, t0]))
                        val1, vec1 = eigh(Ct[:, :, td])
                        val2, vec2 = eigh(Ct[:, :, t0])
                        print(f"{key} {t0} Eigenvalue spectrum and vector:")
                        print(val1)
                        print(vec1[0])
                        print(f"{key} {td} Eigenvalue spectrum and vector:")
                        print(val2)
                        print(vec2[0])
            
            self.eVecs = eVecs_irrep
            self.Ct0   = Ct0_irrep
        
        def do_gevp_rotation(verbose=True):
            nucleon = self.nucleon_data()
            singlet = self.nn_data()

            # BB data will have shape = (Ncfg, Nt, Nop, Nop)
            allsing = {
                (key[0],key[1],0): singlet[key] for key in singlet if len(np.shape(singlet[key])) == 2
            }

            get_gevp_rotation(singlet, verbose=verbose)
            for key in self.eVecs:
                eigVecs = self.eVecs[key]
                Ct = singlet[key]
                if self.gevp == 'evp':
                    # construct G(t) = Ct0**(-1/2) C(t) Ct0**(-1/2)
                    Ct0InvSqrt = inv(sqrtm(self.Ct0[key]))
                    Gt = opt_einsum.contract('ik,cklt,lj->cijt', Ct0InvSqrt, Ct, Ct0InvSqrt)
                    # rotate G(t)
                    rotated_singlet = opt_einsum.contract('in,cijt,jm->cnmt', np.conj(eigVecs), Gt, eigVecs)
                elif self.gevp == 'gevp':
                    # GEVP instead of EVP
                    rotated_singlet = opt_einsum.contract('in,cijt,jm->cnmt', np.conj(eigVecs), Ct, eigVecs)
                # take the diagonal elements only
                for operator in range(np.shape(rotated_singlet)[1]):
                    opkey = (key[0], key[1], operator)
                    allsing[opkey] = rotated_singlet[:, operator, operator, :].real

            if self.params["bootstrap"]:
                # if we have blocked - nucleon has the correct number of "configs"
                ncfg = nucleon[next(iter(nucleon))].shape[0]
                if self.old_bs:
                    with h5.File('data/bslist_802.h5','r') as tmp5:
                        self.draws = tmp5['bslist_802'][()]
                else:
                    self.draws = bs_utils.make_bs_list(ncfg, self.params['Nbs_max'], seed=self.params['bs_seed'])
                self.h5_bs = True
            else:
                self.h5_bs = False                

            self.bsdata = {**nucleon, **allsing}

            return nucleon, allsing


        t0 = self.params["t0"]
        td = self.params["td"]
        if 'deuteron' in self.params["fpath"]["nn"].split('/')[-1]:
            nn = 'deuteron'
        elif 'dineutron' in self.params["fpath"]["nn"].split('/')[-1]:
            nn = 'dineutron'
        else:
            sys.exit('unkown nn data:', self.params["fpath"]["nn"].split('/')[-1])
        datapath = f"./data/gevp_{nn}_tnorm{self.t_norm}_{self.gevp}_{t0}-{td}"
        if 'cfgs' in dir(self):
            cfgs = f"cfgs{self.cfgs[0]}-{self.cfgs[-1]}x{self.cfgs[1]-self.cfgs[0]}"
            datapath = f"{datapath}_{cfgs}"
        if self.block != 1:
            datapath = f"{datapath}_block{self.block}"
        datapath = f"{datapath}.pickle"
        if path.exists(datapath) and self.params["bootstrap"] is False:
            print("Read data from gvar dump")
            print(f"    {datapath}")
            gvdata = gv.load(datapath)
            self.h5_bs = False
            if self.params['svd_study'] or self.params['do_gevp'] or self.params['get_Zj']:
                nucleon, allsing = do_gevp_rotation(verbose=False)
        else:
            print("Constructing data from HDF5")
            nucleon, allsing = do_gevp_rotation()

            print('\nThe principle value ROT correlators are REAL at inf statistics, so we discard imaginary')
            gvdata = gv.dataset.avg_data({**nucleon, **allsing})
            if not os.path.exists(datapath):
                gv.dump(gvdata, datapath)

            if self.params["debug"]:
                for key in gvdata:
                    if len(np.shape(gvdata[key])) == 2:
                        for corr in range(len(gvdata[key][0])):
                            print(key, corr)
                            print(gvdata[key][:, corr])
                            self.plot.simple_plot(
                                {f"{key}_{corr}": gvdata[key][:, corr].flatten()},
                                f"{key}_{corr}",
                            )
                    else:
                        self.plot.simple_plot(gvdata, key)

        irrep_dim = dict()
        for irrep_op in gvdata.keys():
            if irrep_op in ["0", "1", "2", "3", "4", "5F1", "5F2"]:
                continue
            irrep = (irrep_op[0], irrep_op[1])
            if irrep in irrep_dim:
                irrep_dim[irrep] += 1
            else:
                irrep_dim[irrep] = 1

        return gvdata, irrep_dim

    def set_priors(self, prior, data, nbs, type="auto", seed=''):
        """
        Will always generate new, uncorrelated priors for single nucleon in each chain.

        return a dictionary of gvar priors.
        If set to auto, the energy and overlap factors are inferred from meff and zeff.
        Data (x, y) is required define parameters, and is also used to set auto priors(data to be fit)
        If set to manual, the priors are read from priors.py
        """
        if type == "auto":
            autotime = self.params["autotime"]
            datax, datay = data
            for key in sorted(datax.keys()):
                # ground state priors estimated from m_eff and z_eff
                _, meff = self.func.meff(datay[key], datax[key])
                x, zeff = self.func.zeff(datay[key], datax[key])
                if autotime in x:
                    meff = siground(meff[x.index(autotime)].mean)
                    zeff = siground(zeff[x.index(autotime)].mean)
                else:
                    meff = siground(meff[x.index(x[0])].mean)
                    zeff = siground(zeff[x.index(x[0])].mean)
                meff_mean = meff
                meff_sdev = np.absolute(meff)
                zeff_mean = zeff
                zeff_sdev = np.absolute(zeff)

                if nbs > 0:
                    # generate priors
                    Nbs = self.params['nbs']
                    bs_w_fac = self.params['bs0_width']
                    s_e0 = self.posterior[(key, 'e0')].sdev
                    s_z0 = self.posterior[(key, 'z0')].sdev

                    # for the g.s., we tighten the random shift of the prior
                    # since our g.s. prior is very broad.
                    # Otherwise, we end up in local minimum in the BS samples
                    e0_seed = str((key, 'e0'+seed))
                    z0_seed = str((key, 'z0'+seed))
                    e0_bs = bs_utils.bs_prior(Nbs, mean=meff_mean, sdev=bs_w_fac*s_e0, seed=e0_seed)
                    z0_bs = bs_utils.bs_prior(Nbs, mean=zeff_mean, sdev=bs_w_fac*s_z0, seed=z0_seed)

                    if key[1] in ["R"]:
                        prior[(key, "e0")] = gv.gvar(e0_bs[nbs-1], self.params["sig_e0"] * meff_sdev)
                    else:
                        prior[(key, "e0")] = gv.gvar(e0_bs[nbs-1], 0.1*meff_sdev)
                    prior[(key, "z0")] = gv.gvar(z0_bs[nbs-1], zeff_sdev)

                else:
                    # for NN, we use a larger width for the interaction energy, than for the Nucleons
                    if key[1] in ["R"]:
                        prior[(key, "e0")] = gv.gvar(meff_mean, self.params["sig_e0"] * meff_sdev)
                    else:
                        prior[(key, "e0")] = gv.gvar(meff_mean, 0.1*meff_sdev)
                    prior[(key, "z0")]     = gv.gvar(zeff_mean, zeff_sdev)

                # excited state priors
                if key[1] in ["N"]:
                    states = self.params["nstates"]
                    for n in range(1, states):
                        en_mean = self.params["ampi"] * 2
                        if nbs > 0 and self.params['bs_prior'] == 'all':
                            en_bs = bs_utils.bs_prior(Nbs, mean=en_mean, sdev=en_mean/2, seed=str((key, f"e{n}"+seed)), dist='lognormal')
                            s_z   = bs_w_fac * self.posterior[(key, f"z{n}")].sdev
                            zn_bs = bs_utils.bs_prior(Nbs, mean=1, sdev=s_z, seed=str((key, f"z{n}"+seed)))
                            en    = np.log(en_bs[nbs-1])
                            zn    = zn_bs[nbs-1]
                        else:
                            en = np.log(en_mean)
                            zn = 1.0
                        prior[(key, f"e{n}")] = gv.gvar(en, 0.7)
                        prior[(key, f"z{n}")] = gv.gvar(zn, 0.25)

                elif key[1] in ["R"]:
                    # elastic NN priors
                    dE_elastic = self.params["dE_elastic"]
                    for n in range(1,1 + self.r_n_el):
                        if nbs > 0:
                            en_bs = bs_utils.bs_prior(Nbs, mean=dE_elastic, sdev=dE_elastic/2, seed=str((key, f"e_el{n}"+seed)), dist='lognormal')
                            s_z = bs_w_fac * self.posterior[(key, f"z_el{n}")].sdev
                            zn_bs = bs_utils.bs_prior(Nbs, mean=1, sdev=s_z, seed=str((key, f"z_el{n}"+seed)))
                            en    = np.log(en_bs[nbs-1])
                            zn    = zn_bs[nbs-1]
                        else:
                            en = np.log(dE_elastic)
                            zn = 1.0
                        prior[(key, f"e_el{n}")] = gv.gvar(en, 0.7)
                        prior[(key, f"z_el{n}")] = gv.gvar(zn, 0.25)

                    # inelastic NN priors
                    if self.version == 'agnostic':
                        for n in range(1, self.r_n_inel):
                            if nbs > 0:
                                dE    = np.exp(self.posterior[(key, f"e{n}")])
                                en_bs = bs_utils.bs_prior(Nbs, mean=dE.mean, sdev=bs_w_fac*dE.sdev,
                                                          seed=str((key, f"e{n}"+seed)), dist='lognormal')
                                s_z   = bs_w_fac * self.posterior[(key, f"z{n}")].sdev
                                zn_bs = bs_utils.bs_prior(Nbs, mean=1, sdev=s_z, seed=str((key, f"z{n}"+seed)))
                                en    = en_bs[nbs-1]
                                zn    = zn_bs[nbs-1]
                            else:
                                en = 2*self.params['ampi']
                                zn = 1.0
                            prior[(key, f"e{n}")] = gv.gvar(np.log(en), 0.7)
                            prior[(key, f"z{n}")] = gv.gvar(zn, 0.25)

                    elif self.version == 'conspire':
                        # gs_conspire only adds interaction energy to ground state in tower of NN states
                        # which we prior as a normal distribution
                        sig_factor = self.params['sig_enn']
                        for n1 in range(self.nstates):
                            key1 = (key[0],"N",key[2][0])

                            for n2 in range(self.nstates):
                                key2 = (key[0],"N",key[2][1])
                                if n1 == 0 and n2 == 0:
                                    pass
                                else:
                                    # set delta_NN.mean = 0 for these energies
                                    e0 = abs(prior[(key, "e0")].mean)
                                    s_e = sig_factor * e0
                                    if nbs > 0 and self.params['bs_prior'] == 'all':
                                        # deltaE gets normal, not lognormal
                                        en_bs = bs_utils.bs_prior(Nbs, mean=0, sdev=s_e, seed=str((key, f"e_{n1}_{n2}"+seed)))
                                        s_z   = bs_w_fac * self.posterior[(key, f"z_{n1}_{n2}")].sdev
                                        zn_bs = bs_utils.bs_prior(Nbs, mean=1, sdev=s_z, seed=str((key, f"z_{n1}_{n2}"+seed)))
                                        en    = en_bs[nbs-1]
                                        zn    = zn_bs[nbs-1]
                                    else:
                                        en = 0.0
                                        zn = 1.0

                                    if n2 >= n1:
                                        if not self.params['gs_conspire']:
                                            # only add deltaE if we add for all states in tower
                                            prior[(key, f"e_{n1}_{n2}")] = gv.gvar(en, sig_factor * e0)
                                        prior[(key, f"z_{n1}_{n2}")] = gv.gvar(zn, 0.25)
                                    else:
                                        if key2 == key1:
                                            if not self.params['gs_conspire']:
                                                prior[(key, f"e_{n1}_{n2}")] = prior[(key, f"e_{n2}_{n1}")]
                                            prior[(key, f"z_{n1}_{n2}")] = prior[(key, f"z_{n2}_{n1}")]
                                        else:
                                            if not self.params['gs_conspire']:
                                                prior[(key, f"e_{n1}_{n2}")] = gv.gvar(en, sig_factor * e0)
                                            prior[(key, f"z_{n1}_{n2}")] = gv.gvar(zn, 0.25)

                    jsonpriors.save_json_priors(prior,filename="nn_priors.json")

                    if self.params["debug"]:
                        for k in prior:
                            print(k, prior[k])

        elif type == "manual":
            print("Loading manual priors...")
            prior = jsonpriors.load_json_priors(filename="nn_priors.json")
            if self.params["debug"]:
                for k in prior:
                    print(k, prior[k])
        return prior

    def format_data(self, subset, nbs=0, ratio=True, svdcut=False):
        """
        The assumption is that the fit will always simultaneously perform a fit to the
        2N two nucleon correlator
        or
        2N / N1*N2 ratio correlator
        and
        N1 and N2 single nucleon correlators
        """
        x   = dict()
        y0  = dict()
        ybs = dict()
        if svdcut:
            ysvd     = dict()
            svd_cuts = dict()
        for key in subset:
            k0 = self.ratio_denom[key][0]
            k1 = self.ratio_denom[key][1]
            key_ratio = (key, "R", (k0, k1))
            key_nucl0 = (key, "N", k0)
            key_nucl1 = (key, "N", k1)
            try:
                trange = self.params["fit_choices"][key]['trange']
                x[key_ratio] = list(range(trange[0], trange[1] + 1))
            except:
                if self.params["debug"]:
                    print('you have not specified a trange for %s\n' % str(key))
                x[key_ratio] = list(range(self.params["trange"]["R"][0], self.params["trange"]["R"][1] + 1))
            x[key_nucl0]  = list(range(self.params["trange"]["N"][0], self.params["trange"]["N"][1] + 1))
            x[key_nucl1]  = list(range(self.params["trange"]["N"][0], self.params["trange"]["N"][1] + 1))
            numerator     = self.data[key][x[key_ratio]]
            if ratio:
                denominator = (self.data[k0][x[key_ratio]] * self.data[k1][x[key_ratio]])
            else:
                denominator = 1
            y0[key_ratio] = numerator / denominator
            y0[key_nucl0] = self.data[k0][x[key_nucl0]]
            y0[key_nucl1] = self.data[k1][x[key_nucl1]]
            
            if svdcut:
                if 'svdcut' not in dir(self) or ('svdcut' in dir(self) and key_ratio not in self.svdcut):
                    ysvd[key_ratio] = self.bsdata[key][x[key_ratio]]
                    ysvd[key_nucl0] = self.bsdata[k0][x[key_nucl0]]
                    ysvd[key_nucl1] = self.bsdata[k1][x[key_nucl1]]

                    if ratio:
                        def svd_processor(data):
                            d  = gv.dataset.avg_data(data)
                            d2 = gv.BufferDict()
                            d2[key_ratio] = d[key_ratio] / d[key_nucl0] / d[key_nucl1]
                            d2[key_nucl0] = d[key_nucl0]
                            d2[key_nucl1] = d[key_nucl1]
                            return d2
                        svd_test = gv.dataset.svd_diagnosis(ysvd, process_dataset=svd_processor)
                    else:
                        svd_test = gv.dataset.svd_diagnosis(ysvd)
                    svd_cuts[key_ratio] = svd_test.svdcut
            
            if not self.params['bootstrap']:
                ybs = []
            else:
                if self.h5_bs:
                    mask = self.draws[nbs]
                else:
                    mask = [int(m) for m in self.draws["draws"].iloc[nbs][1:-1].split(",")]
                # only select data we want for this particular fit
                # key = NN
                # k0, k1 = single nucleon keys corresponding to NN
                wanted_keys = [key, k0, k1]
                ndata = {k: self.bsdata[k][mask] for k in self.bsdata if k in wanted_keys}
                ndataset  = gv.dataset.avg_data(ndata)

                numerator = ndataset[key][x[key_ratio]]
                if ratio:
                    denominator = (ndataset[k0][x[key_ratio]] * ndataset[k1][x[key_ratio]])
                else:
                    denominator = 1
                ybs[key_ratio] = numerator / denominator
                ybs[key_nucl0] = ndataset[k0][x[key_nucl0]]
                ybs[key_nucl1] = ndataset[k1][x[key_nucl1]]

        if svdcut:
            if 'svdcut' not in dir(self):
                self.svdcut = dict()
            for k in svd_cuts:
                self.svdcut[k] = svd_cuts[k]

        return x, y0, ybs

    def reconstruct_gs(self, posterior):
        """
        Reconstruct ground state energy if ratio fit is used
        """
        for subset in self.d_sets:
            for key in subset:
                dset = self.ratio_denom[key]
                offset = posterior[((key, "N", dset[0]), "e0")] + posterior[((key, "N", dset[1]), "e0")]
                posterior[(key, "egs")] = posterior[((key, "R", (dset[0], dset[1])), "e0")] + offset
        return posterior

    def fit(self, n_start=0, ndraws=0):
        bsresult = dict()
        p0 = dict()
        if n_start==0:
            bi=0
            bf=ndraws+1
        else:
            bi=n_start+1
            bf=n_start+1+ndraws
            b0 = self.get_b0_posteriors()
            # only add self.posterior if not already defined
            if not 'posterior' in dir(self):
                self.posterior = b0
            for subset in self.d_sets:
                p0[subset[0]] = dict()
                for k in b0:
                    if subset[0] == k[0][0]:
                        try:
                            p0[subset[0]][k] = b0[k].mean
                        except:
                            pass

        for nbs in tqdm.tqdm(range(bi,bf,1)):
            posterior = gv.BufferDict()
            masterkey = tqdm.tqdm(self.d_sets)
            #print('\nDEBUG\n')
            #for subset in masterkey:
            #    print(subset)
            #debug = input('continue?')
            for subset in masterkey:
                if nbs > 0:
                    ''' all gvar's created in this switch are destroyed at restore_gvar
                        [they are out of scope] '''
                    gv.switch_gvar()

                masterkey.set_description(f"Fitting {subset}")
                masterkey.bar_format = "{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)
                if self.ratio:
                    x, y0, ybs = self.format_data(subset, nbs, svdcut=self.params['svd_study'])
                    prior = gv.BufferDict()
                    prior = self.set_priors(prior, data=(x, y0), nbs=nbs, type=self.autopriors)
                else:
                    # make Energy priors with ratio
                    x, y0, ybs = self.format_data(subset, nbs, ratio=True)
                    prior = gv.BufferDict()
                    prior = self.set_priors(prior, data=(x, y0), nbs=nbs, type=self.autopriors)

                    # remake data without ratio
                    x, y0, ybs = self.format_data(subset, nbs, ratio=False, svdcut=self.params['svd_study'])
                    # we need to make ground state overlap factor priors with non-ratio data
                    prior_z = gv.BufferDict()
                    prior_z = self.set_priors(prior_z, data=(x, y0), nbs=nbs, type=self.autopriors)
                    for k in prior_z:
                        if k[-1] == 'z0':
                            # the single nucleon overlaps will be determined the same in both cases
                            prior[k] = prior_z[k]


                # SVD cut?
                if self.params['svd_study']:
                    if 'svdcut' in self.params:
                        svdcut = self.params['svdcut']
                    else:
                        k0 = self.ratio_denom[subset[0]][0]
                        k1 = self.ratio_denom[subset[0]][1]
                        key_ratio = (subset[0], "R", (k0, k1))
                        svdcut = self.svdcut[key_ratio]
                else:
                    svdcut=None

                if nbs == 0:
                    result = lsqfit.nonlinear_fit(
                        data=(x, y0), prior=prior, fcn=self.func, 
                        maxit=100000, fitter=self.params['fitter'], svdcut=svdcut
                    )
                    p0[subset[0]] = {k:v.mean for k,v in result.p.items()}
                else:
                    p0_bs = {k:p0[subset[0]][k] for k in prior}
                    try:
                        result = lsqfit.nonlinear_fit(
                            data=(x, ybs), prior=prior, p0=p0_bs, fcn=self.func, 
                            maxit=100000, fitter=self.params['fitter'], svdcut=svdcut
                        )
                    except Exception as e:
                        print(e)
                        print('trying new BS prior_mean')
                        # I need to fix priors based upon ratio or not
                        prior = self.set_priors(prior, data=(x, y0), nbs=nbs, type=self.autopriors, seed='2')
                        result = lsqfit.nonlinear_fit(
                            data=(x, ybs), prior=prior, p0=p0_bs, fcn=self.func, 
                            maxit=100000, fitter=self.params['fitter'], svdcut=svdcut
                        )
                        '''
                        have_result = False
                        svdcut = 1e-11
                        while not have_result:
                            try:
                                result = lsqfit.nonlinear_fit(
                                    data=(x, ybs), prior=prior, p0=p0_bs, fcn=self.func, 
                                    maxit=100000, fitter=self.params['fitter'], svdcut=svdcut
                                )
                                have_result = True
                            except:
                                if svdcut <= 1e-4:
                                    svdcut = 10 * svdcut
                                else:
                                    sys.exit('we tried up to svdcut = %e' %svdcut)
                        '''
                if ndraws == 0:
                    self.plot.plot_result(result, subset)
                stats = dict()
                stats[(tuple(subset), "Q")]      = result.Q
                stats[(tuple(subset), "chi2")]   = result.chi2
                stats[(tuple(subset), "dof")]    = result.dof
                stats[(tuple(subset), "logGBF")] = result.logGBF
                if not self.params['bootstrap']:
                    stats[(tuple(subset), "prior")]  = result.prior
                    stats[(tuple(subset), 'svdcut')] = result.svdcut
                    stats[(tuple(subset), "svdn")]   = result.svdn
                    if self.params['verbose']:
                        print('-------------------------------------------------------')
                        print('-------------------------------------------------------')
                        print(subset)
                        print('-------------------------------------------------------')
                        print(result)
                # Change fit to record "result" instead of pieces of "result"
                #stats[(tuple(subset), "fit")]    = result

                posterior = {**posterior, **result.p, **stats}
                rbs = {**result.pmean, **stats}
                if nbs == 0:
                    rbs = {key: [rbs[key]] for key in rbs}
                    bsresult.update(rbs)
                else:
                    for key in rbs:
                        if key in bsresult:
                            bsresult[key].append(rbs[key])
                        else:
                            bsresult[key] = [rbs[key]]
                if nbs > 0:
                    ''' end of gvar scope used for bootstrap '''
                    gv.restore_gvar()

            posterior = self.reconstruct_gs(posterior)
            posterior = {"masterkey": self.d_sets, **posterior}
            if nbs == 0:
                self.posterior = posterior

        self.bsresult = bsresult

    def save(self):
        if not self.save_fit:
            return

        if self.params['bootstrap']:

            IrrepPath = f"{self.params['masterkey'][0][0][0]}{self.params['masterkey'][0][0][1]}{self.params['masterkey'][0][0][2]}"

            if not os.path.exists(f"./result/{IrrepPath}/{self.filename}_bs"):
                gv.dump(self.bsresult, f"./result/{IrrepPath}/{self.filename}_bs")
            else:
                bsresult = gv.load(f"./result/{IrrepPath}/{self.filename}_bs")
                for k in bsresult:
                    self.bsresult[k] = bsresult[k] + self.bsresult[k]
                gv.dump(self.bsresult, f"./result/{IrrepPath}/{self.filename}_bs")

            del self.bsresult
        else:

            IrrepPath = f"{self.params['masterkey'][0][0][0]}{self.params['masterkey'][0][0][1]}{self.params['masterkey'][0][0][2]}"

            if os.path.exists(f"./result/{IrrepPath}/{self.filename}"):
                os.remove(f"./result/{IrrepPath}/{self.filename}")
            print('saving result')
            gv.dump(self.posterior, f"./result/{IrrepPath}/{self.filename}")


    def get_bs_pickle_Nbs(self):
        IrrepPath = f"{self.params['masterkey'][0][0][0]}{self.params['masterkey'][0][0][1]}{self.params['masterkey'][0][0][2]}"

        if os.path.exists(f"./result/{IrrepPath}/{self.filename}_bs"):
            bsresult = gv.load(f"./result/{IrrepPath}/{self.filename}_bs")
            k = list(bsresult.keys())[0]
            Nbs=len(bsresult[k])-1 # first entry is boot0
        else:
            Nbs=0
        return Nbs

    def get_b0_posteriors(self):
        IrrepPath = f"{self.params['masterkey'][0][0][0]}{self.params['masterkey'][0][0][1]}{self.params['masterkey'][0][0][2]}"
        if os.path.exists(f"./result/{IrrepPath}/{self.boot0_file}"):
            return gv.load(f"./result/{IrrepPath}/{self.boot0_file}")
        else:
            print("can't get boot0, DOES NOT EXISTS: "+f"result/{IrrepPath}/{self.boot0_file}")
            sys.exit('run again with p["bootstrap"] = False')


def siground(x, sigfig=2):
    from math import log10, floor

    factor = 10 ** (sigfig - 1)
    return round(x, -int(floor(log10(abs(x / factor)))))


class Plot:
    def __init__(self, params):
        self.func = Functions(params)
        self.params = params

    def plot_result(self, fit, subset):
        for correlator in fit.x.keys():
            fig = plt.figure(f"{correlator} effective mass")
            ax = plt.axes()
            dx, dy = self.func.meff(fit.y[correlator], fit.x[correlator])
            fc = self.func(fit.x, fit.p)
            fx, fy = self.func.meff(fc[correlator], fit.x[correlator])
            ax.errorbar(x=dx, y=[i.mean for i in dy], yerr=[i.sdev for i in dy], linestyle='None')
            fy1 = np.array([i.mean for i in fy]) - np.array([i.sdev for i in fy])
            fy2 = np.array([i.mean for i in fy]) + np.array([i.sdev for i in fy])
            ax.fill_between(x=fx, y1=fy1, y2=fy2, alpha=0.5)
            if 2 in dx:
                xmin = 2
            else:
                xmin = dx[0]
            if 15 in dx:
                xmax = 15
            else:
                xmax = dx[-1]
            ax.set_xlim([xmin, dx[-1]])
            xfilter = np.arange(fx.index(xmin), fx.index(xmax))
            dy1 = np.array([i.mean for i in dy]) - np.array([i.sdev for i in dy])
            dy2 = np.array([i.mean for i in dy]) + np.array([i.sdev for i in dy])
            ymin = min(dy1[xfilter])
            ymax = max(dy2[xfilter])
            ax.set_ylim([ymin, ymax])
            if not os.path.exists('./plots/check_fits'):
                os.makedirs('./plots/check_fits')
            scorrelator = ''
            for k in correlator:
                if type(k) is tuple:
                    for sub_k in k:
                        scorrelator += '_' + str(sub_k)
                else:
                    scorrelator += '_' + str(k)
            scorrelator = scorrelator[1:]
            if correlator[1] == 'R':
                try:
                    trange = self.params['fit_choices'][correlator[0]]['trange']
                    rstates = self.params['fit_choices'][correlator[0]]['r_n_el']
                except:
                    if self.params["debug"]:
                        print('you have not specified a trange or r_n_el for %s' % str(correlator[0]))
                        print('we used the default params["r_n_el"] and params["trange"]["R"]\n')
                    trange = self.params['trange']['R']
                    rstates = self.params['r_n_el']
            else:
                trange = self.params['trange']['R']
                rstates = self.params['r_n_el']
            scorrelator += f"_N_n{self.params['nstates']}"
            scorrelator += f"_t_{self.params['trange']['N'][0]}_{self.params['trange']['N'][1]}"
            scorrelator += f"_R_n{rstates}"
            scorrelator += f"_t_{trange[0]}_{trange[1]}"
            plt.savefig(f"./plots/check_fits/meff_{scorrelator}.pdf", transparent=True)
            plt.close(fig)

    def simple_plot(self, data, tag, type="meff", x=None):
        try:
            gdata = gv.dataset.avg_data(data[tag])
        except:
            gdata = data[tag]

        xm, meff = self.func.meff(gdata, x)
        xc, corr = self.func.corr(gdata, x)
        if type == "meff":
            y = meff
            x = xm
        elif type == "corr":
            y = corr
            x = xc
        print(tag)
        stag = '_'.join([str(k) for k in tag])
        fig = plt.figure(f"meff {str(tag)}")
        ax = plt.axes()
        ax.errorbar(x=x, y=[i.mean for i in y], yerr=[i.sdev for i in y])
        if self.params['latex']:
            ltype = type.replace('_', '\\_')
            ltag = stag.replace('_', '\\_')
            plt.title(f"{ltype} {ltag}")
        else:
            plt.title(f"{type} {tag}")
        plt.draw()
        if not os.path.exists('plots/check_average'):
            os.makedirs('plots/check_average')

        plt.savefig(f"./plots/check_average/{type}_{stag}.pdf")
        plt.close(fig)

        x, zeff = self.func.zeff(gdata)
        fig = plt.figure(f"zeff {str(tag)}")
        ax = plt.axes()
        ax.errorbar(x=x, y=[i.mean for i in zeff], yerr=[i.sdev for i in zeff])
        plt.title(f"zeff {ltag}")
        plt.draw()
        plt.savefig(f"./plots/check_average/zeff_{stag}.pdf")
        plt.close(fig)


class Functions:
    def __init__(self, params):
        self.params   = params
        self.nstates  = params["nstates"]
        if self.params["version"]=="agnostic":
            try:
                self.r_n_inel = self.params['fit_choices'][key[0]]['r_n_inel']
            except:
                self.r_n_inel = self.params["r_n_inel"]
        self.r_n_el     = params["r_n_el"]
        self.positive_z = params["positive_z"]
        self.ratio      = params['ratio']

    def __call__(self, x, p):
        """
        x is a dictionary of {key: [time slices]}
        p is a dictionary of priors {(key, 'var'): gvar(m, s), ...}
        """
        rd = dict()
        for key in x.keys():
            if key[1] in ["R"]:
                rd[key] = self.pure_ratio(key, x[key], p)
            else:
                rd[key] = self.twopoint(key, x[key], p, "N")
        return rd

    def twopoint(self, key, x, p, tag):
        t    = np.array(x)
        r_es = self.twopoint_excited_states(key, x, p, tag)
        E0   = p[(key, "e0")]
        if not self.ratio and tag in ["R"]:
            key1 = (key[0],"N",key[2][0])
            key2 = (key[0],"N",key[2][1])
            E0  += p[(key1, f"e0")] + p[(key2, f"e0")]
        r = p[(key, "z0")] ** 2 * np.exp( -E0 * t) * r_es
        return r

    def twopoint_excited_states(self, key, x, p, tag):
        # r = 1 + sum_n r_n**2 exp( -dE_n t)
        t = np.array(x)
        r = 1

        # single nucleon
        if tag in ["N"]:
            for n in range(1,self.nstates):
                # En = E0 + sum_i=1^n DeltaE_i,i-1
                En = sum([np.exp(p[(key, f"e{ni}")]) for ni in range(1, n + 1)])
                r += p[(key, f"z{n}")] ** 2 * np.exp(-En * t)
        # two nucleon
        elif tag in ["R"]:
            # first add elastic excite states
            for n in range(1, 1 + self.r_n_el):
                En = sum([np.exp(p[(key, f"e_el{ni}")]) for ni in range(1, n + 1)])
                r += p[(key, f"z_el{n}")] ** 2 * np.exp(-En * t)
            # add inelastic excited states
            if self.params["version"] == "agnostic":
                for n in range(1, self.r_n_inel):
                    En = sum([np.exp(p[(key, f"e{ni}")]) for ni in range(1, n + 1)])
                    r += p[(key, f"z{n}")] ** 2 * np.exp(-En * t)

            elif self.params["version"] == "conspire":
                for n1 in range(self.nstates):
                    key1 = (key[0],"N",key[2][0])
                    En1  = sum([np.exp(p[(key1, f"e{ni}")]) for ni in range(1, n1 + 1)])

                    for n2 in range(self.nstates):
                        key2 = (key[0],"N",key[2][1])
                        En2  = sum([np.exp(p[(key2, f"e{ni}")]) for ni in range(1, n2 + 1)])

                        En = En1 + En2
                        if n1 == 0 and n2 == 0:
                            pass
                        else:
                            if n1 <= n2:
                                if not self.params['gs_conspire']:
                                    En += p[(key, f"e_{n1}_{n2}")]
                                r += p[(key, f"z_{n1}_{n2}")] ** 2 * np.exp(-En * t)
                            else:
                                if key2 == key1:
                                    if not self.params['gs_conspire']:
                                        En += p[(key, f"e_{n2}_{n1}")]
                                    r += p[(key, f"z_{n2}_{n1}")] ** 2 * np.exp(-En * t)
                                else:
                                    if not self.params['gs_conspire']:
                                        En += p[(key, f"e_{n1}_{n2}")]
                                    r += p[(key, f"z_{n1}_{n2}")] ** 2 * np.exp(-En * t)

        return r

    def pure_ratio(self, key, x, p):
        k0_key = (key[0], "N", key[2][0])
        k1_key = (key[0], "N", key[2][1])
        nn = self.twopoint(key, x, p, "R")
        if self.ratio:
            n0 = self.twopoint_excited_states(k0_key, x, p, "N")
            n1 = self.twopoint_excited_states(k1_key, x, p, "N")
        else:
            n0 = 1
            n1 = 1
        return nn / (n0 * n1)

    def corr(self, data, x=None):
        if x is None:
            data = data[2: len(data) * 2 // 2][:-1]
            x = range(2, len(data) + 2)
        else:
            data = data[:-1]
            x = x[:-1]
        return x, data

    def meff(self, data, x=None):
        if x is None:
            data = data[2: len(data) * 2 // 2]
            x = range(2, len(data) + 2)[:-1]
        else:
            x = x[:-1]
        y = np.log(data / np.roll(data, -1))[:-1]
        return x, y

    def zeff(self, data, x=None):
        _, corr = self.corr(data, x)
        x, meff = self.meff(data, x)
        y = corr * np.exp(meff * x)
        if self.positive_z:
            y = np.sqrt(y)
        return x, y


def main(param):
    #import IPython; IPython.embed() # insert this where you want to interact with fit in memory
    for d in ['result', 'figures']:
        if not os.path.exists(d):
            os.makedirs(d)
    fit = Fit(params=param)
    bs_p = param
    IrrepPath = f"{param['masterkey'][0][0][0]}{param['masterkey'][0][0][1]}{param['masterkey'][0][0][2]}"
    if bs_p['get_Zj']:
        if os.path.exists(f"result/{IrrepPath}/{bs_p['Zjn_values']}"):
            # these two lines are just to generate self.irreps
            fit.get_all_levels()
            fit.restore_masterkey()
            fit.read_Zjn()
            ratio_denom = fit.get_ratio_combinations_Zjn()
        else:
            fit.get_all_levels()
            fit.fit(n_start=0,ndraws=0)
            fit.compute_Zjn()
            fit.ratio_denom_Eff = dict(fit.ratio_denom)
            ratio_denom = fit.get_ratio_combinations_Zjn()
            # create Z_jn plots
            if fit.params['show_Zjn']:
                plt.ion()
            fit.report_ZjnSq()
            if fit.params['show_Zjn']:
                plt.ioff()
                plt.show()
            # change back to requested fit info
            fit.restore_masterkey()
        # change NN ref states to match those from Zjn values
        for k in ratio_denom:
            if ratio_denom[k] != fit.ratio_denom[k]:
                fit.ratio_denom[k] = ratio_denom[k]

    if not bs_p['bootstrap']:
        print('bs fits: boot0')
        fit.fit(n_start=0,ndraws=0)
        fit.save()

    else:
        bs_starts = bs_p['nbs_sub']* np.arange(bs_p['nbs']/bs_p['nbs_sub'],dtype=int)
        bs_finished = fit.get_bs_pickle_Nbs()
        for bs_start in bs_starts:
            if bs_start < bs_finished:
                print('bs fits:',bs_start+1,'->',bs_start+bs_p['nbs_sub'],'already done')
            else:
                print('bs fits:',bs_start+1,'->',bs_start+bs_p['nbs_sub'])
                fit.fit(n_start=bs_start,ndraws=bs_p['nbs_sub'])
                fit.save()

if __name__ == "__main__":
    #import IPython; IPython.embed() # insert this where you want to interact with fit in memory
    for d in ['result', 'figures']:
        if not os.path.exists(d):
            os.makedirs(d)
    fit = Fit()
    bs_p = parameters.params()
    if bs_p['get_Zj']:
        if os.path.exists(bs_p['Zjn_values']):
            # these two lines are just to generate self.irreps
            fit.get_all_levels()
            fit.restore_masterkey()
            fit.read_Zjn()
            ratio_denom = fit.get_ratio_combinations_Zjn()
        else:
            fit.get_all_levels()
            fit.fit(n_start=0,ndraws=0)
            fit.compute_Zjn()
            fit.ratio_denom_Eff = dict(fit.ratio_denom)
            ratio_denom = fit.get_ratio_combinations_Zjn()
            # create Z_jn plots
            if fit.params['show_Zjn']:
                plt.ion()
            fit.report_ZjnSq()
            if fit.params['show_Zjn']:
                plt.ioff()
                plt.show()
            # change back to requested fit info
            fit.restore_masterkey()
        # change NN ref states to match those from Zjn values
        for k in ratio_denom:
            if ratio_denom[k] != fit.ratio_denom[k]:
                fit.ratio_denom[k] = ratio_denom[k]

    if not bs_p['bootstrap']:
        print('bs fits: boot0')
        fit.fit(n_start=0,ndraws=0)
        fit.save()

    else:
        bs_starts = bs_p['nbs_sub']* np.arange(bs_p['nbs']/bs_p['nbs_sub'],dtype=int)
        bs_finished = fit.get_bs_pickle_Nbs()
        for bs_start in bs_starts:
            if bs_start < bs_finished:
                print('bs fits:',bs_start+1,'->',bs_start+bs_p['nbs_sub'],'already done')
            else:
                print('bs fits:',bs_start+1,'->',bs_start+bs_p['nbs_sub'])
                fit.fit(n_start=bs_start,ndraws=bs_p['nbs_sub'])
                fit.save()
    
