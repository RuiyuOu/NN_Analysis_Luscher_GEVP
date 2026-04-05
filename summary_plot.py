import matplotlib.pyplot as plt
import numpy as np
import os, sys

import gvar as gv

def summary_ENN(all_results, mN, all_lbls, colors, lbl0=None, spin='deuteron', fig='summary', format='pdf'):
    if spin == 'deuteron':
        irreps = {('0', 'T1g'):0,
                ('1', 'A2') :10, 
                ('1', 'E')  :20,
                ('2', 'A2') :30, 
                ('2', 'B1') :40,  
                ('2', 'B2') :50,
                ('3', 'A2') :60, 
                ('3', 'E')  :70,
                ('4', 'A2') :80, 
                ('4', 'E')  :90
                }
    elif spin == 'dineutron':
        irreps = {('0', 'A1g'):0,
                  ('1', 'A1') :10,
                  ('2', 'A1') :20,
                  ('3', 'A1') :30,
                  ('4', 'A1') :40
                  }
    if spin == 'deuteron':
        irrep_lbls = [
            r'$T_{1g}(0)$', r'$A_2(1)$', r'$E(1)$',
            r'$A_2(2)$', r'$B_1(2)$', r'$B_2(2)$',
            r'$A_2(3)$', r'$E(3)$', r'$A_2(4)$', r'$E(4)$'
        ]
    elif spin == 'dineutron':
        irrep_lbls = [
            r'$A_{1g}(0)$', r'$A_1(1)$', r'$A_1(2)$', r'$A_1(3)$', r'$A_1(4)$'
        ]
    #import IPython; IPython.embed()
    #plt.ion()
    plt.figure(fig, figsize=(6, 6/1.618))
    ax = plt.axes([.14,.10,.85,.885])
    plt.figure(fig+'_dElab', figsize=(6, 6/1.618))
    ax_dE = plt.axes([.165,.10,.83,.885])
    for k in all_results:
        irrep = (k.split('_')[0], k.split('_')[1])
        DE_i = np.array(all_results[k]['DE'])
        E1_i = np.array(all_results[k]['E1'])
        E2_i = np.array(all_results[k]['E2'])
        nsq  = int(k.split('_')[0])
        Psq  = nsq * (2*np.pi / 48)**2
        ENN  = DE_i + E1_i + E2_i
        EcmSq= ENN**2 - Psq
        Ecm  = np.sqrt(EcmSq)
        Ecm_mN = Ecm / mN.mean
        dE_lab = DE_i / mN.mean

        for i,e in enumerate(Ecm_mN):
            if irrep in [('0', 'T1g'), ('0', 'A1g')] and int(k.split('_')[-1]) == 0:
                lbl = str(all_lbls[i])
                if i == 0:
                    lbl = lbl0 + lbl
            else:
                lbl = None
            color = colors[all_lbls[i]]
            ax.errorbar(irreps[irrep]+i-len(Ecm_mN)/2, e.mean, yerr=e.sdev, 
                        marker='s', linestyle='None', mfc='None',
                        color=color, label=lbl)
            
            ax_dE.errorbar(irreps[irrep]+i-len(Ecm_mN)/2, dE_lab[i].mean, yerr=dE_lab[i].sdev, 
                        marker='s', linestyle='None', mfc='None',
                        color=color, label=lbl)
    ticks = [v for k,v in irreps.items()]
    ax.set_xticks(ticks, labels=irrep_lbls, fontsize=12)
    ax.legend(loc=1,fontsize=12,ncol=len(Ecm_mN), columnspacing=0,handletextpad=0.1)
    ax.set_ylabel(r'$E_{\rm cm} / m_N$', fontsize=16)
    ax.axhline(2, linestyle='--', color='k')
    ax.set_ylim(1.995,2.0601)
    ax.tick_params(axis='both', labelsize=14, direction='in')
    # padd the right lim.  Each irrep is spaced at 10, so add 2 to the end
    #xlim = ax.get_xlim()
    #ax.set_xlim(xlim[0],xlim[1]+2)


    plt.figure(fig)
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig('figures/'+fig+'.'+format, transparent=True)
    ax_dE.set_xticks(ticks, labels=irrep_lbls, fontsize=12)
    ax_dE.legend(loc=1,fontsize=12,ncol=len(Ecm_mN), columnspacing=0,handletextpad=0.1)
    ax_dE.set_ylabel(r'$\delta E_{00}^{\rm lab} / m_N$', fontsize=16)
    if spin == 'deuteron':
        ax_dE.set_ylim(-0.0105,0.0005)
    elif spin == 'dineutron':
        ax_dE.set_ylim(-0.0061,0.0005)
    ax_dE.tick_params(axis='both', labelsize=14, direction='in')
    # padd the right lim.  Each irrep is spaced at 10, so add 5 to the end
    #xlim = ax_dE.get_xlim()
    #ax_dE.set_xlim(xlim[0],xlim[1]+2)

    plt.figure(fig+'_dElab')
    plt.savefig('figures/'+fig+'_dElab'+'.'+format, transparent=True)

nnr_lim = {
    # deuteron
    ('0', 'T1g', 0):(-0.0021,0.0005), ('0', 'T1g', 1):(-0.0051,0.0005),
    ('1', 'A2', 0) :(-0.0046,0.0005), ('1', 'A2', 1) :(-0.0041,0.0005),
    ('2', 'A2', 0) :(-0.0066,0.0005), ('3', 'A2', 0) :(-0.0081,0.0005),
    ('4', 'A2', 0) :(-0.0021,0.0005), ('4', 'A2', 1) :(-0.0041,0.0005),
    ('2', 'B1', 0) :(-0.0061,0.0005), ('2', 'B2', 0) :(-0.0061,0.0005),
    ('2', 'B2', 3) :(-0.0056,0.0005), ('1', 'E', 0)  :(-0.0031,0.0005),
    ('1', 'E', 1)  :(-0.0061,0.0005), ('3', 'E', 0)  :(-0.0081,0.0005),
    ('4', 'E', 0)  :(-0.0021,0.0005), ('4', 'E', 1)  :(-0.0046,0.0005),
    # dineutron
    ('0', 'A1g', 0):(-0.0021,0.0005), ('0', 'A1g', 1):(-0.0036,0.0005),
    ('1', 'A1',  0):(-0.0031,0.0005), ('1', 'A1',  1):(-0.0036,0.0005),
    ('1', 'A1',  2):(-0.0021,0.0005),
    ('2', 'A1',  0):(-0.0046,0.0005), ('2', 'A1',  1):(-0.001,0.0005),
    ('2', 'A1',  2):(-0.0046,0.0005), ('2', 'A1',  3):(-0.002,0.00025),
    ('3', 'A1',  0):(-0.0046,0.0005), ('3', 'A1',  1):(-0.0051,0.0005),
    ('3', 'A1',  2):(-0.0046,0.0005),
    ('4', 'A1',  0):(-0.0021,0.0005), ('4', 'A1',  1):(-0.0036,0.0005),
}
nn_lim = {
    # deuteron
    ('0', 'T1g', 0):(1.4016,1.4170), ('0', 'T1g', 1):(1.4216,1.4360),
    ('1', 'A2', 0) :(1.4111,1.4255), ('1', 'A2', 1) :(1.4351,1.4495),
    ('2', 'A2', 0) :(1.4216,1.4370), ('3', 'A2', 0) :(1.4316,1.4470),
    ('4', 'A2', 0) :(1.4261,1.4395), ('4', 'A2', 1) :(1.4461,1.4605),
    ('2', 'B1', 0) :(1.4216,1.4370), ('2', 'B2', 0) :(1.4201,1.4345),
    ('2', 'B2', 3) :(1.4451,1.4595), ('1', 'E', 0)  :(1.4126,1.4270),
    ('1', 'E', 1)  :(1.4326,1.4470), ('3', 'E', 0)  :(1.4301,1.4445),
    ('4', 'E', 0)  :(1.4251,1.4395), ('4', 'E', 1)  :(1.4451,1.4595),
    # dineutron
    ('0', 'A1g', 0):(1.4016,1.4170), ('0', 'A1g', 1):(1.4216,1.4360),
    ('1', 'A1',  0):(1.4131,1.4275), ('1', 'A1',  1):(1.4351,1.4495),
    ('1', 'A1',  2):(1.4351,1.4551),
    ('2', 'A1',  0):(1.4216,1.4370), ('2', 'A1',  1):(1.4271,1.4415),
    ('2', 'A1',  2):(1.4271,1.4815), ('2', 'A1',  3):(1.4271,1.4815),
    ('3', 'A1',  0):(1.4351,1.4495), ('3', 'A1',  1):(1.4391,1.4535),
    ('3', 'A1',  2):(1.4391,1.4735),
    ('4', 'A1',  0):(1.4251,1.4395), ('4', 'A1',  1):(1.4461,1.4605),
}

def get_states(spin='deuteron', test=False, params=[]):
    if spin == 'deuteron':
        if len(params) != 1:
            states = [
                    ('0', 'T1g', 0), ('0', 'T1g', 1), 
                    ('1', 'A2', 0),  ('1', 'E', 0),
                    ('1', 'A2', 1),  ('1', 'E', 1),
                    ('2', 'A2', 0),  ('2', 'B1', 0),  ('2', 'B2', 0),
                    ('3', 'A2', 0),  
                    ('3', 'E', 0),
                    ('4', 'A2', 0),  ('4', 'E', 0),
                    ('4', 'A2', 1),  ('4', 'E', 1)
                ]
        else: 
            states = params
    
    elif spin == 'dineutron':
        # (1,A1,2), (2,A1,1), (2, A1, 2), (3,A1,1), (3, A1, 2), 
        states = [
                ("0", "A1g", 0), ("0", "A1g", 1),
                ("1", "A1", 0),  ("1", "A1", 1),
                ("2", "A1", 0),  #("2", "A1", 3),
                ("3", "A1", 0),  
                ("4", "A1", 0),  ("4", "A1", 1),
            ]
    else:
        sys.exit(f"unrecognized spin [deuteron, dineutron]: {spin}")
    if test:
        return [states[0]]
    else:
        return states
