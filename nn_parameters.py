import numpy as np

def params():
    """
    User input file to control fitting options.

   """
    import numpy as np

    p = dict()
    p["debug"]   = False
    p["verbose"] = False
    p["latex"]   = True

    p["fpath"] = {"nucleon": "./data/cosmon_c103_r005-8_nucleon.hdf5", 
                  "nn": "./data/cosmon_c103_r005-8_deuteron_Swave.hdf5"}

    p["save"] = True

    p["fitter"] = 'scipy_least_squares'

    p["t0"] = 6
    p["td"] = 12
    p['t_norm'] = 3
    p['gevp']   = 'evp' # evp or gevp
    p['get_Zj'] = True
    if 'deuteron' in p["fpath"]["nn"]:
        p['Zjn_values'] = f"result/deuteron_Zjn_tNorm{p['t_norm']}_{p['gevp']}.h5"
    elif 'dineutron' in p["fpath"]["nn"]:
        p['Zjn_values'] = f"result/dineutron_Zjn_tNorm{p['t_norm']}_{p['gevp']}.h5"
    p['show_Zjn']   = False
    p['do_gevp']    = False #set to True if you want to do gevp if it was already done and saved

    p["block"] = 8
    #p['cfgs']  = [0,802[,1]] # use this to cut configs if desired 

    p['svd_study'] = False
    p['svdcut']    = 1e-8

    p["bootstrap"] = False
    p['Nbs_max']   = 5000
    p['bs_seed']   = 'nn_c103_b%d' %p["block"]
    p["nbs"]       = 5000
    p["nbs_sub"]   = 100
    p['bs0_width'] = 5
    p['bs_prior']  = 'all' # 'gs' or 'all': 
                          # randomize prior mean for gs or all priors
    #p['old_bs']    = True # set to True to use BS list from 2009.11825

    p["autotime"]   = 10 # time used to estimate mean gs energy prior
    p["sig_e0"]     = 1 # multiplication factor for meff[autotime] for prior width for deltaE_gs
    p["sig_enn"]    = 1 # multiplication factor for meff[autotime] for prior width for deltaE_nn
    p["positive_z"] = True

    p["ratio"]       = False
    p["ratio_type"]  = "data"
    p["irreps"]      = "irreps_ben" #["irreps", "irreps_ben"]
    p["version"]     = 'conspire'
    p["gs_conspire"] = False # only add deltaE for ground state?
    p["nstates"]     = 4
    p["r_n_inel"]    = 2
    p["r_n_el"]      = 0
    p["trange"]      = {"N": [7, 20], "R": [11, 15]}

    p["ampi"] = 0.310810
    p["amn"]  = 0.70262
    p["dE_elastic"] = 2 * np.sqrt(p["amn"]**2 + 1 * (2 * np.pi / 48) ** 2) -2*p["amn"]

    if 'deuteron' in p["fpath"]["nn"]:
        p["masterkey"] = [
            [("0", "T1g", 0)], [('0', 'T1g', 1)],
            [('1', 'A2', 0)], [('1', 'A2', 1)], 
            [('1', 'E', 0)], [('1', 'E', 1)], [('4', 'E', 0)], [('4', 'E', 1)],
            [('2', 'A2', 0)], [('4', 'A2', 0)], [('4', 'A2', 1)], 
            [('2', 'B1', 0)], [('2', 'B2', 0)], [('2', 'B2', 3)],
            [('3', 'A2', 0)], [('3', 'A2', 1)], [('3', 'E', 0)]
            ]

        #p["masterkey"] = [[("0", "T1g", 0)]] # modify to select single or other channels

    elif 'dineutron' in p["fpath"]["nn"]:
        p["masterkey"] = []
        for n in range(2):#6):
            p["masterkey"].append([("0", "A1g", n)])
        for n in range(3):#10):
            p["masterkey"].append([("1", "A1", n)])
        for n in range(6):#21):
            p["masterkey"].append([("2", "A1", n)])
        for n in range(3):#9):
            p["masterkey"].append([("3", "A1", n)])
        for n in range(4):#10):
            p["masterkey"].append([("4", "A1", n)])

    return p
