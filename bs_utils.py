#!/usr/bin/env python3
import sys
import numpy as np
import hashlib
from numpy.random import Generator, SeedSequence, PCG64


def get_rng(seed: str, verbose=False):
    """Generate a random number generator based on a seed string."""
    # Over python iteration the traditional hash was changed. So, here we fix it to md5
    hash = hashlib.md5(seed.encode("utf-8")).hexdigest()  # Convert string to a hash
    seed_int = int(hash, 16) % (10 ** 6)  # Convert hash to an fixed size integer
    if verbose:
        print("Seed to md5 hash:", seed, "->", hash, "->", seed_int)
    # Create instance of random number generator explicitly to ensure long time support
    # PCG64 -> https://www.pcg-random.org/
    # see https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
    rng = Generator(PCG64(SeedSequence(seed_int)))
    return rng

def make_bs_list(Ndata, Nbs, Mbs=None, seed=None, verbose=False, add_b0=True):
    if Mbs:
        m_bs = Mbs
    else:
        m_bs = Ndata

    # seed the random number generator
    rng = get_rng(seed,verbose=verbose) if seed else np.random.default_rng()

    # make BS list: [low, high)
    bs_list = rng.integers(low=0, high=Ndata, size=[Nbs, m_bs])

    # addpend boot0?
    if add_b0:
        if m_bs != Ndata:
            sys.exit('in order to insert boot0, Mbs must equal Ncfg: Ncfg=%d, Mbs=%d' %(NData,m_bs))
        bs_list = np.insert(bs_list, 0, np.arange(Ndata), axis=0)

    return bs_list

def bs_prior(Nbs, mean=0., sdev=1., seed=None, dist='normal'):
    ''' Generate bootstrap distribution of prior central values
        Args:
            Nbs  : number of values to return
            mean : mean of Gaussian distribution
            sdev : width of Gaussian distribution
            seed : string to seed random number generator
        Return:
            a numpy array of length Nbs of normal(mean, sdev) values
    '''
    # seed the random number generator
    rng = get_rng(seed) if seed else np.random.default_rng()

    if dist == 'normal':
        return rng.normal(loc=mean, scale=sdev, size=Nbs)
    elif dist == 'lognormal':
        return rng.lognormal(mean=mean, sigma=sdev, size=Nbs)
    else:
        sys.exit('you have not given a known distribution, %s' %dist)
