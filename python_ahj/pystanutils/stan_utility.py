'''
see also
 https://github.com/betanalpha/jupyter_case_studies/blob/master/pystan_workflow/stan_utility.pya
'''

from hashlib import md5
import pickle

import numpy as np
import pystan

def check_div(fit):
    """ Check transitions that ended with a divergence """
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    n_divergent = np.int(np.sum(sp['divergent__'].sum() for sp in sampler_params))
    n_tot = np.sum(len(sp['divergent__']) for sp in sampler_params)

    print('{} of {} iterations ended with a divergence ({}%)'.format(n_divergent, n_tot, 100 * n_divergent/n_tot))

    if (n_divergent > 0):
        print('  Try running with larger adapt_delta to remove the divergences')



def check_treedepth(fit, max_depth = 10):
    """Check transitions that ended prematurely due to maximum tree depth limit"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    treedepths=list(sp['treedepth__'] for sp in sampler_params)
    n_deep = np.sum(np.sum(td==max_depth) for td in treedepths)
    n_tot = np.sum(len(td) for td in treedepths)

    print('{} of {} iterations saturated the maximum tree depth of {} ({}%)'.format(n_deep, n_tot, max_depth, 100 * n_deep/n_tot))
    if (n_deep > 0):
        print('  Run again with max_depth set to a larger value to avoid saturation')


def check_energy(fit):
    """Checks the energy Bayesian fraction of missing information (E-BFMI)"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    no_warning = True
    for n, sp in enumerate(sampler_params):
        energies = sp['energy__']
        numer = sum(np.diff(energies)**2) / len(energies)
        denom = np.var(energies)
        if (numer / denom < 0.2):
            print('Chain {}: E-BFMI = {}'.format(n, numer / denom))
            no_warning = True

    if no_warning:
        print('E-BFMI indicated no pathological behavior')
    else:
        print('  E-BFMI below 0.2 indicates you may need to reparameterize your model')


def check_n_eff(fit):
    """Checks the effective sample size per iteration"""
    fit_summary = fit.summary(probs=[0.5,])['summary']
    rownames = fit.summary(probs=[0.5,])['summary_rownames']
    n_iter = len(fit.extract()['lp__'])

    no_warning = True
    for n_eff, name in zip(fit_summary[:,4], rownames):
        ratio = n_eff / n_iter
        if (ratio < 0.001):
            print('n_eff / iter = {}/{} for parameter {} is {}!'.format(n_eff, iter, name, ratio))
            print('E-BFMI below 0.2 indicates you may need to reparameterize your model')
            no_warning = False
    if no_warning:
        print('n_eff / iter looks reasonable for all parameters')
    else:
        print('  n_eff / iter below 0.001 indicates that the effective sample size has likely been overestimated')


def check_rhat(fit):
    """Checks the potential scale reduction factors"""
    fit_summary = fit.summary(probs=[0.5,])['summary']
    rownames = fit.summary(probs=[0.5,])['summary_rownames']

    no_warning = True
    for rhat, name in zip(fit_summary[:,5], rownames):
        if rhat > 1.1 or np.isinf(rhat) or np.isnan(rhat):
            print('Rhat for parameter {} is {}!'.format(name, rhat))
            no_warning = False

    if no_warning:
        print('Rhat looks reasonable for all parameters')
    else:
        print('  Rhat above 1.1 indicates that the chains very likely have not mixed')


def check_all_diagnostics(fit):
    check_n_eff(fit)
    check_rhat(fit)
    check_div(fit)
    check_treedepth(fit)
    check_energy(fit)

def partition_div(fit):
    '''
        Returns parameter arrays separated into divergent and non-divergent transitions.
        Somewhat different from Betancourt's version which collects vector parameters into a single entry
        and (therefore) returns different shapes even for scalars
    '''

    nom_params = fit.extract(permuted=False)

    params = np.concatenate(nom_params.swapaxes(0,1))

    sampler_params = fit.get_sampler_params(inc_warmup=False)
    divergent = np.hstack(np.array([sp['divergent__'] for sp in sampler_params]))

    flatnames = fit.flatnames + ['lp__',]

    ### just make them dicts
    div_params =  dict(zip(flatnames, params[divergent == 1].transpose()))
    nondiv_params = dict(zip(flatnames, params[divergent == 0].transpose()))

    return (nondiv_params, div_params)


## could hard-code extra_compile_args as default?
## adapted from the pystan docs to use stanc_ret in addition to model_code
def StanModel_cache(model_code=None, stanc_ret=None, model_name=None, **kwargs):
    """Use just as you would `pystan.StanModel`"""
    if model_code is not None:
        code_hash = md5(model_code.encode('ascii')).hexdigest()

        if model_name is None:
            cache_fn = 'cached-model-{}.pkl'.format(code_hash)
        else:
            cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)

    elif stanc_ret is not None:
        cache_fn = stanc_ret['model_cppname']+'.pkl'

    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code, stanc_ret=stanc_ret, **kwargs)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm
