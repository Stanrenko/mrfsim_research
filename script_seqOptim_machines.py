
from utils_simu import *
from dictoptimizers import SimpleDictSearch
import json
from scipy.optimize import differential_evolution
import pickle
from datetime import datetime

path = r"/home/cslioussarenko/PythonRepositories"
#path = r"/Users/constantinslioussarenko/PythonGitRepositories/MyoMap"

import sys
sys.path.append(path+"/epgpy")
sys.path.append(path+"/machines")
sys.path.append(path+"/mutools")
sys.path.append(path+"/dicomstack")

from machines import machine, Toolbox, Config, set_parameter, set_output, printer, file_handler, Parameter, RejectException, get_context

DEFAULT_RANDOM_OPT_CONFIG="random_seqOptim_config.json"
DEFAULT_RANDOM_FA_OPT_CONFIG="random_FA_seqOptim_config.json"
DEFAULT_BREAKS_COMMON_OPT_CONFIG="breaks_common_seqOptim_config.json"




@machine
@set_parameter("optimizer_config",type=Config,default=DEFAULT_RANDOM_OPT_CONFIG,description="Optimizer parameters")
def optimize_sequence_random(optimizer_config):

    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")

    with open("./mrf_dictconf_SimReco2.json") as f:
        dict_config = json.load(f)

    fat_amp = np.array(dict_config["fat_amp"])
    fat_shift = -np.array(dict_config["fat_cshift"])

    optimizer_config["fat_amp"]=fat_amp
    optimizer_config["fat_shift"] = fat_shift

    H = optimizer_config["H"]

    bounds = [(0, 1)] * 4 * H + [(0, 4)] + [(15 * np.pi / 180, 70 * np.pi / 180)]

    maxiter = optimizer_config["maxiter"]

    log_cost = LogCost(lambda p:cost_function_simul_breaks_random(p,**optimizer_config))

    res = differential_evolution(lambda p:cost_function_simul_breaks_random(p,**optimizer_config), bounds=bounds, callback=log_cost,maxiter=maxiter)

    with open("./optims/res_simul_random_{}.pkl".format(date_time), "wb") as file:
        pickle.dump(res, file)

    np.save("./optims/log_cost_{}.npy".format(date_time),log_cost.cost_values)

    optimizer_config["fat_amp"] = list(fat_amp)
    optimizer_config["fat_shift"] = list(fat_shift)

    with open("./optims/random_opt_config_{}.json".format(date_time), "w") as file:
        json.dump(optimizer_config, file)

    return

@machine
@set_parameter("optimizer_config",type=Config,default=DEFAULT_RANDOM_FA_OPT_CONFIG,description="Optimizer parameters")
def optimize_sequence_random_FA(optimizer_config):

    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")

    with open("./mrf_dictconf_SimReco2.json") as f:
        dict_config = json.load(f)

    fat_amp = np.array(dict_config["fat_amp"])
    fat_shift = -np.array(dict_config["fat_cshift"])

    optimizer_config["fat_amp"]=fat_amp
    optimizer_config["fat_shift"] = fat_shift

    H = optimizer_config["H"]
    num_params_FA=2*H
    optimizer_config["num_params_FA"]=num_params_FA
    num_breaks_TE=optimizer_config["num_breaks_TE"]
    bound_min_TE=optimizer_config["bound_min_TE"]
    bound_max_TE = optimizer_config["bound_max_TE"]
    spokes_count = optimizer_config["spokes_count"]


    bounds = [(100, 400)] * (num_breaks_TE) + [(bound_min_TE, bound_max_TE)] * (num_breaks_TE + 1) + [(0, 1)] * num_params_FA + [
        (0, 4)] + [(15 * np.pi / 180, 70 * np.pi / 180)]

    from scipy.optimize import LinearConstraint
    A_TEbreaks = np.zeros((1, len(bounds)))
    A_TEbreaks[0, :num_breaks_TE] = 1
    con1 = LinearConstraint(A_TEbreaks, lb=-np.inf, ub=spokes_count - 100)
    constraints = (con1)

    maxiter = optimizer_config["maxiter"]

    log_cost = LogCost(lambda p:cost_function_simul_breaks_random_FA(p,**optimizer_config))

    res = differential_evolution(lambda p:cost_function_simul_breaks_random_FA(p,**optimizer_config), bounds=bounds, callback=log_cost,maxiter=maxiter,constraints=constraints)

    with open("./optims/res_simul_random_FA_{}.pkl".format(date_time), "wb") as file:
        pickle.dump(res, file)

    np.save("./optims/log_cost_{}.npy".format(date_time),log_cost.cost_values)

    optimizer_config["fat_amp"] = list(fat_amp)
    optimizer_config["fat_shift"] = list(fat_shift)

    with open("./optims/random_FA_opt_config_{}.json".format(date_time), "w") as file:
        json.dump(optimizer_config, file)

    return


@machine
@set_parameter("optimizer_config", type=Config, default=DEFAULT_BREAKS_COMMON_OPT_CONFIG,
               description="Optimizer parameters")
def optimize_sequence_breaks_common(optimizer_config):
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")

    with open("./mrf_dictconf_SimReco2.json") as f:
        dict_config = json.load(f)

    fat_amp = np.array(dict_config["fat_amp"])
    fat_shift = -np.array(dict_config["fat_cshift"])

    optimizer_config["fat_amp"] = fat_amp
    optimizer_config["fat_shift"] = fat_shift

    num_breaks_TE = optimizer_config["num_breaks_TE"]
    num_breaks_FA = optimizer_config["num_breaks_FA"]

    bound_min_TE = optimizer_config["bound_min_TE"]
    bound_max_TE = optimizer_config["bound_max_TE"]
    spokes_count=optimizer_config["spokes_count"]

    bounds = [(100, 400)] * (num_breaks_TE + num_breaks_FA) + [(bound_min_TE, bound_max_TE)] * (num_breaks_TE + 1) + [
        (5 * np.pi / 180, 70 * np.pi / 180)] * (num_breaks_FA + num_breaks_TE + 1) + [(0, 4)]

    from scipy.optimize import LinearConstraint
    A_TEbreaks = np.zeros((1, len(bounds)))
    A_TEbreaks[0, :(num_breaks_TE + num_breaks_FA)] = 1

    con1 = LinearConstraint(A_TEbreaks, lb=-np.inf, ub=spokes_count - 100)
    constraints = (con1)

    maxiter = optimizer_config["maxiter"]

    log_cost = LogCost(lambda p: cost_function_simul_breaks_common(p, **optimizer_config))

    res = differential_evolution(lambda p: cost_function_simul_breaks_common(p, **optimizer_config), bounds=bounds,
                                 callback=log_cost, maxiter=maxiter, constraints=constraints)

    with open("./optims/res_simul_breaks_common_{}.pkl".format(date_time), "wb") as file:
        pickle.dump(res, file)

    np.save("./optims/log_cost_{}.npy".format(date_time), log_cost.cost_values)

    optimizer_config["fat_amp"] = list(fat_amp)
    optimizer_config["fat_shift"] = list(fat_shift)

    with open("./optims/breaks_common_opt_config_{}.json".format(date_time), "w") as file:
        json.dump(optimizer_config, file)

    return


def cost_function_simul_breaks_random(params,**kwargs):
    # global result
    sigma = kwargs["sigma"]
    noise_type = kwargs["noise_type"]
    noise_size = kwargs["noise_size"]

    DFs = kwargs["DFs"]
    FFs = kwargs["FFs"]
    B1s = kwargs["B1s"]
    T1s = kwargs["T1s"]

    inversion = kwargs["inversion"]
    min_TR_delay = kwargs["min_TR_delay"]
    bound_min_TE = kwargs["bound_min_TE"]
    bound_max_TE = kwargs["bound_max_TE"]
    bound_min_FA = kwargs["bound_min_FA"]
    spokes_count = kwargs["spokes_count"]

    lambda_FA = kwargs["lambda_FA"]
    lambda_time = kwargs["lambda_time"]
    lambda_FF = kwargs["lambda_FF"]
    lambda_T1 = kwargs["lambda_T1"]
    lambda_stdT1 = kwargs["lambda_stdT1"]

    fat_amp = kwargs["fat_amp"]
    fat_shift = kwargs["fat_shift"]


    group_size = 8


    bound_max_FA = params[-1]
    params_for_curve = params[:-1]
    TR_, FA_, TE_ = convert_params_to_sequence_breaks_random(params_for_curve, min_TR_delay, spokes_count, bound_min_TE,
                                                             bound_max_TE, bound_min_FA, bound_max_FA, inversion)

    s, s_w, s_f, keys = simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs, T1s, 300 / 1000, B1s, T_2w=40 / 1000,
                                               T_2f=80 / 1000,
                                               amp=fat_amp, shift=fat_shift, sigma=sigma, noise_size=noise_size,
                                               noise_type=noise_type, group_size=group_size,
                                               return_fat_water=True)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

    s_w = s_w.reshape(s_w.shape[0], -1).T
    s_f = s_f.reshape(s_f.shape[0], -1).T
    # keys=keys.reshape(-1,4)
    # keys=[tuple(p) for p in keys]
    s = s.reshape(s.shape[0], -1)

    # plt.close("all")
    # plt.figure()
    # plt.plot(s_w[:len(DFs)].T)
    # plt.figure()
    # plt.plot(s_w[len(DFs):2*len(DFs)].T)

    nb_signals = s.shape[-1]
    mask = None
    pca = True
    threshold_pca_bc = 20

    split = 100
    dict_optim_bc_cf = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                        threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False,
                                        useGPU_simulation=False,
                                        gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                        return_matched_signals=True)

    all_maps, matched_signals = dict_optim_bc_cf.search_patterns_test((s_w, s_f, keys), s)

    # j=np.random.choice(range(matched_signals.shape[-1]))
    # plt.figure()
    # plt.plot(matched_signals[:,j])
    # plt.plot(s[:,j])

    keys_all = list(product(keys, FFs))
    # print(keys_all.shape)
    keys_all = [(*rest, a) for rest, a in keys_all]
    keys_all = np.array(keys_all)

    key = "wT1"
    map = all_maps[0][0][key].reshape(-1, noise_size)
    keys_all_current = np.array(list(keys_all[:, 0]) * noise_size).reshape(noise_size, -1).T
    error = np.abs(map - keys_all_current)
    error_wT1 = np.mean((np.mean(error, axis=-1)) / keys_all[:, 0])

    print("wT1 Cost : {}".format(error_wT1))

    std_wT1 = np.mean(np.std(error, axis=-1) / keys_all[:, 0])
    print("wT1 Std : {}".format(std_wT1))

    key = "ff"
    map = all_maps[0][0][key].reshape(-1, noise_size)
    keys_all_current = np.array(list(keys_all[:, -1]) * noise_size).reshape(noise_size, -1).T
    error = np.abs(map - keys_all_current)
    error_ff = np.mean((np.mean(error, axis=-1)))

    print("FF Cost : {}".format(error_ff))

    # num_breaks_TE=len(TE_breaks)
    FA_cost = np.mean((np.abs(np.diff(FA_[1:]))))
    # print("FA Cost : {}".format(FA_cost))

    time_cost = np.sum(TR_)
    print("Time Cost : {}".format(time_cost))
    # test_time=time_cost-TR_[-1]
    # print("Time Test : {}".format(test_time))

    result = lambda_T1 * error_wT1 + lambda_FF * error_ff + lambda_FA * FA_cost + lambda_time * time_cost + lambda_stdT1 * std_wT1

    return result


def cost_function_simul_breaks_random_FA(params,**kwargs):
    #global result

    group_size = 8

    sigma = kwargs["sigma"]
    noise_type = kwargs["noise_type"]
    noise_size = kwargs["noise_size"]

    DFs = kwargs["DFs"]
    FFs = kwargs["FFs"]
    B1s = kwargs["B1s"]
    T1s = kwargs["T1s"]

    inversion = kwargs["inversion"]
    min_TR_delay = kwargs["min_TR_delay"]
    bound_min_FA = kwargs["bound_min_FA"]
    num_breaks_TE = kwargs["num_breaks_TE"]
    num_params_FA = kwargs["num_params_FA"]

    spokes_count = kwargs["spokes_count"]

    lambda_FA = kwargs["lambda_FA"]
    lambda_time = kwargs["lambda_time"]
    lambda_FF = kwargs["lambda_FF"]
    lambda_T1 = kwargs["lambda_T1"]
    lambda_stdT1 = kwargs["lambda_stdT1"]

    fat_amp = kwargs["fat_amp"]
    fat_shift = kwargs["fat_shift"]


    bound_max_FA=params[-1]
    params_for_curve=params[:-1]
    TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,num_params_FA,bound_min_FA,bound_max_FA,inversion)

    s, s_w, s_f, keys = simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs, T1s, 300 / 1000,B1s, T_2w=40 / 1000, T_2f=80 / 1000,
                                               amp=fat_amp, shift=fat_shift, sigma=sigma, noise_size=noise_size,
                                               noise_type=noise_type, group_size=group_size,
                                               return_fat_water=True)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

    s_w = s_w.reshape(s_w.shape[0], -1).T
    s_f = s_f.reshape(s_f.shape[0], -1).T
    s = s.reshape(s.shape[0], -1)


    mask = None
    pca = True
    threshold_pca_bc = 20

    split = 100
    dict_optim_bc_cf = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                        threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False,
                                        useGPU_simulation=False,
                                        gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                        return_matched_signals=True)

    all_maps, matched_signals = dict_optim_bc_cf.search_patterns_test((s_w, s_f, keys), s)

    keys_all = list(product(keys, FFs))
    #print(keys_all.shape)
    keys_all = [(*rest, a) for rest, a in keys_all]
    keys_all = np.array(keys_all)


    key = "wT1"
    map = all_maps[0][0][key].reshape(-1, noise_size)
    keys_all_current = np.array(list(keys_all[:, 0]) * noise_size).reshape(noise_size, -1).T
    error = np.abs(map - keys_all_current)
    error_wT1 = np.mean((np.mean(error, axis=-1)) / keys_all[:, 0])

    print("wT1 Cost : {}".format(error_wT1))

    std_wT1 = np.mean(np.std(error, axis=-1)/ keys_all[:, 0])
    print("wT1 Std : {}".format(std_wT1))

    key = "ff"
    map = all_maps[0][0][key].reshape(-1, noise_size)
    keys_all_current = np.array(list(keys_all[:, -1]) * noise_size).reshape(noise_size, -1).T
    error = np.abs(map - keys_all_current)
    error_ff = np.mean((np.mean(error, axis=-1)))

    print("FF Cost : {}".format(error_ff))

    # num_breaks_TE=len(TE_breaks)
    FA_cost = np.mean((np.abs(np.diff(FA_[1:]))))
    # print("FA Cost : {}".format(FA_cost))

    time_cost = np.sum(TR_)
    print("Time Cost : {}".format(time_cost))
    #test_time=time_cost-TR_[-1]
    #print("Time Test : {}".format(test_time))

    result=lambda_T1 * error_wT1 + lambda_FF * error_ff + lambda_FA * FA_cost + lambda_time * time_cost+lambda_stdT1 * std_wT1

    return result

def cost_function_simul_breaks_common(params,**kwargs):





    sigma = kwargs["sigma"]
    noise_type = kwargs["noise_type"]
    noise_size = kwargs["noise_size"]

    DFs = kwargs["DFs"]
    FFs = kwargs["FFs"]
    B1s = kwargs["B1s"]
    T1s = kwargs["T1s"]

    inversion = kwargs["inversion"]
    min_TR_delay = kwargs["min_TR_delay"]
    num_breaks_TE = kwargs["num_breaks_TE"]
    num_breaks_FA = kwargs["num_breaks_FA"]

    spokes_count = kwargs["spokes_count"]

    lambda_FA = kwargs["lambda_FA"]
    lambda_time = kwargs["lambda_time"]
    lambda_FF = kwargs["lambda_FF"]
    lambda_T1 = kwargs["lambda_T1"]
    lambda_stdT1 = kwargs["lambda_stdT1"]

    fat_amp = kwargs["fat_amp"]
    fat_shift = kwargs["fat_shift"]

    group_size = 8

    # print(params)


    # print(params)
    # print(num_breaks_TE)
    # print(num_breaks_FA)
    TR_, FA_, TE_ = convert_params_to_sequence_breaks_common(params, min_TR_delay, num_breaks_TE, num_breaks_FA, spokes_count,
                                                      inversion)
    # print(FA_[:10])
    # print(params)

    s, s_w, s_f, keys = simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs, T1s, 300 / 1000,B1s, T_2w=40 / 1000, T_2f=80 / 1000,
                                               amp=fat_amp, shift=fat_shift, sigma=sigma, noise_size=noise_size,
                                               noise_type=noise_type, group_size=group_size,
                                               return_fat_water=True)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

    s_w = s_w.reshape(s_w.shape[0], -1).T
    s_f = s_f.reshape(s_f.shape[0], -1).T
    # keys=keys.reshape(-1,4)
    # keys=[tuple(p) for p in keys]
    s = s.reshape(s.shape[0], -1)

    nb_signals = s.shape[-1]
    mask = None
    pca = True
    threshold_pca_bc = 20

    split = nb_signals + 1
    dict_optim_bc_cf = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                        threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False,
                                        useGPU_simulation=False,
                                        gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                        return_matched_signals=True)

    all_maps, matched_signals = dict_optim_bc_cf.search_patterns_test((s_w, s_f, keys), s)

    # j=np.random.choice(range(matched_signals.shape[-1]))
    # plt.figure()
    # plt.plot(matched_signals[:,j])
    # plt.plot(s[:,j])

    keys_all = list(product(keys, FFs))
    #print(keys_all.shape)
    keys_all = [(*rest, a) for rest, a in keys_all]
    keys_all = np.array(keys_all)


    key = "wT1"
    map = all_maps[0][0][key].reshape(-1, noise_size)
    keys_all_current = np.array(list(keys_all[:, 0]) * noise_size).reshape(noise_size, -1).T
    error = np.abs(map - keys_all_current)
    error_wT1 = np.mean((np.mean(error, axis=-1)) / keys_all[:, 0])

    print("wT1 Cost : {}".format(error_wT1))

    std_wT1 = np.mean(np.std(error, axis=-1)/ keys_all[:, 0])
    print("wT1 Std : {}".format(std_wT1))

    key = "ff"
    map = all_maps[0][0][key].reshape(-1, noise_size)
    keys_all_current = np.array(list(keys_all[:, -1]) * noise_size).reshape(noise_size, -1).T
    error = np.abs(map - keys_all_current)
    error_ff = np.mean((np.mean(error, axis=-1)))

    print("FF Cost : {}".format(error_ff))

    # num_breaks_TE=len(TE_breaks)
    FA_cost = (np.linalg.norm(np.diff([0] + params[num_breaks_TE + 1 + num_breaks_TE + num_breaks_FA:])))
    # print("FA Cost : {}".format(FA_cost))

    time_cost = np.sum(TR_)
    print("Time Cost : {}".format(time_cost))

    return lambda_T1 * error_wT1 + lambda_FF * error_ff + lambda_FA * FA_cost + lambda_time * time_cost+lambda_stdT1 * std_wT1


class LogCost(object):
    def __init__(self,f):
        self.cost_values=[]
        self.it=1
        self.f=f

    def __call__(self,x,**kwargs):
        print("############# ITERATION {}###################".format(self.it))
        self.cost_values.append(self.f(x))
        self.it +=1

    def reset(self):
        self.cost_values = []
        self.it=1

toolbox = Toolbox("script_seqOptim_machines", description="MRF sequence optimization")
toolbox.add_program("optimize_sequence_random", optimize_sequence_random)
toolbox.add_program("optimize_sequence_random_FA", optimize_sequence_random_FA)
toolbox.add_program("optimize_sequence_breaks_common", optimize_sequence_breaks_common)


if __name__ == "__main__":
    toolbox.cli()
