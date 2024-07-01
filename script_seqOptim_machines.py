
from utils_simu import *
from dictoptimizers import SimpleDictSearch
import json
from scipy.optimize import differential_evolution
import pickle
from datetime import datetime
from image_series import *
from trajectory import *

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
DEFAULT_RANDOM_FA_US_OPT_CONFIG="random_FA_US_seqOptim_config.json"
DEFAULT_FF_SINGLE_FA_US_OPT_CONFIG="single_FA_US_seqOptim_config.json"
DEFAULT_RANDOM_FA_US_OPT_CONFIG_CORREL="random_FA_correl_seqOptim_config.json"
DEFAULT_RANDOM_FA_US_OPT_CONFIG_CRLB="random_FA_crlb_seqOptim_config.json"
DEFAULT_RANDOM_FA_US_VARSP_OPT_CONFIG="random_FA_US_varsp_seqOptim_config.json"

BUMP_WINDOWS={
    "wT1":[0,200],
    "ff":[0,0.5]
}

BUMP_STD={
    "wT1":10,
    "ff":0.01
}


@machine
@set_parameter("optimizer_config",type=Config,default=DEFAULT_RANDOM_OPT_CONFIG,description="Optimizer parameters")
def optimize_sequence_random(optimizer_config):

    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")

    with open("./mrf_dictconf_SimReco2.json") as f:
        dict_config = json.load(f)

    fat_amp = np.array(dict_config["fat_amp"])
    fat_shift = -np.array(dict_config["fat_cshift"])


    H = optimizer_config["H"]

    bounds = [(0, 1)] * 4 * H + [(0, 4)] + [(15 * np.pi / 180, 70 * np.pi / 180)]

    maxiter = optimizer_config["maxiter"]

    optimizer_config["fat_amp"] = list(fat_amp)
    optimizer_config["fat_shift"] = list(fat_shift)

    with open("./optims/random_opt_config_{}.json".format(date_time), "w") as file:
        json.dump(optimizer_config, file)

    log_cost = LogCost(lambda p:cost_function_simul_breaks_random(p,**optimizer_config),date_time=date_time)

    res = differential_evolution(lambda p:cost_function_simul_breaks_random(p,**optimizer_config), bounds=bounds, callback=log_cost,maxiter=maxiter)

    with open("./optims/res_simul_random_{}.pkl".format(date_time), "wb") as file:
        pickle.dump(res, file)

    np.save("./optims/log_cost_{}.npy".format(date_time),log_cost.cost_values)

    

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

    optimizer_config["fat_amp"] = list(fat_amp)
    optimizer_config["fat_shift"] = list(fat_shift)

    with open("./optims/random_FA_opt_config_{}.json".format(date_time), "w") as file:
        json.dump(optimizer_config, file)

    log_cost = LogCost(lambda p:cost_function_simul_breaks_random_FA(p,**optimizer_config),date_time=date_time)

    res = differential_evolution(lambda p:cost_function_simul_breaks_random_FA(p,**optimizer_config), bounds=bounds, callback=log_cost,maxiter=maxiter,constraints=constraints)

    with open("./optims/res_simul_random_FA_{}.pkl".format(date_time), "wb") as file:
        pickle.dump(res, file)

    np.save("./optims/log_cost_{}.npy".format(date_time),log_cost.cost_values)

    

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

    optimizer_config["fat_amp"] = list(fat_amp)
    optimizer_config["fat_shift"] = list(fat_shift)

    with open("./optims/breaks_common_opt_config_{}.json".format(date_time), "w") as file:
        json.dump(optimizer_config, file)

    log_cost = LogCost(lambda p: cost_function_simul_breaks_common(p, **optimizer_config),date_time=date_time)

    res = differential_evolution(lambda p: cost_function_simul_breaks_common(p, **optimizer_config), bounds=bounds,
                                 callback=log_cost, maxiter=maxiter, constraints=constraints)

    with open("./optims/res_simul_breaks_common_{}.pkl".format(date_time), "wb") as file:
        pickle.dump(res, file)

    np.save("./optims/log_cost_{}.npy".format(date_time), log_cost.cost_values)

    

    return



@machine
@set_parameter("optimizer_config",type=Config,default=DEFAULT_RANDOM_FA_US_OPT_CONFIG,description="Optimizer parameters")
def optimize_sequence_random_FA_undersampling(optimizer_config):

    global map_index
    global optimizer_config_for_cost

    map_index=0

    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")

    with open("./mrf_dictconf_SimReco2.json") as f:
        dict_config = json.load(f)

    fat_amp = np.array(dict_config["fat_amp"])
    fat_shift = -np.array(dict_config["fat_cshift"])

    optimizer_config["fat_amp"]=list(fat_amp)
    optimizer_config["fat_shift"] = list(fat_shift)

    H = optimizer_config["H"]
    num_params_FA=2*H
    optimizer_config["num_params_FA"]=num_params_FA
    num_breaks_TE=optimizer_config["num_breaks_TE"]
    bound_min_TE=optimizer_config["bound_min_TE"]
    bound_max_TE = optimizer_config["bound_max_TE"]
    spokes_count = optimizer_config["spokes_count"]

    bounds=[(100,400)]*(num_breaks_TE)+[(bound_min_TE,bound_max_TE)]*(num_breaks_TE+1)+[(0,1)]*num_params_FA+[(0,3)]+[(15*np.pi/180,70*np.pi/180)]

    from scipy.optimize import LinearConstraint
    A_TEbreaks = np.zeros((1, len(bounds)))
    A_TEbreaks[0, :num_breaks_TE] = 1
    con1 = LinearConstraint(A_TEbreaks, lb=-np.inf, ub=spokes_count - 100)
    constraints = (con1)

    maxiter = optimizer_config["maxiter"]

    num_maps=len(optimizer_config["file_phantom"])


    #log_cost = LogCost(lambda p:cost_function_simul_breaks_random_FA_KneePhantom(p,**optimizer_config),optimizer_config["dumpfile"],num_maps)

    


    #res = differential_evolution(lambda p:cost_function_simul_breaks_random_FA_KneePhantom(p,**optimizer_config), bounds=bounds, callback=log_cost,maxiter=maxiter,constraints=constraints,workers=24)

    optimizer_config_for_cost=optimizer_config

    
    with open("./optims/random_FA_opt_config_{}.json".format(date_time), "w") as file:
        json.dump(optimizer_config, file)
    
    log_cost = LogCost(cost_function,optimizer_config["dumpfile"],num_maps,date_time=date_time)
    res = differential_evolution(cost_function, bounds=bounds, callback=log_cost,maxiter=maxiter,constraints=constraints,workers=24)

    with open("./optims/res_simul_random_FA_US_{}.pkl".format(date_time), "wb") as file:
        pickle.dump(res, file)

    np.save("./optims/log_cost_{}.npy".format(date_time),log_cost.cost_values)



    return



@machine
@set_parameter("optimizer_config",type=Config,default=DEFAULT_RANDOM_FA_US_VARSP_OPT_CONFIG,description="Optimizer parameters")
@set_parameter("workers",int,default=24,description="Number of workers for differential evolution")
def optimize_sequence_random_FA_undersampling_variablesp(optimizer_config,workers):

    global map_index
    global optimizer_config_for_cost

    map_index=0

    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")

    with open("./mrf_dictconf_SimReco2.json") as f:
        dict_config = json.load(f)

    fat_amp = np.array(dict_config["fat_amp"])
    fat_shift = -np.array(dict_config["fat_cshift"])

    optimizer_config["fat_amp"]=list(fat_amp)
    optimizer_config["fat_shift"] = list(fat_shift)

    H = optimizer_config["H"]
    num_params_FA=2*H
    optimizer_config["num_params_FA"]=num_params_FA
    num_breaks_TE=optimizer_config["num_breaks_TE"]
    bound_min_TE=optimizer_config["bound_min_TE"]
    bound_max_TE = optimizer_config["bound_max_TE"]
    #spokes_count = optimizer_config["spokes_count"]

    bounds=[(0.1,0.9)]*(num_breaks_TE)+[(bound_min_TE,bound_max_TE)]*(num_breaks_TE+1)+[(0,1)]*num_params_FA+[(0,3)]+[(15*np.pi/180,70*np.pi/180)]+[(0,1)]

    # from scipy.optimize import LinearConstraint
    # A_TEbreaks = np.zeros((1, len(bounds)))
    # A_TEbreaks[0, :num_breaks_TE] = 1
    # con1 = LinearConstraint(A_TEbreaks, lb=-np.inf, ub=spokes_count - 100)
    # constraints = ()

    maxiter = optimizer_config["maxiter"]

    num_maps=len(optimizer_config["file_phantom"])


    #log_cost = LogCost(lambda p:cost_function_simul_breaks_random_FA_KneePhantom(p,**optimizer_config),optimizer_config["dumpfile"],num_maps)

    


    #res = differential_evolution(lambda p:cost_function_simul_breaks_random_FA_KneePhantom(p,**optimizer_config), bounds=bounds, callback=log_cost,maxiter=maxiter,constraints=constraints,workers=24)

    optimizer_config_for_cost=optimizer_config

    with open("./optims/random_FA_opt_config_{}.json".format(date_time), "w") as file:
        json.dump(optimizer_config, file)
    
    log_cost = LogCost(cost_function,optimizer_config["dumpfile"],num_maps,date_time=date_time)
    res = differential_evolution(cost_function, bounds=bounds, callback=log_cost,maxiter=maxiter,constraints=(),workers=workers)

    with open("./optims/res_simul_random_FA_US_{}.pkl".format(date_time), "wb") as file:
        pickle.dump(res, file)

    np.save("./optims/log_cost_{}.npy".format(date_time),log_cost.cost_values)


    

    return





@machine
@set_parameter("optimizer_config",type=Config,default=DEFAULT_FF_SINGLE_FA_US_OPT_CONFIG,description="Optimizer parameters")
def optimize_sequence_FF_single_FA_undersampling(optimizer_config):

    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")

    with open("./mrf_dictconf_SimReco2.json") as f:
        dict_config = json.load(f)

    fat_amp = np.array(dict_config["fat_amp"])
    fat_shift = -np.array(dict_config["fat_cshift"])


    
    num_breaks_TE=optimizer_config["num_breaks_TE"]
    bound_min_TE=optimizer_config["bound_min_TE"]
    bound_max_TE = optimizer_config["bound_max_TE"]
    spokes_count = optimizer_config["spokes_count"]




    bounds=[(50,100)]*(num_breaks_TE)+[(bound_min_TE,bound_max_TE)]*(num_breaks_TE+1)+[(5*np.pi/180,70*np.pi/180)]

    from scipy.optimize import LinearConstraint
    A_TEbreaks = np.zeros((1, len(bounds)))
    A_TEbreaks[0, :num_breaks_TE] = 1
    con1 = LinearConstraint(A_TEbreaks, lb=-np.inf, ub=spokes_count - 50)
    constraints = (con1)

    maxiter = optimizer_config["maxiter"]

    optimizer_config["fat_amp"] = list(fat_amp)
    optimizer_config["fat_shift"] = list(fat_shift)

    with open("./optims/single_FA_opt_config_{}.json".format(date_time), "w") as file:
        json.dump(optimizer_config, file)

    log_cost = LogCost(lambda p:cost_function_simul_breaks_single_FA_FF_KneePhantom(p,**optimizer_config),optimizer_config["dumpfile"],date_time=date_time)

    res = differential_evolution(lambda p:cost_function_simul_breaks_single_FA_FF_KneePhantom(p,**optimizer_config), bounds=bounds, callback=log_cost,maxiter=maxiter,constraints=constraints)

    with open("./optims/res_simul_single_FA_US_{}.pkl".format(date_time), "wb") as file:
        pickle.dump(res, file)

    np.save("./optims/log_cost_{}.npy".format(date_time),log_cost.cost_values)

    

    

    return

@machine
@set_parameter("optimizer_config",type=Config,default=DEFAULT_RANDOM_FA_US_OPT_CONFIG_CORREL,description="Optimizer parameters")
def optimize_sequence_random_FA_correl(optimizer_config):

    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")

    with open("./mrf_dictconf_SimReco2.json") as f:
        dict_config = json.load(f)

    fat_amp = np.array(dict_config["fat_amp"])
    fat_shift = -np.array(dict_config["fat_cshift"])

    optimizer_config["fat_amp"]=list(fat_amp)
    optimizer_config["fat_shift"] = list(fat_shift)

    H = optimizer_config["H"]
    num_params_FA=2*H
    optimizer_config["num_params_FA"]=num_params_FA
    num_breaks_TE=optimizer_config["num_breaks_TE"]
    bound_min_TE=optimizer_config["bound_min_TE"]
    bound_max_TE = optimizer_config["bound_max_TE"]
    spokes_count = optimizer_config["spokes_count"]
    max_reco = optimizer_config["max_reco"]



    bounds=[(100,400)]*(num_breaks_TE)+[(bound_min_TE,bound_max_TE)]*(num_breaks_TE+1)+[(0,1)]*num_params_FA+[(0,max_reco)]+[(15*np.pi/180,70*np.pi/180)]

    from scipy.optimize import LinearConstraint
    A_TEbreaks = np.zeros((1, len(bounds)))
    A_TEbreaks[0, :num_breaks_TE] = 1
    con1 = LinearConstraint(A_TEbreaks, lb=-np.inf, ub=spokes_count - 100)
    constraints = (con1)

    maxiter = optimizer_config["maxiter"]

    with open("./optims/random_FA_opt_config_{}.json".format(date_time), "w") as file:
        json.dump(optimizer_config, file)

    log_cost = LogCost(lambda p:cost_function_simul_breaks_random_FA_correl(p,**optimizer_config),optimizer_config["dumpfile"],date_time=date_time)

    res = differential_evolution(lambda p:cost_function_simul_breaks_random_FA_correl(p,**optimizer_config), bounds=bounds, callback=log_cost,maxiter=maxiter,constraints=constraints)

    with open("./optims/res_simul_random_FA_US_{}.pkl".format(date_time), "wb") as file:
        pickle.dump(res, file)

    np.save("./optims/log_cost_{}.npy".format(date_time),log_cost.cost_values)


    

    return


@machine
@set_parameter("optimizer_config",type=Config,default=DEFAULT_RANDOM_FA_US_OPT_CONFIG_CRLB,description="Optimizer parameters")
def optimize_sequence_random_FA_crlb(optimizer_config):

    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")

    with open("./mrf_dictconf_SimReco2.json") as f:
        dict_config = json.load(f)

    fat_amp = np.array(dict_config["fat_amp"])
    fat_shift = -np.array(dict_config["fat_cshift"])

    optimizer_config["fat_amp"]=list(fat_amp)
    optimizer_config["fat_shift"] = list(fat_shift)

    H = optimizer_config["H"]
    num_params_FA=2*H
    optimizer_config["num_params_FA"]=num_params_FA
    num_breaks_TE=optimizer_config["num_breaks_TE"]
    bound_min_TE=optimizer_config["bound_min_TE"]
    bound_max_TE = optimizer_config["bound_max_TE"]
    spokes_count = optimizer_config["spokes_count"]
    max_reco = optimizer_config["max_reco"]



    bounds=[(100,400)]*(num_breaks_TE)+[(bound_min_TE,bound_max_TE)]*(num_breaks_TE+1)+[(0,1)]*num_params_FA+[(0,max_reco)]+[(15*np.pi/180,70*np.pi/180)]

    from scipy.optimize import LinearConstraint
    A_TEbreaks = np.zeros((1, len(bounds)))
    A_TEbreaks[0, :num_breaks_TE] = 1
    con1 = LinearConstraint(A_TEbreaks, lb=-np.inf, ub=spokes_count - 100)
    constraints = (con1)

    maxiter = optimizer_config["maxiter"]

    with open("./optims/random_FA_opt_config_{}.json".format(date_time), "w") as file:
        json.dump(optimizer_config, file)

    log_cost = LogCost(lambda p:cost_function_simul_breaks_random_FA_crlb(p,**optimizer_config),optimizer_config["dumpfile"],date_time=date_time)

    res = differential_evolution(lambda p:cost_function_simul_breaks_random_FA_crlb(p,**optimizer_config), bounds=bounds, callback=log_cost,maxiter=maxiter,constraints=constraints)

    with open("./optims/res_simul_random_FA_US_{}.pkl".format(date_time), "wb") as file:
        pickle.dump(res, file)

    np.save("./optims/log_cost_{}.npy".format(date_time),log_cost.cost_values)


    

    return




@machine
@set_parameter("ts",str,default=None,description="Optimization timestamp")
@set_parameter("proportional",bool,default=True,description="TE breaks definition mode")
@set_parameter("suffix",str,default="",description="Suffix to identify optimized sequence")
@set_parameter("dictconf",str,default="mrf_dictconf_Dico2_Invivo.json",description="Dictionary grid")
@set_parameter("dictconf_light",str,default="mrf_dictconf_Dico2_Invivo_light_for_matching.json",description="Light dictionary grid")

def generate_dico_and_files(ts,proportional,suffix,dictconf,dictconf_light):

    file_params="./optims/res_simul_random_FA_US_{}.pkl".format(ts)
    file_config="./optims/random_FA_opt_config_{}.json".format(ts)
    file_cost="./optims/log_cost_{}.npy".format(ts)

    log_cost=np.load(file_cost)
    plt.figure()
    plt.plot(log_cost)
    plt.savefig("./optims/log_cost_{}.jpg".format(ts))

    with open(file_config, "rb") as file:
        config=json.load(file)

    with open(file_params, "rb") as file:
        res=pickle.load(file)

    x=res.x

    bound_min_FA=config["bound_min_FA"]
    
    num_breaks_TE=config["num_breaks_TE"]
    H=config["H"]
    num_params_FA=2*H
    min_TR_delay=2.22*10**-3

    if "spokes_count" not in config.keys():
        spokes_count = 520 + 8 * int(x[-1] * (1400 - 520) / 8)
        bound_max_FA = x[-2]
        params_for_curve = x[:-2]
        reco=x[-3]
    else:
        spokes_count=config["spokes_count"]
        bound_max_FA = x[-1]
        params_for_curve = x[:-1]
        reco=x[-2]

    inversion=config["inversion"]
    optim=config["optim"]


    sequence_file="./mrf_sequence_random_FA_{}{}_allparams.json".format(optim,suffix)
    #file_params="x_random_FA_US_H4_variablespokecount_log_time0_01_allparams.pkl"



    if proportional:
        TR_, FA_, TE_ = convert_params_to_sequence_breaks_proportional_random_FA(params_for_curve, min_TR_delay, spokes_count,
                                                                            num_breaks_TE, num_params_FA, bound_min_FA,
                                                                            bound_max_FA, inversion)
    
    else:
        TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,num_params_FA,bound_min_FA,bound_max_FA,inversion)
    plt.close("all")
    plt.figure()
    plt.plot(TE_[1:])
    plt.savefig("./optims/TE_{}.jpg".format(ts))

    plt.figure()
    plt.plot(FA_[1:])
    plt.savefig("./optims/FA_{}.jpg".format(ts))


    


    
    new_sequence_file=str.split(sequence_file,".json")[0]+"_{}.json".format(np.round(min_TR_delay*1000,2))
    print(new_sequence_file)

    generate_epg_dico_T1MRFSS(new_sequence_file,dictconf,FA_,TE_,np.round(reco,2),min_TR_delay,rep=3)
    generate_epg_dico_T1MRFSS(new_sequence_file,dictconf_light,FA_,TE_,np.round(reco,2),min_TR_delay,rep=3)

    with open(new_sequence_file) as f:
        sequence_config = json.load(f)

    TE=sequence_config["TE"]
    TE=np.array(TE)

    #TE=np.maximum(TE,2.2)
    TE=np.round(TE,2)
    #TE=np.maximum(np.round(TE,2),2.2)

    TE_file="TE_temp_{}.text".format(ts)

    with open(TE_file,"w") as file:
        for te in TE:
            file.write(str(te)+"\n")

    #pd.DataFrame(np.maximum(np.round(TE,2),2.2)).to_clipboard()

    FA=sequence_config["B1"]

    FA=np.array(FA)
    FA=np.round(FA,3)
    #pd.DataFrame(np.round(FA,3)).to_clipboard(excel=False,index=False)

    FA_file="FA_temp_{}.text".format(ts)

    with open(FA_file,"w") as file:
        for fa in FA:
            file.write(str(fa)+"\n")
    print(np.min(np.array(TE)))
    print(np.max(np.array(TE)))
    print(np.min(np.array(FA)))
    print(np.max(np.array(FA)))

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

    fat_amp = np.array(kwargs["fat_amp"])
    fat_shift = np.array(kwargs["fat_shift"])


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

    fat_amp = np.array(kwargs["fat_amp"])
    fat_shift = np.array(kwargs["fat_shift"])


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

    fat_amp = np.array(kwargs["fat_amp"])
    fat_shift = np.array(kwargs["fat_shift"])

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



def cost_function_simul_breaks_random_FA_KneePhantom(params,**kwargs):
    #global result

    global map_index

    DFs = kwargs["DFs"]
    FFs = None
    B1s = kwargs["B1s"]
    T1s = kwargs["T1s"]

    inversion = kwargs["inversion"]
    min_TR_delay = kwargs["min_TR_delay"]
    bound_min_FA = kwargs["bound_min_FA"]
    num_breaks_TE = kwargs["num_breaks_TE"]
    num_params_FA = kwargs["num_params_FA"]

    spokes_count = kwargs["spokes_count"]
    npoint = kwargs["npoint"]

    lambda_FA = kwargs["lambda_FA"]
    lambda_time = kwargs["lambda_time"]
    lambda_FF = kwargs["lambda_FF"]
    lambda_T1 = kwargs["lambda_T1"]
    lambda_DF = kwargs["lambda_DF"]
    lambda_B1 = kwargs["lambda_B1"]

    lambda_stdT1 = kwargs["lambda_stdT1"]

    include_flat_part=kwargs["include_flat"]

    fat_amp = np.array(kwargs["fat_amp"])
    fat_shift = np.array(kwargs["fat_shift"])

    useGPU=kwargs["useGPU"]

    image_size = (int(npoint / 2), int(npoint / 2))


    file = kwargs["file_phantom"][map_index]

    m_ = MapFromFile("Phantom_Optim", file=file, rounding=True)


    m_.buildParamMap()


    start_time=datetime.now()
    group_size = 8


    bound_max_FA=params[-1]
    params_for_curve=params[:-1]
    TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,num_params_FA,bound_min_FA,bound_max_FA,inversion)

    if include_flat_part:
        print("Include flat part")
        nb_flat_spokes=kwargs["nb_flat"]
        FA_=FA_+[6*np.pi/180]*nb_flat_spokes
        #print("FA_ {}".format(FA_[-1]))
        TE_=TE_+[3.45/1000]*nb_flat_spokes
        #print("TE {}".format(TE_[-1]))
        TR_[1:] = np.array(TE_[1:]) + min_TR_delay
        TR_[-1] = TR_[-1] + params_for_curve[-1]
        TR_ = list(TR_)
        #print("TR {}".format(TR_[-1]))
        spokes_count += nb_flat_spokes

    #print(spokes_count)
    radial_traj = Radial(total_nspokes=spokes_count, npoint=npoint)
    ntimesteps = int(spokes_count / group_size)

    m_.build_ref_images_bloch(TR_,FA_,TE_)

    s, s_w, s_f, keys = simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs, T1s, 300 / 1000,B1s, T_2w=40 / 1000, T_2f=80 / 1000,
                                               amp=fat_amp, shift=fat_shift, sigma=None, group_size=group_size,
                                               return_fat_water=True,return_combined_signal=False)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

    s_w = s_w.reshape(s_w.shape[0], -1).T
    s_f = s_f.reshape(s_f.shape[0], -1).T

    data = m_.generate_kdata(radial_traj, useGPU=useGPU,nthreads=1,fftw=0)
    data = np.array(data)
    data = data.reshape(spokes_count, -1, npoint)
    volumes_all = simulate_radial_undersampled_images(data, radial_traj, image_size, density_adj=True, useGPU=useGPU,ntimesteps=ntimesteps,nthreads=1,fftw=0)

    # from PIL import Image
    # gif=[]
    # volume_for_gif = np.abs(volumes_all)
    # for i in range(volume_for_gif.shape[0]):
    #     img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
    #     img=img.convert("P")
    #     gif.append(img)

    # filename_gif = "test_US_optim.gif"
    # gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)


    mask = m_.mask
    pca = True
    threshold_pca_bc = 10

    split = 100
    dict_optim_bc_cf = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                        threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False,
                                        useGPU_simulation=False,
                                        gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                        return_matched_signals=True)

    all_maps, matched_signals = dict_optim_bc_cf.search_patterns_test((s_w, s_f, keys), volumes_all)

    #print(all_maps)
    #print(matched_signals.shape)



    key = "wT1"
    map = all_maps[0][0][key][m_.paramMap["ff"]<0.7]*1000
    map_gt=m_.paramMap[key][m_.paramMap["ff"]<0.7]
    error = np.abs(map - map_gt)
    error_wT1 = np.mean(error/ map_gt)

    print("wT1 Cost : {}".format(error_wT1))

    std_wT1 = np.std(error/ map_gt)
    print("wT1 Std : {}".format(std_wT1))

    key = "ff"
    map = all_maps[0][0][key]
    map_gt = m_.paramMap[key]
    error = np.abs(map - map_gt)
    error_ff = np.mean(error)

    print("FF Cost : {}".format(error_ff))

    key = "df"
    map = all_maps[0][0][key]/1000
    map_gt = m_.paramMap[key]
    error = np.abs(map - map_gt)
    error_df = np.mean(error)

    print("DF Cost : {}".format(error_df))

    key = "attB1"
    map = all_maps[0][0][key]
    map_gt = m_.paramMap[key]
    error = np.abs(map - map_gt)
    error_b1 = np.mean(error)

    print("B1 Cost : {}".format(error_b1))

    # num_breaks_TE=len(TE_breaks)
    FA_cost = np.mean((np.abs(np.diff(FA_[1:]))))
    # print("FA Cost : {}".format(FA_cost))

    time_cost = np.sum(TR_)
    print("Time Cost : {}".format(time_cost))
    #test_time=time_cost-TR_[-1]
    #print("Time Test : {}".format(test_time))

    result=lambda_T1 * error_wT1 + lambda_FF * error_ff + lambda_DF * error_df + lambda_B1 * error_b1 +lambda_FA * FA_cost + lambda_time * time_cost+lambda_stdT1 * std_wT1

    end_time=datetime.now()
    print(end_time-start_time)

    return result



def cost_function_simul_breaks_random_FA_KneePhantom_variablesp(params,**kwargs):
    #global result

    global map_index

    DFs = kwargs["DFs"]
    FFs = None
    B1s = kwargs["B1s"]
    T1s = kwargs["T1s"]

    DFs_light = kwargs["DFs_light"]
    B1s_light = kwargs["B1s_light"]
    T1s_light = kwargs["T1s_light"]

    inversion = kwargs["inversion"]
    min_TR_delay = kwargs["min_TR_delay"]
    bound_min_FA = kwargs["bound_min_FA"]
    num_breaks_TE = kwargs["num_breaks_TE"]
    num_params_FA = kwargs["num_params_FA"]


    lambda_FA = kwargs["lambda_FA"]
    lambda_time = kwargs["lambda_time"]
    lambda_FF = kwargs["lambda_FF"]
    lambda_T1 = kwargs["lambda_T1"]
    lambda_DF = kwargs["lambda_DF"]
    lambda_B1 = kwargs["lambda_B1"]

    lambda_stdT1 = kwargs["lambda_stdT1"]

    include_flat_part=kwargs["include_flat"]

    fat_amp = np.array(kwargs["fat_amp"])
    fat_shift = np.array(kwargs["fat_shift"])
    compression_factor=kwargs["compression"]
    sigma=kwargs["sigma"]

    useGPU=kwargs["useGPU"]


    file = kwargs["file_phantom"][map_index]

    m_ = MapFromFile("Phantom_Optim", file=file, rounding=True)


    if ("maskROI" in kwargs.keys())and(not(kwargs["maskROI"]=="")):
        print("Randomizing parameter values inside each numerical phantom")
        maskROI=np.load(kwargs["maskROI"])[map_index]
        #print(maskROI.shape)
        ROI_bumped_count=kwargs["ROI_count"]
        num_ROIs=len(np.unique(maskROI))
        bumped_ROIs=np.random.choice(num_ROIs,ROI_bumped_count)
        bumped_ROIs=np.unique(bumped_ROIs)
        #print(bumped_ROIs)
        dico_bumps={}
        for k in BUMP_WINDOWS.keys():
            print(k)
            bump_values=np.zeros(shape=maskROI.shape)
            for roi in bumped_ROIs:
                if not (roi == 0):
                    shape=maskROI[maskROI==roi].shape
                    bump_values[maskROI==roi]=np.random.uniform(low=BUMP_WINDOWS[k][0],high=BUMP_WINDOWS[k][1])+np.random.normal(0,BUMP_STD[k],size=shape)
                    #print(bump_values[maskROI==roi])
            dico_bumps[k]=bump_values

        #print(dico_bumps)





    else:
        dico_bumps=None

    m_.buildParamMap(dico_bumps=dico_bumps)
    if compression_factor>1:
        m_.change_resolution(compression_factor)
    npoint_image=m_.image_size[-1]

    npoint = npoint_image*2
    image_size = (npoint_image, npoint_image)


    start_time=datetime.now()
    group_size = 8

    spokes_count=520+8*int(params[-1]*(1400-520)/8)
    bound_max_FA=params[-2]
    params_for_curve=params[:-2]
    TR_, FA_, TE_ = convert_params_to_sequence_breaks_proportional_random_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,num_params_FA,bound_min_FA,bound_max_FA,inversion)

    if include_flat_part:
        print("Include flat part")
        nb_flat_spokes=kwargs["nb_flat"]
        FA_=FA_+[6*np.pi/180]*nb_flat_spokes
        #print("FA_ {}".format(FA_[-1]))
        TE_=TE_+[3.45/1000]*nb_flat_spokes
        #print("TE {}".format(TE_[-1]))
        TR_[1:] = np.array(TE_[1:]) + min_TR_delay
        TR_[-1] = TR_[-1] + params_for_curve[-1]
        TR_ = list(TR_)
        #print("TR {}".format(TR_[-1]))
        spokes_count += nb_flat_spokes

    #print(spokes_count)
    radial_traj = Radial(total_nspokes=spokes_count, npoint=npoint)
    ntimesteps = int(spokes_count / group_size)

    m_.build_ref_images_bloch(TR_,FA_,TE_)

    s, s_w, s_f, keys = simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs, T1s, 300 / 1000,B1s, T_2w=40 / 1000, T_2f=80 / 1000,
                                               amp=fat_amp, shift=fat_shift, sigma=None, group_size=group_size,
                                               return_fat_water=True,return_combined_signal=False)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

    s_w = s_w.reshape(s_w.shape[0], -1).T
    s_f = s_f.reshape(s_f.shape[0], -1).T


    s_light, s_w_light, s_f_light, keys_light = simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs_light, T1s_light, 300 / 1000, B1s_light, T_2w=40 / 1000,
                                               T_2f=80 / 1000,
                                               amp=fat_amp, shift=fat_shift, sigma=None, group_size=group_size,
                                               return_fat_water=True,
                                               return_combined_signal=False)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

    s_w_light = s_w_light.reshape(s_w_light.shape[0], -1).T
    s_f_light = s_f_light.reshape(s_f_light.shape[0], -1).T


    #Rescaling parameters
    keys = [(t[0] * 1000, t[1] * 1000, t[2], t[3] / 1000) for t in keys]
    keys_light = [(t[0] * 1000, t[1] * 1000, t[2], t[3] / 1000) for t in keys_light]

    data = m_.generate_kdata(radial_traj, useGPU=useGPU,nthreads=1,fftw=0)
    data = np.array(data)
    data = data.reshape(spokes_count, -1, npoint)
    volumes_all = simulate_radial_undersampled_images(data, radial_traj, image_size, density_adj=True, useGPU=useGPU,ntimesteps=ntimesteps,nthreads=1,fftw=0)

    

    # from PIL import Image
    # gif=[]
    # volume_for_gif = np.abs(volumes_all)
    # for i in range(volume_for_gif.shape[0]):
    #     img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
    #     img=img.convert("P")
    #     gif.append(img)

    # filename_gif = "test_US_optim.gif"
    # gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)


    mask = m_.mask
    pca = True
    threshold_pca_bc = 10

    signals=volumes_all[:,mask>0]
    num_signals=signals.shape[1]
    if sigma>0:
        noise=sigma*np.random.normal(size=signals.shape)
        signals+=noise

    

    split = 1000
    #dict_optim_bc_cf = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
    #                                    threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False,
    #                                    useGPU_simulation=False,
    #                                    gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
    #                                    return_matched_signals=True)



    #all_maps, matched_signals = dict_optim_bc_cf.search_patterns_test((s_w, s_f, keys), signals)

    dict_optim_bc_cf = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                        threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False,
                                        useGPU_simulation=False,
                                        gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                        return_matched_signals=True,dictfile_light=(s_w_light, s_f_light, keys_light),threshold_ff=0.9)

    all_maps= dict_optim_bc_cf.search_patterns_test_multi_2_steps_dico((s_w, s_f, keys), signals)
    matched_signals = all_maps[0][-1]

    #print(all_maps)
    #print(matched_signals.shape)

    
    plt.close("all")
    plt.figure()
    i=np.random.choice(num_signals)
    print("Plotting Match Example")
    print(signals.shape)
    print(matched_signals.shape)
    print(i)
    
    plt.plot(signals[:,i],label="Orig")
    plt.plot(matched_signals[:,i],label="Matched")
    plt.legend()
    plt.savefig("Optim_Matching_example.jpg")

    key = "wT1"
    map = all_maps[0][0][key][m_.paramMap["ff"]<0.7]#*1000
    map_gt=m_.paramMap[key][m_.paramMap["ff"]<0.7]
    error = np.abs(map - map_gt)
    error_wT1 = np.mean(error/ map_gt)

    print("wT1 Cost : {}".format(error_wT1))

    std_wT1 = np.std(error/ map_gt)
    print("wT1 Std : {}".format(std_wT1))

    key = "ff"
    map = all_maps[0][0][key]
    map_gt = m_.paramMap[key]
    error = np.abs(map - map_gt)
    error_ff = np.mean(error)

    print("FF Cost : {}".format(error_ff))

    key = "df"
    map = all_maps[0][0][key]#/1000
    map_gt = m_.paramMap[key]
    error = np.abs(map - map_gt)
    error_df = np.mean(error)

    print("DF Cost : {}".format(error_df))

    key = "attB1"
    map = all_maps[0][0][key]
    map_gt = m_.paramMap[key]
    error = np.abs(map - map_gt)
    error_b1 = np.mean(error)

    print("B1 Cost : {}".format(error_b1))

    # num_breaks_TE=len(TE_breaks)
    FA_cost = np.mean((np.abs(np.diff(FA_[1:]))))
    # print("FA Cost : {}".format(FA_cost))

    time_cost = np.sum(TR_)
    print("Time Cost : {}".format(time_cost))
    #test_time=time_cost-TR_[-1]
    #print("Time Test : {}".format(test_time))

    result=lambda_T1 * error_wT1 + lambda_FF * error_ff + lambda_DF * error_df + lambda_B1 * error_b1 +lambda_FA * FA_cost + lambda_time * time_cost+lambda_stdT1 * std_wT1

    end_time=datetime.now()
    print(end_time-start_time)

    return result


def cost_function_simul_breaks_single_FA_FF_KneePhantom(params,**kwargs):
    #global result


    DFs = kwargs["DFs"]
    FFs = kwargs["FFs"]
    B1s = kwargs["B1s"]
    T1s = kwargs["T1s"]

    inversion = kwargs["inversion"]
    min_TR_delay = kwargs["min_TR_delay"]
    num_breaks_TE = kwargs["num_breaks_TE"]

    spokes_count = kwargs["spokes_count"]
    npoint = kwargs["npoint"]

    lambda_FA = kwargs["lambda_FA"]
    lambda_time = kwargs["lambda_time"]
    lambda_FF = kwargs["lambda_FF"]
    lambda_T1 = kwargs["lambda_T1"]
    lambda_DF = kwargs["lambda_DF"]
    lambda_B1 = kwargs["lambda_B1"]

    lambda_stdT1 = kwargs["lambda_stdT1"]

    fat_amp = np.array(kwargs["fat_amp"])
    fat_shift = np.array(kwargs["fat_shift"])

    useGPU=kwargs["useGPU"]

    image_size = (int(npoint / 2), int(npoint / 2))


    file = kwargs["file_phantom"]

    m_ = MapFromFile("Knee2D_Optim", file=file, rounding=True)
    radial_traj = Radial(total_nspokes=spokes_count, npoint=npoint)

    m_.buildParamMap()


    start_time=datetime.now()
    group_size = 8
    ntimesteps=int(spokes_count/group_size)

    params_for_curve=params[:-1]
    FA=params[-1]
    TR_, FA_, TE_ = convert_params_to_sequence_breaks_FF_single_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,FA,inversion)

    m_.build_ref_images_bloch(TR_,FA_,TE_)

    s, s_w, s_f, keys = simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs, T1s, 300 / 1000,B1s, T_2w=40 / 1000, T_2f=80 / 1000,
                                               amp=fat_amp, shift=fat_shift, sigma=None, group_size=group_size,
                                               return_fat_water=True)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

    s_w = s_w.reshape(s_w.shape[0], -1).T
    s_f = s_f.reshape(s_f.shape[0], -1).T

    data = m_.generate_kdata(radial_traj, useGPU=useGPU)
    data = np.array(data)
    data = data.reshape(spokes_count, -1, npoint)
    volumes_all = simulate_radial_undersampled_images(data, radial_traj, image_size, density_adj=True, useGPU=useGPU,ntimesteps=ntimesteps)

    mask = m_.mask
    pca = True
    threshold_pca_bc = 10

    split = 100
    dict_optim_bc_cf = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
                                        threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False,
                                        useGPU_simulation=False,
                                        gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
                                        return_matched_signals=True)

    all_maps, matched_signals = dict_optim_bc_cf.search_patterns_test((s_w, s_f, keys), volumes_all)

    #print(all_maps)
    #print(matched_signals.shape)



    key = "wT1"
    map = all_maps[0][0][key][m_.paramMap["ff"]<0.7]*1000
    map_gt=m_.paramMap[key][m_.paramMap["ff"]<0.7]
    error = np.abs(map - map_gt)
    error_wT1 = np.mean(error/ map_gt)

    print("wT1 Cost : {}".format(error_wT1))

    std_wT1 = np.std(error/ map_gt)
    print("wT1 Std : {}".format(std_wT1))

    key = "ff"
    map = all_maps[0][0][key]
    map_gt = m_.paramMap[key]
    error = np.abs(map - map_gt)
    error_ff = np.mean(error)

    print("FF Cost : {}".format(error_ff))

    key = "df"
    map = all_maps[0][0][key]/1000
    map_gt = m_.paramMap[key]
    error = np.abs(map - map_gt)
    error_df = np.mean(error)

    print("DF Cost : {}".format(error_df))

    key = "attB1"
    map = all_maps[0][0][key]
    map_gt = m_.paramMap[key]
    error = np.abs(map - map_gt)
    error_b1 = np.mean(error)

    print("B1 Cost : {}".format(error_b1))

    # num_breaks_TE=len(TE_breaks)
    FA_cost = np.mean((np.abs(np.diff(FA_[1:]))))
    # print("FA Cost : {}".format(FA_cost))

    time_cost = np.sum(TR_)
    print("Time Cost : {}".format(time_cost))
    #test_time=time_cost-TR_[-1]
    #print("Time Test : {}".format(test_time))

    result=lambda_T1 * error_wT1 + lambda_FF * error_ff + lambda_DF * error_df + lambda_B1 * error_b1 +lambda_FA * FA_cost + lambda_time * time_cost+lambda_stdT1 * std_wT1

    end_time=datetime.now()
    print(end_time-start_time)

    return result

def cost_function_simul_breaks_random_FA_correl(params,**kwargs):
    #global result

    DFs = kwargs["DFs"]
    FFs = kwargs["FFs"]
    B1s = kwargs["B1s"]
    T1s = kwargs["T1s"]

    inversion = kwargs["inversion"]
    min_TR_delay = kwargs["min_TR_delay"]
    bound_min_FA = kwargs["bound_min_FA"]
    num_breaks_TE = kwargs["num_breaks_TE"]
    spokes_count = kwargs["spokes_count"]

    lambda_FA = kwargs["lambda_FA"]
    lambda_time = kwargs["lambda_time"]

    num_params_FA = kwargs["num_params_FA"]

    fat_amp = np.array(kwargs["fat_amp"])
    fat_shift = np.array(kwargs["fat_shift"])

    useGPU=kwargs["useGPU"]


    #start_time=datetime.now()
    group_size = 8
    #ntimesteps=int(spokes_count/group_size)

    bound_max_FA=params[-1]
    params_for_curve=params[:-1]
    TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,num_params_FA,bound_min_FA,bound_max_FA,inversion)


    signals= simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs, T1s, 300 / 1000,B1s, T_2w=40 / 1000, T_2f=80 / 1000,
                                               amp=fat_amp, shift=fat_shift, sigma=None, group_size=None,
                                               return_fat_water=False)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):


    #print(all_maps)
    #print(matched_signals.shape)

    signals_for_corr = signals.reshape(signals.shape[0], -1)
    cov_signal = np.abs(signals_for_corr.conj().T @ signals_for_corr)
    inverse_std_signal = np.diag(np.sqrt(1 / np.diag(cov_signal)))
    corr_signal = inverse_std_signal @ cov_signal @ inverse_std_signal

    error_correl=np.linalg.norm(corr_signal-np.eye(corr_signal.shape[0]))
    #size=np.prod(corr_signal.shape)
    #print("Correl Cost : {}".format(error_correl))
    # num_breaks_TE=len(TE_breaks)
    #FA_cost = np.mean((np.abs(np.diff(FA_[1:]))))
    # print("FA Cost : {}".format(FA_cost))

    time_cost = np.sum(TR_)
    #print("Time Cost : {}".format(time_cost))
    #test_time=time_cost-TR_[-1]
    #print("Time Test : {}".format(test_time))

    result= error_correl + lambda_time * time_cost
    print("Error correl : {} ".format(error_correl))

    #end_time=datetime.now()
    #print(end_time-start_time)

    return result


def cost_function_simul_breaks_random_FA_crlb(params,**kwargs):
    #global result

    DFs = kwargs["DFs"]
    FFs = kwargs["FFs"]
    B1s = kwargs["B1s"]
    T1s = kwargs["T1s"]

    inversion = kwargs["inversion"]
    min_TR_delay = kwargs["min_TR_delay"]
    bound_min_FA = kwargs["bound_min_FA"]
    num_breaks_TE = kwargs["num_breaks_TE"]
    spokes_count = kwargs["spokes_count"]

    lambda_FA = kwargs["lambda_FA"]
    lambda_time = kwargs["lambda_time"]

    num_params_FA = kwargs["num_params_FA"]

    fat_amp = np.array(kwargs["fat_amp"])
    fat_shift = np.array(kwargs["fat_shift"])

    useGPU=kwargs["useGPU"]
    sigma2=kwargs["useGPU"]

    list_deriv=kwargs["list_deriv"]
    weights_deriv=kwargs["weights_deriv"]

    #start_time=datetime.now()
    group_size = 8
    #ntimesteps=int(spokes_count/group_size)

    bound_max_FA=params[-1]
    params_for_curve=params[:-1]
    TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,num_params_FA,bound_min_FA,bound_max_FA,inversion)


    signals,dico_deriv= simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs, T1s, 300 / 1000,B1s, T_2w=40 / 1000, T_2f=80 / 1000,
                                               amp=fat_amp, shift=fat_shift, sigma=None, group_size=None,
                                               return_fat_water=False,list_deriv=list_deriv)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):


    # print(signals.shape)
    # print(dico_deriv["ff"].shape)
    # print(dico_deriv["wT1"].shape)
    # print("#######################################")
    
    jac=convert_dico_to_jac(dico_deriv)
    # print(jac.shape)
    crlb=crlb_split(jac)
    #print("#######################################")
    #print(crlb)
    #print(crlb.shape)
    #print(all_maps)
    #print(matched_signals.shape)

    
    crlb=crlb.reshape(-1,len(list_deriv))
    crlb_error=np.linalg.norm(crlb,axis=0)
    
    
    crlb_error*=np.array(weights_deriv)

    for i,p in enumerate(list_deriv):
        print("CRLB {}: {}".format(p,crlb_error[i]))
        
    crlb_error=np.sum(crlb_error)

    print("#########################################")
    #size=np.prod(corr_signal.shape)
    #print("Correl Cost : {}".format(error_correl))
    # num_breaks_TE=len(TE_breaks)
    #FA_cost = np.mean((np.abs(np.diff(FA_[1:]))))
    # print("FA Cost : {}".format(FA_cost))


    time_cost = np.sum(TR_)
    #print("Time Cost : {}".format(time_cost))
    #test_time=time_cost-TR_[-1]
    #print("Time Test : {}".format(test_time))

    result= crlb_error + lambda_time * time_cost
    print("Error CRLB : {} ".format(crlb_error))

    #end_time=datetime.now()
    #print(end_time-start_time)

    return result


class LogCost(object):
    def __init__(self,f,dumpfile,num_maps=1,date_time=None):
        self.cost_values=[]
        self.it=1
        self.f=f
        self.dumpfile=dumpfile
        self.num_maps=num_maps
        if date_time is None:
            date_time = now.strftime("%Y%m%d_%H%M%S")
        self.date_time=date_time

    def __call__(self,x,**kwargs):
        global map_index
        print("############# ITERATION {}###################".format(self.it))
        print("Cost {}".format(self.f(x)))
        self.cost_values.append(self.f(x))
        self.it +=1
        if self.num_maps==1:
            map_index=0
        else:
            map_index=np.random.randint(0,self.num_maps-1)
        
        print("#########################################################")
        with open(self.dumpfile, "wb") as file:
             pickle.dump(x, file)

        np.save("./optims/log_cost_{}.npy".format(self.date_time),np.array(self.cost_values))


    def reset(self):
        self.cost_values = []
        self.it=1


def cost_function(p):
    global optimizer_config_for_cost
    return cost_function_simul_breaks_random_FA_KneePhantom_variablesp(p,**optimizer_config_for_cost)
    


toolbox = Toolbox("script_seqOptim_machines", description="MRF sequence optimization")
toolbox.add_program("optimize_sequence_random", optimize_sequence_random)
toolbox.add_program("optimize_sequence_random_FA", optimize_sequence_random_FA)
toolbox.add_program("optimize_sequence_breaks_common", optimize_sequence_breaks_common)
toolbox.add_program("optimize_sequence_random_FA_undersampling", optimize_sequence_random_FA_undersampling)
toolbox.add_program("optimize_sequence_random_FA_undersampling_variablesp", optimize_sequence_random_FA_undersampling_variablesp)
toolbox.add_program("optimize_sequence_FF_single_FA_undersampling", optimize_sequence_FF_single_FA_undersampling)
toolbox.add_program("optimize_sequence_random_FA_correl", optimize_sequence_random_FA_correl)
toolbox.add_program("optimize_sequence_random_FA_crlb", optimize_sequence_random_FA_crlb)
toolbox.add_program("generate_dico_and_files", generate_dico_and_files)

if __name__ == "__main__":
    toolbox.cli()
