import copy
from qutip import *
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm

# Functions to save and load the datas
def save_object(obj, filename):
    try:
        with open(filename, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)



def process_parallel_data(data):
    x = []
    y = []
    eta = []
    s = []
    if len(data[0]) == 4:
        for d in data:
            x += d[0]
            y += d[1]
            eta += d[2]
            s += d[3]
        outputs = (x, y, eta, s)
    else:
        for d in data:
            x += d[0]
            y += d[1]
            eta += d[2]
        outputs = (x, y, eta)
    
    return outputs



# Definition of states and operators

sz1 = tensor(sigmaz(), identity(2))
sz2 = tensor(identity(2), sigmaz())

g = basis(2, 0)
e = basis(2, 1)

states = [ 
            tensor(g , g),                           #state |down,down>
            tensor(e , e),                           #state |up,up>
            (tensor(e , g) + tensor(g , e)).unit(),  #state |2>
            (tensor(g , e) - tensor(e , g)).unit()   #state |3> 
            ]


O = [[states[i]*states[j].dag() for j in range(4)] for i in range(4)]



# Hamiltonian for correlated, non-Markovian noise
def get_H(δ1, δ2, pulse_ab, args):
    
    # time independent part of the Hamiltonian
    H0 = args["H0"]
    Hc = args["Hc"]
    H_noise = -0.5*δ1*sz1 - 0.5*δ2*sz2
    H = [H0 + H_noise, [Hc,  pulse_ab]]

    return H

# Hamiltonian for Markovian noise cases
def get_H_markov(pulse_ab,args):
    
    # time independent part of the Hamiltonian
    H0 = args["H0"]
    Hc = args["Hc"]
    H = [H0, [Hc,  pulse_ab]]

    return H

# Single noise realization efficiency for Markovian noise 
def efficiency(t, δ1, δ2, pulse_ab, args, opts):
    """
    Returns the STIRAP population transfer efficiecy (population of state |1> )given the noise values δ1 and δ2   
    """
    H_ = get_H(δ1, δ2, pulse_ab, args)
    res = mesolve(H_, args["ψ0"], t, args = args, options = opts)
    return expect(O[1][1], res.states[-1])


# definition of probability distributions
def gaussian(x, σ):
    """
    Returns the value of a gaussian with mean 0 and variance σ^2.
    """
    return np.exp(-(x**2) / (2 * σ**2)) / (σ * np.sqrt(2 * np.pi))




# Function for generating quasistatic noise
def rejection_sampling(pdf, σ, num_trajectories, xrange, maxp=None):
    """
    Generates quasistatic noise samples (no. of samples = num_trajectories) based on a specified probability density function 'pdf'.

    This function generates a list of noise samples where each sample is selected based on 
    the probability density function provided. It supports an optional maximum probability to 
    optimize the sampling process.
    Returns:
        list: A list of x-values that represent the generated noise samples.
    """
    noise = []
    if maxp == None:
        xlist = np.linspace(xrange[0], xrange[-1], 9999)
        y_max = np.max(pdf(xlist, σ))
    else:
        y_max = maxp
    while len(noise) < (num_trajectories):
        x = np.random.uniform(low=xrange[0], high=xrange[-1])
        y = np.random.uniform(low=0, high=y_max)
        if y <= pdf(x, σ):
            noise.append(x)
    return noise



# If noise = 1(by default) this func calculates the population of state |up,up> at final time tf (efficiency),
# with perfect projective measurement i.e. considering (anti-)correlated non-Markovian noise
# If noise = any other number, it calculates the efficiency considering uncorrelated non-Markovian noise
def eff_nonMarkovian(
    t, pdf, pulse_ab, Ωs, Ωp, σ1, σ2, num_trajectories, η, pars, opts, noise = 1
):
    """Calculates the efficiency (the population of state |up,up> at final time 't[-1]') in the quantum system
    influenced by non-Markovian noise, either (anti-)correlated or uncorrelated. The function uses different
    types of non-Markovian noise to determine how it affects the system's final state efficiency.

    Returns:
        float: The efficiency of the population in state |up,up> at time 't[-1]', calculated
               based on the specified type of non-Markovian noise.

    Note:
        Efficiency calculation for Markovian noise in this particular way is not supported!!!
    """

    weighted_sum = 0
    w = 0

    N = pars["N"]
    ψ0 = pars["ψ0"]


    pars_ = copy.deepcopy(pars)
    pars_["Ωs"] = Ωs
    pars_["Ωp"] = Ωp


    δ_range = np.linspace(-5*σ1, 5*σ1, num_trajectories)
    # weighted average for correlated/anti correlated noise
    if noise == 1:
        if η >= 0:
          sign_η = 1
        else:
          sign_η = -1

        η_abs = np.abs(η)
        
        global pops_wrapper
        def pops_wrapper(δ):
            δ1 = δ/np.sqrt(η_abs)
            δ2 = sign_η*np.sqrt(η_abs)*δ
            pops = efficiency(t, δ1, δ2, pulse_ab, pars_, opts)
            return pops
            
        with Pool(processes=cpu_count()) as pool:
            for i, pops in enumerate(pool.imap(pops_wrapper, δ_range)):
                w += pdf(δ_range[i], σ1)
                weighted_sum += pops * pdf(δ_range[i], σ1)
                
        weighted_avg = weighted_sum / w
    
    # weighted average for not correlated noise.
    # In this case pdf has to be 2D.
    else:
        δ1_range = np.linspace(-5*σ1, 5*σ1, num_trajectories)
        δ2_range = np.linspace(-5*σ2, 5*σ2, num_trajectories)
        δcombs = [(a, b) for a in δ1_range for b in δ2_range]
        
        
        def pops_wrapper(δcomb):
            δ1 = δcomb[0]
            δ2 = δcomb[1]
            pops = efficiency(t, δ1, δ2, pulse_ab, pars_, opts)
                
            return pops

        with Pool(processes=cpu_count()) as pool:
            for i, pops in enumerate(pool.imap(pops_wrapper, δcombs)):
                w += pdf(δcombs[i][0], σ1)*pdf(δcombs[i][1], σ2)
                weighted_sum += pops * pdf(δcombs[i][0], σ1)*pdf(δcombs[i][1], σ2)

        weighted_avg = weighted_sum / w

    return weighted_avg

# If noise = 1(by default) this func calculates the population of state |up,up> at final time tf (efficiency),
# by solving a master equation, considering (anti-)correlated Markovian noise,
# If noise = any other number, it calculates the efficiency considering uncorrelated Markovian noise
def eff_Markovian(
    t, pulse_ab, Ωs, Ωp, γ1, γ2, η, pars, opts, noise = 1):
    
    """Calculates the efficiency (the population of state |up,up> at final time 't[-1]') in the quantum system
    influenced by Markovian noise, either (anti-)correlated or uncorrelated.

    Returns:
        float: The efficiency of the population in state |up,up> at time 't[-1]'
    """

    N = pars["N"]
    ψ0 = pars["ψ0"]

    pars_ = copy.deepcopy(pars)
    pars_["Ωs"] = Ωs
    pars_["Ωp"] = Ωp
    

    H_mrkv = get_H_markov(pulse_ab, pars)

   
    if noise == 1:

        η_abs = np.abs(η)

        if η >= 0:
            sign_η = 1
        else:
            sign_η = -1
            
        c_op=(-0.5/np.sqrt(η_abs)*sz1-0.5*sign_η*np.sqrt(η_abs)*sz2)*np.sqrt(γ1) # correlated case
        res = mesolve(H_mrkv, ψ0, t, [c_op], args = pars_, options=opts)
        efficiency = expect(O[1][1], res.states[-1])

    else:
        
        c_op1= -0.5*sz1*np.sqrt(γ1) # uncorrelated case
        c_op2= -0.5*sz1*np.sqrt(γ2)
        res = mesolve(H_mrkv, ψ0, t, [c_op1,c_op2], args = pars_, options=opts)
        efficiency = expect(O[1][1], res.states[-1])
        
    return efficiency


# (Calculating data with perfect measurement)
# The next 4 functions generates the data for ML for the 6 noise respectively and
# return efficiencies calculated under 3 pulse conditions, the corresponding label and correlation parameter for 1 sample.

def get_data_CN(    
    t,
    pdf,
    pulse_ab,
    Ωs_max,
    Ωp_max,
    σ,
    ηrange,
    num_trajectories,
    num_samples,
    pars,
    opts,
):

    efficiency_correlated = []
    y_correlated = []
    eta_CN = []
    for i in tqdm(range(num_samples)):
        # efficiencies for (anti)correlated quasistatic noise when Ω_a = Ω_b
        eff1_CN = eff_nonMarkovian(
                t,
                pdf,
                pulse_ab,
                Ωs_max[0],
                Ωp_max[0],
                σ,
                None,
                num_trajectories,
                ηrange[i],
                pars,
                opts, 
                noise=1,
            )  
        # efficiencies for (anti)correlated quasistatic noise when Ω_a > Ω_b
        eff2_CN = eff_nonMarkovian(
                t,
                pdf,
                pulse_ab,
                Ωs_max[1],
                Ωp_max[1],
                σ,
                None,
                num_trajectories,
                ηrange[i],
                pars,
                opts,
                noise=1,
            )  
        # efficiencies for (anti)correlated quasistatic noise when Ω_a < Ω_b
        eff3_CN = eff_nonMarkovian(
                t,
                pdf,
                pulse_ab,
                Ωs_max[2],
                Ωp_max[2],
                σ,
                None,
                num_trajectories,
                ηrange[i],
                pars,
                opts,
                noise=1,
            )  

        eta_CN.append(ηrange[i])
        efficiency_correlated.append([eff1_CN, eff2_CN, eff3_CN])
        if ηrange[i]>0:
            y_correlated.append(0)
        else:
            y_correlated.append(1)

    return efficiency_correlated, y_correlated, eta_CN



def get_data_UCN(
    t,
    pdf,
    pulse_ab,
    Ωs_max,
    Ωp_max,
    σ1, 
    σ2,
    num_trajectories,
    num_samples,
    pars,
    opts,
):

    efficiency_uncorrelated = []
    y_uncorrelated = []

    for i in tqdm(range(num_samples)):
        # efficiencies for uncorrelated quasistatic noise when  Ω_a = Ω_b
        eff1_UCN = eff_nonMarkovian(
            t,
            pdf,
            pulse_ab,
            Ωs_max[0],
            Ωp_max[0],
            σ1[i], 
            σ2[i],
            num_trajectories,
            None,
            pars,
            opts,
            noise=2,
        )  
        # efficiencies for uncorrelated quasistatic noise when  Ω_a > Ω_b
        eff2_UCN = eff_nonMarkovian(
            t,
            pdf,
            pulse_ab,
            Ωs_max[1],
            Ωp_max[1],
            σ1[i], 
            σ2[i],
            num_trajectories,
            None,
            pars,
            opts,
            noise=2,
        ) 
         # efficiencies for uncorrelated quasistatic noise when  Ω_a < Ω_b
        eff3_UCN = eff_nonMarkovian(
            t,
            pdf,
            pulse_ab,
            Ωs_max[2],
            Ωp_max[2],
            σ1[i], 
            σ2[i],
            num_trajectories,
            None,
            pars,
            opts,
            noise=2,
        )  

        efficiency_uncorrelated.append([eff1_UCN, eff2_UCN, eff3_UCN])
        y_uncorrelated.append(2)

    return efficiency_uncorrelated, y_uncorrelated


def get_data_CM(                          
    t, 
    pulse_ab, 
    Ωs_max, 
    Ωp_max, 
    γ, 
    η,
    num_samples, 
    pars, 
    opts
):

    efficiency_correlated = []
    y_correlated = []
    eta_correlated = []
    gamma = []


    # efficiency for (anti)correlated non quasistatic noise when Ω_p = Ω_s       
    eff1_CM = eff_Markovian(
            t, 
            pulse_ab, 
            Ωs_max[0], 
            Ωp_max[0],
            γ,
            None,
            η, 
            pars,
            opts,
            noise = 1
        )  
    # efficiencies for (anti)correlated quasistatic noise when  Ω_a > Ω_b
    eff2_CM = eff_Markovian(
            t, 
            pulse_ab, 
            Ωs_max[1], 
            Ωp_max[1],
            γ,
            None,
            η, 
            pars,
            opts,
            noise = 1
        )
    # efficiencies (anti)correlated quasistatic noise when  Ω_a < Ω_b
    eff3_CM = eff_Markovian(
            t, 
            pulse_ab, 
            Ωs_max[2], 
            Ωp_max[2],
            γ,
            None,
            η, 
            pars,
            opts,
            noise = 1
        )  

    eta_correlated.append(η)
    gamma.append(γ)
    efficiency_correlated.append([eff1_CM, eff2_CM, eff3_CM])
        
    if η > 0:
       y_correlated.append(3)
    if η < 0:
       y_correlated.append(4)

    return efficiency_correlated, y_correlated, eta_correlated, gamma




def get_data_UCM(                          
    t,                                        
    pulse_ab, 
    Ωs_max, 
    Ωp_max, 
    γ1, 
    γ2,
    num_samples, 
    pars, 
    opts
):

    efficiency_uncorrelated = []
    y_uncorrelated = []
    γs_uncorrelated = []
   

    # efficiency for uncorrelated Markovian noise when Ω_p = Ω_s
    eff1_UCM = eff_Markovian(                            
            t, 
            pulse_ab, 
            Ωs_max[0], 
            Ωp_max[0],
            γ1,
            γ2,
            None, 
            pars,
            opts,
            noise = 2
        )  
    # efficiencies for uncorrelated Markovian noise when  Ω_a > Ω_b
    eff2_UCM = eff_Markovian(
            t, 
            pulse_ab, 
            Ωs_max[1], 
            Ωp_max[1],
            γ1,
            γ2,
            None,
            pars,
            opts,
            noise = 2
        )
    # efficiencies for uncorrelated Markovian noise when  Ω_a < Ω_b
    eff3_UCM = eff_Markovian(
            t, 
            pulse_ab, 
            Ωs_max[2], 
            Ωp_max[2],
            γ1,
            γ2,
            None,
            pars,
            opts,
            noise = 2
        )  

    
    efficiency_uncorrelated.append([eff1_UCM, eff2_UCM, eff3_UCM])
    γs_uncorrelated.append([γ1, γ2])
    y_uncorrelated.append(5)

    return efficiency_uncorrelated, y_uncorrelated, γs_uncorrelated
    

