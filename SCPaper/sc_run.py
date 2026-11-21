"""
Convenience functions for running pedestrian-vehicle interaction simulations
with the models from the SCPaper, using parameterisations that were retained
in the SCPaper model testing/fitting (see also "sc_run demo.ipynb"):

    - get_model_with_params(): get a model with its retained
      parameterisations, as an sc_fitting.ModelWithParams object.
    - run_simulation(): run one simulation, with specified initial
      conditions, for one parameterisation of a model.
    - run_population(): run the same scenario across many retained
      parameterisations of a model (an "artificial population" of agents),
      optionally in parallel over multiple processes.
    - summarise_population(): get a pandas DataFrame with interaction summary
      metrics (see the sc_analysis module in the repository root) for a list
      of simulations returned by run_population().
"""
from enum import Enum
import os
import concurrent.futures
import numpy as np
import sc_fitting


def is_stochastic_model(model_name):
    for stoch_str in ('oSN', 'oAN'):
        if stoch_str in model_name:
            return True
    else:
        return False
    

def get_model_with_params(model_name, param_bounds=None):
    """
        Return an sc_fitting.ModelWithParams object for a specified model, with
        retained parameterisations, i.e., parameterisations found to be working 
        well for this model in the SCPaper model testing/fitting.
        
        Parameters
        ----------
        model_name: str
            A string defining a model (e.g., 'oVAoBEvoAI').
        param_bounds: dict
            A dictionary with parameter names as keys, and tuples with lower and 
            upper bounds as values. If None, the full list of retained 
            parameterisations. If not None, these bounds will be used to 
            remove any parameterisations outside of the bounds before returning.
            Default: None
            
        Returns
        -------
        An sc_fitting.ModelWithParams object.
    """
    # load the right file with retained model information
    if is_stochastic_model(model_name):
        ret_models_file_name = sc_fitting.RETAINED_COMB_FNAME
    else:
        ret_models_file_name = sc_fitting.RETAINED_DET_FNAME
    retained_models = sc_fitting.load_results(ret_models_file_name)
    # find the correct retained model info
    for retained_model in retained_models:
        if retained_model.model == model_name:
            # found the model
            if not (param_bounds is None):
                # apply bounds to parameters
                for bounded_param in param_bounds.keys():
                    if not(bounded_param in retained_model.param_names):
                        raise Exception(f'Cannot find a parameter "{bounded_param}" for model "{model_name}".')
                    idx_param = retained_model.param_names.index(bounded_param)
                    ok_idxs = ((retained_model.params_array[:, idx_param]
                                >= param_bounds[bounded_param][0]) 
                               & (retained_model.params_array[:, idx_param]
                                <= param_bounds[bounded_param][1])) 
                    retained_model.params_array = retained_model.params_array[ok_idxs, :]
            # get a new ModelWithParams object, to get all the latest features
            new_retained_model = sc_fitting.ModelWithParams(
                model=model_name, param_names=retained_model.param_names,
                param_ranges=retained_model.param_ranges,
                params_array=retained_model.params_array)
            return new_retained_model
    raise Exception(f'Model "{model_name}" not found in {ret_models_file_name}.')


def run_simulation(model, initial_cp_distances, initial_speeds, end_time,
                   ped_prio=False, params_dict=None, idx_parameterisation=None, 
                   noise_seeds=(None, None)):
    """ Run one simulation.
    
        Parameters
        ----------
        model: str or sc_fitting.ModelWithParams
            Either a string model identifier (e.g., 'oVAoBEvoAI') or a 
            ModelWithParams object.
        initial_cp_distances: tuple of float
            Initial distances, in metres, from the centre points of pedestrian 
            and car (in that order) to the crossing point of their trajectories.
        initial_speeds: tuple of float
            Initial speeds, in metres per second, of pedestrian and car. Can be
            zero for the pedestrian, but not for the car.
        end_time: float
            Duration of simulation.
        ped_prio: bool
            If True, means that the pedestrian has crossing priority (e.g., zebra
            crossing). Default: False
        params_dict: dict
            A dict with parameter name strings as dict keys and parameter values as
            dict values. If the type of input parameter model is str, then this 
            parameter needs to be provided. Default: None
        idx_parameterisation: int
            Only considered if input parameter model is of 
            type sc_fitting.ModelWithParams. An index specifying which 
            parameterisation to use from model.params_array. If both this input
            parameter and params_dict are provided, params_dict will be used.
        noise_seeds: 2-tuple of int
            One value each for pedestrian and car agent (in that order). If not
            None, the value will be used to initialise the random number
            generator for this agent. Only relevant for stochastic model variants.
            Default: (None, None)
        
        Returns
        -------
        An sc_scenario.SCSimulation object with the simulation results.  
    """
    # get model name and params dict
    if isinstance(model, str):
        model_name = model
        if params_dict is None:
            raise Exception('Have to provide params_dict if specifying the model using a string identifier.')
    elif isinstance(model, sc_fitting.ModelWithParams):
        model_name = model.model
    else: 
        raise Exception(f'Unexpected type of input parameter model: {type(model)}')
    # get model parameters dict
    if params_dict is None:
        if idx_parameterisation is None:
            raise Exception('Have to provide at least one of params_dict and idx_parameterisation.')
        params_dict = model.get_params_dict(idx_parameterisation)
    # define SCPaperScenario object
    # - calculate initial times to conflict area from distances and speeds
    #   (assuming non-zero speeds for now)
    initial_ttcas = ((initial_cp_distances 
                      - np.array(sc_fitting.AGENT_COLL_DISTS)) 
                     / initial_speeds)             
    # - handle zero speeds
    # -- pedestrian
    if initial_speeds[sc_fitting.i_PED_AGENT] == 0:
        ped_start_standing = True
        ped_standing_margin = initial_cp_distances[sc_fitting.i_PED_AGENT]
    else:
        ped_start_standing = False
        ped_standing_margin = None
    # -- vehicle
    if initial_speeds[sc_fitting.i_VEH_AGENT] == 0:
        raise Exception('Zero initial speed for car agent not supported.')
    # - get correct simulation time step
    if is_stochastic_model(model_name):
        time_step = sc_fitting.PROB_SIM_TIME_STEP
    else:
        time_step = sc_fitting.DET_SIM_TIME_STEP
    scenario = sc_fitting.SCPaperScenario('sc_run scenario', 
                                          initial_ttcas=initial_ttcas, 
                                          ped_prio=ped_prio, 
                                          ped_start_standing=ped_start_standing, 
                                          ped_standing_margin=ped_standing_margin, 
                                          ped_initial_speed=initial_speeds[sc_fitting.i_PED_AGENT],
                                          veh_initial_speed=initial_speeds[sc_fitting.i_VEH_AGENT],
                                          time_step=time_step,
                                          end_time=end_time)
    # run simulation
    sc_simulation = sc_fitting.construct_model_and_simulate_scenario(model_name, params_dict, scenario, noise_seeds=noise_seeds)
    return sc_simulation


def get_population_noise_seeds(n_simulations, base_seed=None):
    """ Return a list of noise seeds for a population of simulations.
    
        Parameters
        ----------
        n_simulations: int
            Number of simulations.
        base_seed: int
            If None, no seeding, i.e., (None, None) is returned for each 
            simulation. Otherwise, numpy.random.SeedSequence(base_seed) is used
            to deterministically derive two independent integer seeds 
            (pedestrian, car) per simulation, such that the seeds depend only
            on base_seed and on the position of the simulation in the 
            population. Default: None
            
        Returns
        -------
        A list of n_simulations 2-tuples, usable as input parameter
        noise_seeds of run_simulation().
    """
    if base_seed is None:
        return [(None, None)] * n_simulations
    child_seed_seqs = np.random.SeedSequence(base_seed).spawn(n_simulations)
    return [tuple(int(seed) for seed in seed_seq.generate_state(
                sc_fitting.N_AGENTS)) for seed_seq in child_seed_seqs]


def run_population_member(model_name, params_dict, initial_cp_distances,
                          initial_speeds, end_time, ped_prio,
                          idx_parameterisation, noise_seeds):
    """ Run one simulation of a population, as part of run_population(). 
        Defined at module level so that it can be pickled for use with 
        multiple processes. Calls run_simulation(), and then stores the 
        following additional attributes on the returned sc_scenario.SCSimulation 
        object: model_name, params_dict, idx_parameterisation, noise_seeds.
    """
    sim = run_simulation(model_name, initial_cp_distances, initial_speeds, 
                         end_time, ped_prio=ped_prio, params_dict=params_dict,
                         noise_seeds=noise_seeds)
    sim.model_name = model_name
    sim.params_dict = params_dict
    sim.idx_parameterisation = idx_parameterisation
    sim.noise_seeds = noise_seeds
    return sim


def _run_population_member_from_args(args):
    return run_population_member(*args)


def run_population(model, initial_cp_distances, initial_speeds, end_time,
                   ped_prio=False, n_parameterisations=None, 
                   idx_parameterisations=None, base_seed=None, parallel=True,
                   n_workers=None, rng_seed=0):
    """ Run the same scenario across a population of retained 
        parameterisations of a model, one simulation per parameterisation.
        
        If parallel is True, the simulations are run in multiple processes
        using concurrent.futures.ProcessPoolExecutor. On platforms where new
        processes are spawned (e.g., Windows and macOS), calls to this function
        from a script must therefore be protected by an 
        if __name__ == "__main__": guard.
    
        Parameters
        ----------
        model: sc_fitting.ModelWithParams
            The model, with parameterisations, e.g., as returned by 
            get_model_with_params().
        initial_cp_distances, initial_speeds, end_time, ped_prio:
            Initial conditions and scenario settings, see run_simulation().
        n_parameterisations: int
            If not None, a random subsample (drawn without replacement, using 
            numpy.random.default_rng(rng_seed)) of this many of the 
            parameterisations in model.params_array is simulated. Cannot be 
            combined with idx_parameterisations. If both n_parameterisations 
            and idx_parameterisations are None, all parameterisations are
            simulated. Default: None
        idx_parameterisations: iterable of int
            If not None, explicit indices into model.params_array of the 
            parameterisations to simulate (repetitions are allowed, and will 
            get different noise seeds). Default: None
        base_seed: int
            If not None, used to derive deterministic per-simulation noise 
            seeds, see get_population_noise_seeds(), such that results are 
            reproducible (and identical between parallel and serial runs).
            If None, the simulations are not seeded. Only relevant for 
            stochastic model variants. Default: None
        parallel: bool
            If True, run in parallel over multiple processes, otherwise 
            serially in the current process. Default: True
        n_workers: int
            Maximum number of worker processes if parallel is True. If None,
            the number of CPUs is used. Default: None
        rng_seed: int
            Seed for drawing the random subsample of parameterisations, if 
            n_parameterisations is not None. Default: 0
            
        Returns
        -------
        A list of sc_scenario.SCSimulation objects, in the same order as the
        parameterisations simulated (for a random subsample, in the order 
        drawn). Each object has the additional attributes model_name, 
        params_dict, idx_parameterisation and noise_seeds, see 
        run_population_member().
    """
    if not isinstance(model, sc_fitting.ModelWithParams):
        raise Exception('Input parameter model should be an sc_fitting.ModelWithParams object.')
    # get the parameterisations to simulate
    if idx_parameterisations is not None:
        if n_parameterisations is not None:
            raise Exception('Cannot specify both n_parameterisations and idx_parameterisations.')
        idx_params_to_simulate = [int(idx) for idx in idx_parameterisations]
        for idx in idx_params_to_simulate:
            if idx < 0 or idx >= model.n_parameterisations:
                raise Exception(f'Parameterisation index {idx} out of range for model with {model.n_parameterisations} parameterisations.')
    elif n_parameterisations is not None:
        if n_parameterisations > model.n_parameterisations:
            raise Exception(f'Cannot draw {n_parameterisations} parameterisations from model with {model.n_parameterisations} parameterisations.')
        rng = np.random.default_rng(seed=rng_seed)
        idx_params_to_simulate = [int(idx) for idx in rng.choice(
            model.n_parameterisations, n_parameterisations, replace=False)]
    else:
        idx_params_to_simulate = list(range(model.n_parameterisations))
    n_simulations = len(idx_params_to_simulate)
    # get the noise seeds
    noise_seeds_list = get_population_noise_seeds(n_simulations, base_seed)
    # prepare the inputs for each simulation
    sims_args = []
    for idx_params, noise_seeds in zip(idx_params_to_simulate, noise_seeds_list):
        # (keeping the parameter values exactly as returned by 
        # get_params_dict(), so that results are identical to calling 
        # run_simulation() with idx_parameterisation)
        params_dict = model.get_params_dict(idx_params)
        sims_args.append((model.model, params_dict, initial_cp_distances, 
                          initial_speeds, end_time, ped_prio, idx_params, 
                          noise_seeds))
    # run the simulations
    if parallel and n_simulations > 1:
        if n_workers is None:
            n_workers = os.cpu_count() or 1
        n_workers = max(1, min(n_workers, n_simulations))
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            sims = list(executor.map(_run_population_member_from_args, sims_args))
    else:
        sims = [run_population_member(*sim_args) for sim_args in sims_args]
    return sims


def summarise_population(sims, consider_acc=False):
    """ Return a pandas DataFrame with one row per simulation in sims, with 
        the column idx_parameterisation, the interaction summary metrics 
        from sc_analysis.summarise_interaction(), and one column per model 
        parameter value.
    
        Parameters
        ----------
        sims: list of sc_scenario.SCSimulation
            Typically as returned by run_population(). For simulations without
            the attributes idx_parameterisation and params_dict (e.g., from 
            run_simulation()), idx_parameterisation is set to None and no
            parameter values are included.
        consider_acc: bool
            Passed on to sc_analysis.summarise_interaction(). Default: False
            
        Returns
        -------
        A pandas.DataFrame.
    """
    import pandas as pd
    import sc_analysis
    rows = []
    for sim in sims:
        row = {'idx_parameterisation': getattr(sim, 'idx_parameterisation', None)}
        row.update(sc_analysis.summarise_interaction(sim, consider_acc=consider_acc))
        params_dict = getattr(sim, 'params_dict', None)
        if params_dict is not None:
            row.update({param_name: float(param_value) for param_name, 
                        param_value in params_dict.items()})
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import numpy as np
    model_w_params = get_model_with_params('oVAoBEvoAI', param_bounds={'T_delta': (-np.inf, np.inf)})
    print(model_w_params)
    sim = run_simulation(model_w_params, initial_cp_distances=(3, 40), 
                         initial_speeds=sc_fitting.AGENT_FREE_SPEEDS, 
                         end_time=10, ped_prio=False, idx_parameterisation=0)
    print(sim.summarise_interaction())
    sim.do_plots(kinem_states=True, beh_probs=True)
    # run a population of 20 randomly drawn parameterisations of the 
    # maximally successful stochastic model, in parallel, with seeded noise
    stoch_model = get_model_with_params('oVAoBEvoAIoEAoSNvoPF')
    sims = run_population(stoch_model, initial_cp_distances=(3, 40), 
                          initial_speeds=sc_fitting.AGENT_FREE_SPEEDS, 
                          end_time=10, ped_prio=True, n_parameterisations=20,
                          base_seed=0)
    print(summarise_population(sims))
    



