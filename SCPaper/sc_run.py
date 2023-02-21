from enum import Enum
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
    sc_simulation = sc_fitting.construct_model_and_simulate_scenario(model_name, params_dict, scenario)
    return sc_simulation


if __name__ == "__main__":
    import numpy as np
    model_w_params = get_model_with_params('oVAoBEvoAI', param_bounds={'T_delta': (-np.inf, np.inf)})
    print(model_w_params)
    sim = run_simulation(model_w_params, initial_cp_distances=(3, 40), 
                         initial_speeds=sc_fitting.AGENT_FREE_SPEEDS, 
                         end_time=10, ped_prio=False, idx_parameterisation=0)
    sim.do_plots(kinem_states=True, beh_probs=True)
    



