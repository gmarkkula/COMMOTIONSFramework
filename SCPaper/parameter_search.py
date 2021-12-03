# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 09:26:51 2021

@author: tragma
"""
import math
import numpy as np
import datetime
import pickle
import multiprocessing as mp

STATUS_REP_HEADER_LEN = 40

class ParameterSearchResults:
    def __init__(self, n_params, n_metrics, n_parameterisations, 
                 n_repetitions=1, params_matrix=None):
        if params_matrix is None:
            self.params_matrix = np.full((n_parameterisations, n_params), 
                                         math.nan)
        else:
            self.params_matrix = np.copy(params_matrix)
        if n_repetitions == 1:
            self.metrics_matrix = np.full((n_parameterisations, n_metrics), 
                                          math.nan)
        else:
            self.metrics_matrix = np.full((n_parameterisations, n_metrics, 
                                           n_repetitions), math.nan)

class ParameterSearch:
    
    def verbosity_push(self, levels=1):
        """
        To be called by descendant classes to indicate an increase in depth
        of processing.
        
        Parameters
        ----------
        levels: int
            The number of levels by which the depth of processing increased. 
            Default is 1.

        Returns
        -------
        None.

        """
        self.curr_verbosity_depth += levels
    
    def verbosity_pop(self, levels=1):
        """
        To be called by descendant classes to indicate a decrease in depth
        of processing.
        
        Parameters
        ----------
        levels: int
            The number of levels by which the depth of processing decreased. 
            Default is 1.

        Returns
        -------
        None.
        """
        self.curr_verbosity_depth = max(0, self.curr_verbosity_depth - levels)
        
    def verbose_now(self):
        """
        Indicate whether the user has requested status output at the current 
        depth of processing.

        Returns
        -------
        Boolean
            True if the current depth of processing is less than or equal to
            the user-specified threshold.
        """
        return (self.curr_verbosity_depth <= self.max_verbosity_depth)
    
    def get_report_prefix(self):
        """
        Return a string with information and indentation suitable as prefix 
        for a status report at the current depth of processing.
        """
        header = datetime.datetime.now().strftime('%x %X') + ' ' + self.name
        if len(header) > STATUS_REP_HEADER_LEN:
            header = header[:STATUS_REP_HEADER_LEN]
        elif len(header) < STATUS_REP_HEADER_LEN:
            header = header + ' ' * (STATUS_REP_HEADER_LEN - len(header))
        padding = ('->' * (self.curr_verbosity_depth-1) 
                   + ' ' * (self.curr_verbosity_depth > 1))
        return header + padding    
    
    def report(self, message):
        """
        If the user has requested status reports at the current depth of 
        processing, print the provided message, after a header indicating time
        and the name of this search.

        Parameters
        ----------
        message : string
            The message to report.

        Returns
        -------
        None.

        """
        if self.verbose_now():
            print(self.get_report_prefix() + message)
    
    def get_params_dict(self, params_array):
        """
        Translate a vector of parameter values to a dict, with param_names as
        keys. 

        Parameters
        -------
        array : 1D array
            Parameter values, in the same order as in param_names.

        Returns
        -------
        The resulting dict.

        """
        params = {}
        for i in range(self.n_params):
            params[self.param_names[i]] = params_array[i]
        return params
    
    def get_metrics_array(self, metrics):
        """
        Translate a dict of metric values to an array.

        Parameters
        ----------
        metrics : dict
            The metric values, with metric_names as keys.

        Returns
        -------
        1D numpy array of metric values, in the same order as in metric_names.
        """
        metrics_array = np.full(self.n_metrics, math.nan)
        for i in range(self.n_metrics):
            metrics_array[i] = metrics[self.metric_names[i]]
        return metrics_array
    
    def get_metrics_for_params(self, params_dict, i_parameterisation=None,
                               i_repetition=None):
        """
        To be overridden by descendant classes.

        Parameters
        ----------
        params : dict
            Parameter values for the parameterisation to analyse, with 
            param_names as keys.
        i_parameterisation : int, optional
            The index number of the parameterisation being tested. Only needed
            if self.parallel is True or self.n_repetitions > 1. Default is None.
        i_repetition : int, optional
            The index number of the repetition for the parameterisation being
            tested. Only needed if self.parallel is True or 
            self.n_repetitions is > 1.

        Returns
        -------
        Implementation in descendant classes should return:
        If self.parallel is False:
            A dict of calculated metric values for the parameterisation, with 
            self.param_names as keys.
        If self.parallel is True:
            The descendant class implementation should not return anything, but
            should use self.pool.apply_async or similar, with 
            self.receive_metrics_for_params as callback, to queue a (non-class) 
            function call that returns a tuple of i_parameterisation,
            i_repetition, and the same dict of metric values as mentioned above. 

        """
        raise Exception('This method should be overridden.')
        
    def receive_metrics_for_params(self, results_tuple):
        self.verbosity_push()
        i_parameterisation = results_tuple[0]
        i_repetition = results_tuple[1]
        assert(i_repetition < self.n_repetitions)
        metrics_dict = results_tuple[2]
        if self.n_repetitions > 1:
            rep_str = f'rep. #{i_repetition+1}/{self.n_repetitions} for '
        else:
            rep_str = ''
        self.report(f'Received results for {rep_str}params #{i_parameterisation+1}'
                    f'/{self.n_parameterisations}.')
        if self.n_repetitions == 1:
            self.results.metrics_matrix[
                i_parameterisation, :] = self.get_metrics_array(metrics_dict)
        else:
            self.results.metrics_matrix[
                i_parameterisation, :, i_repetition] = self.get_metrics_array(
                    metrics_dict)
        self.verbosity_pop()
        self.n_evals_done += 1
        percent_done = math.floor(100 * self.n_evals_done / self.n_evals_to_do)
        if percent_done > self.last_percent_reported:
            self.report(f'Parameter search {percent_done} % done.')
            self.last_percent_reported = percent_done
    
    def search_list(self, params_matrix):
        """
        Go through a list of parameterisations, call get_metrics_for_params() 
        for each, and store the results as a ParameterSearchResults object in 
        self.results.
 
        Parameters
        ----------
        params_matrix : 2D numpy array
            One row for each parameterisation, one column for each parameter,
            in the same order as in param_names.

        Returns
        -------
        None.

        """
        self.verbosity_push()
        shape = params_matrix.shape
        assert(shape[1] == self.n_params)
        self.n_parameterisations = shape[0]
        self.results = ParameterSearchResults(n_params = self.n_params, 
                                              n_metrics = self.n_metrics,
                                              n_parameterisations = self.n_parameterisations,
                                              n_repetitions = self.n_repetitions,
                                              params_matrix = params_matrix)
        if self.parallel:
            self.report(f'Starting a pool of {self.n_workers} workers for parallel'
                        ' evaluation of parameterisations...')
            self.pool = mp.Pool(self.n_workers)
            self.report('Pool of workers initialised.')
        self.report(f'Searching {self.n_parameterisations} parameterisations for'
                    f' parameter set {self.param_names}...')
        self.last_percent_reported = 0
        self.n_evals_done = 0
        self.n_evals_to_do = self.n_parameterisations * self.n_repetitions
        for i_parameterisation in range(self.n_parameterisations):
            params_dict = self.get_params_dict(
                self.results.params_matrix[i_parameterisation, :])
            for i_repetition in range(self.n_repetitions):
                if self.parallel:
                    self.get_metrics_for_params(params_dict, i_parameterisation, 
                                                i_repetition)
                else:
                    if self.n_repetitions == 1:
                        metrics_dict = self.get_metrics_for_params(params_dict)
                    else:
                        metrics_dict = self.get_metrics_for_params(params_dict,
                                                                   i_parameterisation,
                                                                   i_repetition)
                    self.receive_metrics_for_params((i_parameterisation, 
                                                     i_repetition, metrics_dict))
        if self.parallel:
            # close pool for further calls
            self.pool.close()
            # wait for pool workers to complete
            self.pool.join()
            # get rid of pool object (can't be pickled/saved with the rest)
            self.pool = None 
        self.report('Parameter search complete.')
        self.verbosity_pop()
            
    
    def search_grid(self, param_arrays):
        """
        Create a list of parameterisations as all combinations of a provided
        array of values for each parameter, and then call search_list() for 
        that list.

        Parameters
        ----------
        param_arrays : dict
            A dict of arrays of parameter values for each parameter, with the
            names in param_names as keys.

        Returns
        -------
        None.

        """
        # figure out how many parameterisations are in the grid
        n_param_values = np.full(self.n_params, math.nan, dtype=int)
        for i in range(self.n_params):
            n_param_values[i] = len(param_arrays[self.param_names[i]])
        n_cum_param_combs = np.cumprod(n_param_values)
        n_parameterisations = n_cum_param_combs[-1]
        params_matrix = np.full((n_parameterisations, self.n_params), math.nan)
        # loop through parameterisations and create matrix of parameterisations
        for i_parameterisation in range(n_parameterisations):
            for i_param in range(self.n_params):
                i_param_val_pos = (math.floor(n_param_values[i_param] 
                                              * i_parameterisation 
                                              / n_cum_param_combs[i_param])
                                   % n_param_values[i_param])
                params_matrix[i_parameterisation, i_param] = \
                    param_arrays[self.param_names[i_param]][i_param_val_pos]
        # search the matrix of parameterisations
        self.search_list(params_matrix)
        
    
    def get_metric_results(self, metric_name):
        """
        Return an array with the results for a metric.

        Parameters
        ----------
        metric_name : string
            The name of the metric for which to get results.

        Returns
        -------
        numpy array
            If self.n_repetitions=1: A view of the 1D numpy array for the 
            metric, from self.results.metrics_matrix.
            If self.n_repetitions>1: A vnew 2D numpy array for the 
            metric, from self.results.metrics_matrix, with parameterisations
            as first dimension and repetitions as the second.

        """
        if self.n_repetitions == 1:
            return self.results.metrics_matrix[
                :, self.metric_names.index(metric_name)]
        else:
            return np.squeeze(self.results.metrics_matrix[
                :, self.metric_names.index(metric_name), :])
        
    
    def save(self, file_name):
        self.verbosity_push()
        self.report(f'Saving results of "{self.name}" to file "{file_name}"...')
        file_obj = open(file_name, 'wb')
        pickle.dump(self, file_obj)
        file_obj.close()
        self.report('Saving done.')
        self.verbosity_pop()
        
    
    def __init__(self, param_names, metric_names, name='Unnamed', 
                 n_repetitions=1, parallel=False, n_workers=mp.cpu_count()-1, 
                 verbosity=0):
        """
        Constructor.

        Parameters
        ----------
        param_names : tuple of strings
            Names of the parameters to be searched.
        metric_names : tuple of strings
            Names of the metrics to be calculated for each parameterisation.
        name : string, optional
            Name for the parameter search. The default is 'Unnamed'.
        n_repetitions : int, optional
            The number of times metrics should be calculated for each
            parameterisation. The default is 1.
        parallel : bool, optional
            If True, a multiprocessing.Pool will be created, for each parameter
            search, to evaluate parameterisations in parallel. The default is 
            False.
        n_workers : int, optional
            The number of parallel processes to run across if parallel is True.
            The default is the number of available CPU cores minus one.
        verbosity : int, optional
            The deepest level of processing at which to provide status report
            information. The default is 0, i.e., no status reports at all.

        Returns
        -------
        None.

        """
        self.name = name
        self.n_params = len(param_names)
        self.param_names = param_names
        self.n_metrics = len(metric_names)
        self.metric_names = metric_names
        self.n_repetitions = n_repetitions
        self.parallel = parallel
        self.n_workers = n_workers
        self.max_verbosity_depth = verbosity
        self.curr_verbosity_depth = 0
        

def load(file_name, verbose=False):
    if verbose:
        print(f'Loading parameter search from file "{file_name}"...')
    file_obj = open(file_name, 'rb')
    loaded_obj = pickle.load(file_obj)
    file_obj.close()
    if not isinstance(loaded_obj, ParameterSearch):
        raise Exception(f'File {file_name} did not contain a ParameterSearch object.')
    if verbose:
        print('\tDone.')
    return loaded_obj
        


# function used by the unit testing code below
import time, random
def par_get_metrics_for_params(params, i_parameterisation, i_repetition):
    time.sleep(random.uniform(0.25, 0.5))
    metrics = {}
    metrics['m1'] = params['p1'] + params['p2'] + params['p3']
    metrics['m2'] = -metrics['m1']
    return (i_parameterisation, i_repetition, metrics)

    
# unit testing
if __name__ == "__main__":
    
    # simple descendant class for testing
    class TestParameterSearch(ParameterSearch):
        def get_metrics_for_params(self, params):
            metrics = {}
            metrics['m1'] = params['p1'] + params['p2'] + params['p3']
            metrics['m2'] = -metrics['m1']
            return metrics
        def __init__(self, name, verbosity):
            super().__init__(param_names=('p1', 'p2', 'p3'),
                             metric_names=('m1', 'm2'), name=name,
                             verbosity=verbosity)
    
    # test list search        
    test_search = TestParameterSearch(name='test1', verbosity=1)
    params_matrix = np.array((
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2)))
    test_search.search_list(params_matrix)
    results = np.concatenate((test_search.results.params_matrix, 
               test_search.results.metrics_matrix), axis=1)
    print(results)
    
    # test saving and loading
    file_name = '_delme/search_save_test.pkl'
    test_search.save(file_name)
    reloaded_search = load(file_name, verbose=True)
    results = np.concatenate((reloaded_search.results.params_matrix, 
               reloaded_search.results.metrics_matrix), axis=1)
    print(results)
    
    # test grid search
    test_search = TestParameterSearch(name='test2', verbosity=0)
    param_arrays = {}
    param_arrays['p1'] = (0,1,2,3)
    param_arrays['p2'] = np.arange(0, 9, 3)
    param_arrays['p3'] = param_arrays['p1']
    test_search.search_grid(param_arrays)
    results = np.concatenate((test_search.results.params_matrix, 
               test_search.results.metrics_matrix), axis=1)
    print(results)
    
    
    # test parallelised grid search
    # - simple descendant class for testing
    class TestParallelParameterSearch(ParameterSearch):
        def get_metrics_for_params(self, params, i_parameterisation, i_repetition):
            self.verbosity_push()
            self.report(f'Setting up test of parameterisation #{i_parameterisation}:'
                        f' {params}')
            self.pool.apply_async(par_get_metrics_for_params, 
                                  (params, i_parameterisation, i_repetition),
                                  callback = self.receive_metrics_for_params)
            self.verbosity_pop()
        def __init__(self, name, verbosity, n_repetitions=1):
            super().__init__(param_names=('p1', 'p2', 'p3'),
                             metric_names=('m1', 'm2'), name=name,
                             n_repetitions=n_repetitions, parallel=True, 
                             verbosity=verbosity)
    # - run the test
    test_search = TestParallelParameterSearch(name='test3', verbosity=3)
    param_arrays = {}
    param_arrays['p1'] = (0,1,2,3)
    param_arrays['p2'] = np.arange(0, 9, 3)
    param_arrays['p3'] = param_arrays['p1']
    test_search.search_grid(param_arrays)
    results = np.concatenate((test_search.results.params_matrix, 
               test_search.results.metrics_matrix), axis=1)
    print(results)
    # - run the test, with repetitions
    test_search = TestParallelParameterSearch(name='test4', n_repetitions=2, 
                                              verbosity=3)
    param_arrays = {}
    param_arrays['p1'] = (0,1,2,3)
    param_arrays['p2'] = np.arange(0, 9, 3)
    param_arrays['p3'] = param_arrays['p1']
    test_search.search_grid(param_arrays)
    results = np.concatenate((test_search.results.params_matrix, 
               test_search.results.metrics_matrix[:, :, 0]), axis=1)
    print(results)
    results = np.concatenate((test_search.results.params_matrix, 
               test_search.results.metrics_matrix[:, :, 1]), axis=1)
    print(results)
    
    