# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 09:26:51 2021

@author: tragma
"""
import math
import numpy as np
import datetime
import pickle

STATUS_REP_HEADER_LEN = 25

class ParameterSearchResults:
    def __init__(self, n_params, n_metrics, n_parameterisations, 
                 params_matrix=None):
        if params_matrix is None:
            self.params_matrix = np.full((n_parameterisations, n_params), 
                                         math.nan)
        else:
            self.params_matrix = np.copy(params_matrix)
        self.metrics_matrix = np.full((n_parameterisations, n_metrics), 
                                      math.nan)


class ParameterSearch:
    
    def verbosity_push(self):
        """
        To be called by descendant classes to indicate an increase in depth
        of processing.

        Returns
        -------
        None.

        """
        self.curr_verbosity_depth += 1
    
    def verbosity_pop(self):
        """
        To be called by descendant classes to indicate a decrease in depth
        of processing.

        Returns
        -------
        None.
        """
        self.curr_verbosity_depth = max(0, self.curr_verbosity_depth - 1)
        
    def verbose_now(self):
        """
        Indicates whether the user has requested status output at the current 
        depth of processing.

        Returns
        -------
        Boolean
            True if the current depth of processing is less than or equal to
            the user-specified threshold.

        """
        return (self.curr_verbosity_depth <= self.max_verbosity_depth)
    
    def report(self, message):
        if self.verbose_now():
            header = datetime.datetime.now().strftime('%x %X') + ' ' + self.name
            if len(header) > STATUS_REP_HEADER_LEN:
                header = header[:STATUS_REP_HEADER_LEN]
            elif len(header) < STATUS_REP_HEADER_LEN:
                header = header + ' ' * (STATUS_REP_HEADER_LEN - len(header))
            padding = '--' * (self.curr_verbosity_depth-1)
            print(header + ' ' + padding + ' ' + message)
                
        
    
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
    
    def get_metrics_for_params(self, params_vector):
        """
        To be overridden by descendant classes.

        Parameters
        ----------
        params : 1D numpy array 
            Parameter values for the parameterisation to analyse, with 
            param_names as keys.

        Returns
        -------
        Implementation in descendant classes should return a 1D numpy array of 
        calculated metric values for the parameterisation.

        """
        raise Exception('This method should be overridden.')
    
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
        n_parameterisations = shape[0]
        self.results = ParameterSearchResults(n_params = self.n_params, 
                                              n_metrics = self.n_metrics,
                                              n_parameterisations = n_parameterisations,
                                              params_matrix = params_matrix)
        self.report(f'Searching {n_parameterisations} parameterisations for'
                    f' parameters {self.param_names}...')
        for i in range(n_parameterisations):
            self.results.metrics_matrix[i, :] = self.get_metrics_for_params(
                self.results.params_matrix[i, :])
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
        
    
    def save(self, file_name):
        self.verbosity_push()
        self.report(f'Saving results of "{self.name}" to file "{file_name}"...')
        file_obj = open(file_name, 'wb')
        pickle.dump(self, file_obj)
        file_obj.close()
        self.report('\tSaving done.')
        self.verbosity_pop()
        
    
    def __init__(self, param_names, metric_names, name='Unnamed', verbosity=0):
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
        
    
# unit testing
if __name__ == "__main__":
    
    # simple descendant class for testing
    class TestParameterSearch(ParameterSearch):
        def get_metrics_for_params(self, params_vector):
            params = self.get_params_dict(params_vector)
            metrics = {}
            metrics['m1'] = params['p1'] + params['p2'] + params['p3']
            metrics['m2'] = -metrics['m1']
            return self.get_metrics_array(metrics)
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
    
    
    
    
    