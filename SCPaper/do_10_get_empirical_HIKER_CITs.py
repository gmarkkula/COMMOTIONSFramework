# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 18:28:33 2021

@author: tragma
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sc_fitting

hiker_df = pd.read_csv(sc_fitting.DATA_FOLDER + '/hiker_cts.csv')

# get the empirical HIKER data 
hiker_data = {}
fig, axs = plt.subplots(nrows=2, ncols=len(sc_fitting.HIKER_VEH_TIME_GAPS), 
                        sharex=True, sharey=True, num='Empirical CDFs',
                        figsize=(10, 6))
n_data_points = 0
for i_speed, veh_speed_mph in enumerate(sc_fitting.HIKER_VEH_SPEEDS_MPH):
    for i_gap, veh_time_gap in enumerate(sc_fitting.HIKER_VEH_TIME_GAPS):
        for i_yield, veh_yielding in enumerate((False, True)):
            scenario_name = sc_fitting.get_hiker_scen_name(veh_speed_mph, 
                                                           veh_time_gap, 
                                                           veh_yielding)
            scenario_rows = ((hiker_df['orig_speed'] == veh_speed_mph) 
                             & (hiker_df['time_gap'] == veh_time_gap)
                             & (hiker_df['is_braking'] == veh_yielding)
                             & (hiker_df['has_ehmi'] == False))
            hiker_data[scenario_name] = \
                hiker_df[['subject', 'crossing_time']][scenario_rows]
            n_data_points += np.count_nonzero(scenario_rows)
            
print(f'Total number of data points: {n_data_points}')

# plot CDFs
plt.close('all')
sc_fitting.do_hiker_cit_cdf_plot(hiker_data)

# save data
print('Saving HIKER CIT data...')
with open(sc_fitting.DATA_FOLDER + '/' + sc_fitting.HIKER_DATA_FILE_NAME, 
          'wb') as file_obj:
    pickle.dump(hiker_data, file_obj)
            

