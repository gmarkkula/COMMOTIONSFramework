# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 18:28:33 2021

@author: tragma
"""
import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import pickle
import sc_fitting

hiker_df = pd.read_csv(sc_fitting.DATA_FOLDER + '/hiker_cts.csv')
#print(hiker_df)

plt.close('all')
hiker_data = {}
fig, axs = plt.subplots(nrows=2, ncols=len(sc_fitting.HIKER_VEH_TIME_GAPS), 
                        sharex=True, sharey=True, num='Empirical CDFs',
                        figsize=(10, 6))
for i_speed, veh_speed_mph in enumerate(sc_fitting.HIKER_VEH_SPEEDS_MPH):
    for i_gap, veh_time_gap in enumerate(sc_fitting.HIKER_VEH_TIME_GAPS):
        for i_yield, veh_yielding in enumerate((False, True)):
            
            # get data
            scenario_name = sc_fitting.get_hiker_scen_name(veh_speed_mph, veh_time_gap, 
                                                  veh_yielding)
            scenario_rows = ((hiker_df['orig_speed'] == veh_speed_mph) 
                             & (hiker_df['time_gap'] == veh_time_gap)
                             & (hiker_df['is_braking'] == veh_yielding)
                             & (hiker_df['has_ehmi'] == False))
            #print(scenario_name)
            hiker_data[scenario_name] = \
                hiker_df[['subject', 'crossing_time']][scenario_rows]
            #print(hiker_data[scenario_name])
            
            # plot
            ax = axs[i_yield, i_gap]
            ecdf = ECDF(hiker_data[scenario_name]['crossing_time'])
            alpha = (1 - float(i_speed)/len(sc_fitting.HIKER_VEH_SPEEDS_MPH)) ** 2
            ax.step(ecdf.x, ecdf.y, 'k-', lw=i_speed+1, alpha=alpha)
            ax.set_xlim(-1, 11)
            ax.set_ylim(-.1, 1.1)
        axs[0, i_gap].set_title(f'Gap {veh_time_gap} s\n')
        axs[1, i_gap].set_xlabel('CIT (s)')
    
# finalise plotting
axs[0, 0].set_ylabel('Constant speed scenario\n\nCDF')  
axs[1, 0].set_ylabel('Yielding scenario\n\nCDF')        
axs[0,-1].legend(tuple(f'{spd} mph' for spd in sc_fitting.HIKER_VEH_SPEEDS_MPH))
plt.tight_layout()
plt.show()

# save data
print('Saving HIKER CIT data...')
with open(sc_fitting.DATA_FOLDER + '/' + sc_fitting.HIKER_DATA_FILE_NAME, 
          'wb') as file_obj:
    pickle.dump(hiker_data, file_obj)
            

