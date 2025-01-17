import subprocess
import re
import astroplan
from astroplan import FixedTarget,Observer
from matplotlib.backends.backend_pdf import PdfPages
from astropy.coordinates import ICRS, SkyCoord, AltAz
from astropy import units as u
from astropy.utils.data import download_file
from astropy.table import Table, QTable, join
from astropy.time import Time
from astropy_healpix import *
from ligo.skymap import plot
from ligo.skymap.io import read_sky_map
import healpy as hp
import os
from matplotlib.pyplot import imread, figure, imshow, axis
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import datetime as dt
import time
import pickle
import json
import pandas as pd
import warnings
from docplex.mp.model import Model
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
warnings.simplefilter('ignore', astroplan.TargetNeverUpWarning)
warnings.simplefilter('ignore', astroplan.TargetAlwaysUpWarning)



def run_gwemopt_and_get_probability(
    fits_file, start_time_, end_time_, filters="g,i,r",exposure_times="180.0,180.0,180.0"):
    # tobs = (end_time_-start_time_)
    gpsTime = start_time_.mjd
    cmd = [
        "gwemopt-run",
        "-t", "ZTF",
        "--doTiles",
        "--doSchedule",
        "--timeallocationType", "powerlaw",
        "--scheduleType", "greedy",
        "-e", fits_file,
        "--filters", filters,
        "--exposuretimes", exposure_times,
        "--doSingleExposure",
        "--confidence_level", "0.9",
        "--airmass", "2.5",
        "--mindiff", "30",
        "--doUsePrimary",
        "--geometry", "3d",
        "--solverType", "heuristic",
        "--nside", "256",
        "--gpstime", str(gpsTime),
        "--Tobs", "0.0,0.5" 
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr
        matches = re.findall(r"Cumultative probability: ([\d.]+)", output)
        if matches:
            probability = float(matches[-1])
            return probability, output
        else:
            raise ValueError("Could not find cumulative probability in output")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed with error: {e}")

directory_path = "/u/ywagh/test_skymaps/"
filelist = sorted([f for f in os.listdir(directory_path) if f.endswith('.gz')])

slew_speed = 2.5 * u.deg / u.s
slew_accel = 0.4 * u.deg / u.s**2
readout = 8.2 * u.s



ns_nchips = 4
ew_nchips = 4
ns_npix = 6144
ew_npix = 6160
plate_scale = 1.01 * u.arcsec
ns_chip_gap = 0.205 * u.deg
ew_chip_gap = 0.140 * u.deg

ns_total = ns_nchips * ns_npix * plate_scale + (ns_nchips - 1) * ns_chip_gap
ew_total = ew_nchips * ew_npix * plate_scale + (ew_nchips - 1) * ew_chip_gap

rcid = np.arange(64)

chipid, rc_in_chip_id = np.divmod(rcid, 4)
ns_chip_index, ew_chip_index = np.divmod(chipid, ew_nchips)
ns_rc_in_chip_index = np.where(rc_in_chip_id <= 1, 1, 0)
ew_rc_in_chip_index = np.where((rc_in_chip_id == 0) | (rc_in_chip_id == 3), 0, 1)

ew_offsets = ew_chip_gap * (ew_chip_index - (ew_nchips - 1) / 2) + ew_npix * plate_scale * (ew_chip_index - ew_nchips / 2) + 0.5 * ew_rc_in_chip_index * plate_scale * ew_npix
ns_offsets = ns_chip_gap * (ns_chip_index - (ns_nchips - 1) / 2) + ns_npix * plate_scale * (ns_chip_index - ns_nchips / 2) + 0.5 * ns_rc_in_chip_index * plate_scale * ns_npix

ew_ccd_corners = 0.5 * plate_scale * np.asarray([ew_npix, 0, 0, ew_npix])
ns_ccd_corners = 0.5 * plate_scale * np.asarray([ns_npix, ns_npix, 0, 0])

ew_vertices = ew_offsets[:, np.newaxis] + ew_ccd_corners[np.newaxis, :]
ns_vertices = ns_offsets[:, np.newaxis] + ns_ccd_corners[np.newaxis, :]

def get_footprint(center):
    return SkyCoord(
        ew_vertices, ns_vertices,
        frame=center[..., np.newaxis, np.newaxis].skyoffset_frame()
    ).icrs

url = 'https://github.com/ZwickyTransientFacility/ztf_information/raw/master/field_grid/ZTF_Fields.txt'
filename = download_file(url)
field_grid = QTable(np.recfromtxt(filename, comments='%', usecols=range(3), names=['field_id', 'ra', 'dec']))
field_grid['coord'] = SkyCoord(field_grid.columns.pop('ra') * u.deg, field_grid.columns.pop('dec') * u.deg)
field_grid = field_grid[0:881]

observer = astroplan.Observer.at_site('Palomar')
night_horizon = -18 * u.deg
min_airmass = 2.5 * u.dimensionless_unscaled
airmass_horizon = (90 * u.deg - np.arccos(1 / min_airmass))

targets = field_grid['coord']
##############################################################################
# exposure_time = 120 * u.second
# exposure_time_day = exposure_time.to_value(u.day)

num_visits = 3
num_filters = 1

cadence = 30         #minutes
cadence_days = cadence / (60 * 24)
##############################################################################
hpx = HEALPix(nside=256, frame=ICRS())

problem_setup_time = []
coverage_problem_time = []
scheduler_time = []
solved_scheduler_list = []
total_solver_time = []

MILP1_prob = []
MILP2_prob = []
GWEMOPT_prob = []
MILP_time = []
GWEMOPT_time = []

def scheduler(skymap_file,n,exp_time):
    exposure_time = exp_time * u.second
    exposure_time_day = exposure_time.to_value(u.day)
    t1 = time.time()
    skymap, metadata = read_sky_map(skymap_file)
    print("SkyMap",n, "loaded")
    event_time = Time(metadata['gps_time'], format='gps').utc
    event_time.format = 'iso'

    if observer.is_night(event_time, horizon=night_horizon):
        start_time = event_time
    else:
        start_time = observer.sun_set_time(
            event_time, horizon=night_horizon, which='next')
    end_time = observer.sun_rise_time(
        start_time, horizon=night_horizon, which='next')
    target_start_time = Time(np.where(
        observer.target_is_up(start_time, targets, horizon=airmass_horizon),
        start_time,
        observer.target_rise_time(start_time, targets, which='next', horizon=airmass_horizon)))
    target_start_time.format = 'iso'
    target_end_time = observer.target_set_time(
        target_start_time, targets, which='next', horizon=airmass_horizon)
    target_end_time[
        (target_end_time.mask & ~target_start_time.mask) | (target_end_time > end_time)
    ] = end_time
    target_end_time.format = 'iso'

    field_grid['start_time'] = target_start_time
    field_grid['end_time'] = target_end_time
    observable_fields = field_grid[target_end_time - target_start_time >= exposure_time]

    footprint = np.moveaxis(
        get_footprint(SkyCoord(0 * u.deg, 0 * u.deg)).cartesian.xyz.value, 0, -1)
    footprint_healpix = np.unique(np.concatenate(
        [hp.query_polygon(hpx.nside, v, nest=(hpx.order == 'nested')) for v in footprint]))

    footprints = np.moveaxis(np.array(get_footprint(observable_fields['coord']).cartesian.xyz.value), 0, -1)
    footprints_healpix = [
        np.unique(np.concatenate([hp.query_polygon(hpx.nside, v) for v in footprint]))
        for footprint in tqdm(footprints)]
    prob = hp.ud_grade(skymap, hpx.nside, power=-2)
    # k = max number of 300s exposures 
    min_start = min(observable_fields['start_time'])
    max_end =max(observable_fields['end_time'])
    min_start.format = 'jd'
    max_end.format = 'jd'
    k = int(np.floor((max_end - min_start)/(exposure_time.to(u.day))))
    k_single = k/2
    k_revisit = np.floor(k/(num_visits*num_filters))
    print(k_single," number of exposures for single visit available")
    print(k_revisit," number of exposures for revisit available")
    print("problem setup",n, "completed")
    m1 = Model('max coverage problem for revisit')
    field_vars_1 = m1.binary_var_list(len(footprints), name='field')
    pixel_vars_1 = m1.binary_var_list(hpx.npix, name='pixel')

    footprints_healpix_inverse = [[] for _ in range(hpx.npix)]
    for field, pixels in enumerate(footprints_healpix):
        for pixel in pixels:
            footprints_healpix_inverse[pixel].append(field)

    for i_pixel, i_fields in enumerate(footprints_healpix_inverse):
        m1.add_constraint(m1.sum(field_vars_1[i] for i in i_fields) >= pixel_vars_1[i_pixel])

    m1.add_constraint(m1.sum(field_vars_1) <= k_revisit)
    m1.maximize(m1.dot(pixel_vars_1, prob))
    solution1 = m1.solve(log_output=False)

    # Second model for single visit coverage
    m4 = Model('max coverage problem for single')
    field_vars_4 = m4.binary_var_list(len(footprints), name='field')
    pixel_vars_4 = m4.binary_var_list(hpx.npix, name='pixel')

    # footprints_healpix_inverse = [[] for _ in range(hpx.npix)]
    # for field, pixels in enumerate(footprints_healpix):
    #     for pixel in pixels:
    #         footprints_healpix_inverse[pixel].append(field)

    for i_pixel, i_fields in enumerate(footprints_healpix_inverse):
        m4.add_constraint(m4.sum(field_vars_4[i] for i in i_fields) >= pixel_vars_4[i_pixel])

    m4.add_constraint(m4.sum(field_vars_4) <= k_single)
    m4.maximize(m4.dot(pixel_vars_4, prob))
    solution4 = m4.solve(log_output=False)
    
    # solution1 = coverage_problem(prob_ = prob, footprints_healpix_ = footprints_healpix, footprint_healpix_ = footprint_healpix)
    if solution1:
        print("optimization completed")
        total_prob_covered_REVISIT = solution1.objective_value
        total_prob_covered_SINGLE = solution4.objective_value
        print("Total probability covered for revisit:",total_prob_covered_REVISIT)
        print("Total probability covered for single visit:",total_prob_covered_SINGLE)
        '''we are adding a limit here, that if the total probability covered throughout the night is 
        less than 0.01, we won't be solving it for scheduling further [only for tests]'''
        
        if total_prob_covered_REVISIT>=0.01:            
            # selected_fields_ID = [i for i, v in enumerate(field_vars) if v.solution_value == 1]
            selected_fields = observable_fields[[solution1.get_value(v) == 1 for v in field_vars]]
            delta = exposure_time.to_value(u.day)
            limit_duration = ((end_time-start_time).value*(1-1/(num_visits*num_filters))) + delta
            filtered_fields = selected_fields[(selected_fields['end_time'] - selected_fields['start_time']).to_value(u.day) > limit_duration]
            if len(filtered_fields) == 0:
                print("No fields available for the entire night after filtering")
                return
            selected_fields = filtered_fields
            print(f"Selected {len(selected_fields)} fields after filtering")
            separation_matrix = selected_fields['coord'][:,np.newaxis].separation(selected_fields['coord'][np.newaxis,:])
                
            def slew_time(separation):
                return np.where(separation <= (slew_speed**2 / slew_accel),
                                np.sqrt(2 * separation / slew_accel),
                                (2 * slew_speed / slew_accel) + (separation - slew_speed**2 / slew_accel) / slew_speed)

            slew_times = slew_time(separation_matrix).value

            slew_time_value = slew_times*u.second
            slew_time_day = slew_time_value.to_value(u.day)
            m2 = Model("Telescope timings for revisit")
            footprints_selected = np.moveaxis(get_footprint(selected_fields['coord']).cartesian.xyz.value, 0, -1)
            footprints_healpix_selected = [
                np.unique(np.concatenate([hp.query_polygon(hpx.nside, v) for v in footprint]))
                for footprint in tqdm(footprints_selected)]

            probabilities_revisit = []

            for field_index in range(len(footprints_healpix_selected)):
                probability_field = np.sum(prob[footprints_healpix_selected[field_index]])
                probabilities_revisit.append(probability_field)
            print("worked for",len(probabilities_revisit),"fields")

            selected_fields['probabilities'] = probabilities_revisit

            delta = exposure_time.to_value(u.day)
            M = (selected_fields['end_time'].max() - selected_fields['start_time'].min()).to_value(u.day).item()
            x = [[m2.binary_var(name=f"x_{i}_visit_{v}") 
                for v in range(num_visits*num_filters)] 
                for i in range(len(selected_fields))]

            tc = [[m2.continuous_var(
                lb=(row['start_time'] - start_time).to_value(u.day),
                ub=(row['end_time'] - start_time - exposure_time).to_value(u.day),
                name=f"start_time_field_{i}_visit_{v}")
                for v in range(num_visits*num_filters)] 
                for i, row in enumerate(selected_fields)]

            visit_transition_times = [m2.continuous_var(
                lb=0,ub=M,name=f"visit_transition_{v}")
                                    for v in range(num_visits*num_filters-1)]  

            # for i in range(len(selected_fields)):
            #     priority_factor = priority_weights[i]
            #     for v in range(num_visits*num_filters):
            #         # Tighter time windows for high-priority fields
            #         if priority_factor > 1.5:  # High priority fields
            #             m2.add_constraint(
            #                 tc[i][v] <= (selected_fields['end_time'][i] - start_time - exposure_time).to_value(u.day) * 0.8,
            #                 ctname=f"priority_time_window_upper_{i}_{v}"
            #             )

            # Isolating visits
            for v in range(1, num_visits*num_filters):
                for i in range(len(selected_fields)):
                    m2.add_constraint(tc[i][v-1] + delta * x[i][v-1] <= visit_transition_times[v-1],
                        ctname=f"visit_end_{i}_visit_{v-1}")
                    m2.add_constraint(tc[i][v] >= visit_transition_times[v-1],
                        ctname=f"visit_start_{i}_visit_{v}")

            # Cadence constraints
            for i in range(len(selected_fields)):
                for v in range(1, num_visits*num_filters):
                    m2.add_constraint(tc[i][v] - tc[i][v-1] >= (cadence_days+delta) * (x[i][v] + x[i][v-1] - 1),
                        ctname=f"cadence_constraint_field_{i}_visits_{v}")

            #non-overlapping
            for v in range(num_visits*num_filters):
                for i in range(len(selected_fields)):
                    for j in range(i):
                        m2.add_constraint(tc[i][v] + delta * x[i][v] + slew_time_day[i][j] - tc[j][v] <= M * (2 - x[i][v] - x[j][v]),
                                        ctname=f"non_overlapping_cross_fields_{i}_{j}_visits_{v}")
                        m2.add_constraint(tc[j][v] + delta * x[j][v] + slew_time_day[i][j] - tc[i][v] <= M * (-1 + x[i][v] + x[j][v]),
                            ctname=f"non_overlapping_cross_fields_{j}_{i}_visits_{v}")

            # objective = m2.sum([
            #     probabilities[i] * priority_weights[i] * x[i][v]
            #     for i in range(len(selected_fields))
            #     for v in range(num_visits*num_filters)
            # ])

            # high_prob_threshold = np.percentile(probabilities, 75)
            # high_prob_fields = [i for i, p in enumerate(probabilities) if p >= high_prob_threshold]

            # if high_prob_fields:
            #     m2.add_constraint(
            #         m2.sum([x[i][v] for i in high_prob_fields for v in range(num_visits*num_filters)]) >= 
            #         len(high_prob_fields) * num_visits * num_filters * 0.5,  # At least 50% coverage of high-prob fields
            #         ctname="high_probability_coverage"
            #     )
            # m2.maximize(objective)

            m2.parameters.timelimit = 120  # Increased time limit
            m2.parameters.mip.tolerances.mipgap = 0.01
            m2.parameters.emphasis.mip = 2
            m2.parameters.mip.strategy.variableselect = 4  # Strong branching
            m2.parameters.mip.strategy.probe = 3  # Aggressive probing
            # m2.parameters.preprocessing.linear = 2  # Aggressive linear reduction
            m2.maximize(m2.sum([probabilities_revisit[i] * x[i][v]
                                for i in range(len(selected_fields))
                                for v in range(num_visits*num_filters)]))

            solution2 = m2.solve(log_output=False)
            
            selected_fields_single = observable_fields[[solution4.get_value(v) == 1 for v in field_vars_4]]
            delta = exposure_time.to_value(u.day)

            print(f"Selected {len(selected_fields_single)} fields after filtering")
            separation_matrix_single = selected_fields_single['coord'][:,np.newaxis].separation(selected_fields_single['coord'][np.newaxis,:])

            slew_times_single = slew_time(separation_matrix_single).value

            slew_time_value_single = slew_times_single*u.second
            slew_time_day_single = slew_time_value_single.to_value(u.day)
            m3 = Model("Telescope timings single visit")
            
            footprints_selected_single = np.moveaxis(get_footprint(selected_fields_single['coord']).cartesian.xyz.value, 0, -1)
            footprints_healpix_selected_single = [
                np.unique(np.concatenate([hp.query_polygon(hpx.nside, v) for v in footprint]))
                for footprint in tqdm(footprints_selected_single)]

            probabilities_single = []

            for field_index in range(len(footprints_healpix_selected_single)):
                probability_field = np.sum(prob[footprints_healpix_selected_single[field_index]])
                probabilities_single.append(probability_field)
            print("worked for",len(probabilities_single),"fields")

            selected_fields_single['probabilities'] = probabilities_single            
            
            x_ = m3.binary_var_list(len(selected_fields_single), name='selected field')
            s_ = [[m3.binary_var(name=f's_{i}_{j}') for j in range(i)] for i in range(len(selected_fields_single))]

            tc_ = [m3.continuous_var(
                    lb=(row['start_time'] - start_time).to_value(u.day),
                    ub=(row['end_time'] - start_time - exposure_time).to_value(u.day),
                    name=f'start_times_{i}'
                ) for i, row in enumerate(selected_fields_single)]
                    
            for i in range(len(tc_)):
                for j in range(i):
                    m3.add_constraint(tc_[i] + delta * x_[i] + slew_time_day_single[i][j] - tc_[j] <= M * (1 - s_[i][j]),
                                    ctname=f'non_overlap_gap_1_{i}_{j}')
                    m3.add_constraint(tc_[j] + delta * x_[j] + slew_time_day_single[i][j] - tc_[i] <= M * s_[i][j], 
                                    ctname=f'non_overlap_gap_2_{i}_{j}')

            m3.maximize(m3.sum(probabilities_single[i] * x_[i] for i in range(len(selected_fields_single))))
            m3.parameters.timelimit = 60
            solution3 = m3.solve(log_output=False)
            if solution2:
                t4 = time.time()
                MILP_time.append(t4-t1)
                MILP_1_prob = (solution2.objective_value)/(num_visits * num_filters)
                if __name__ == "__main__":
                    try:
                        t5 = time.time()
                        probability_, full_output = run_gwemopt_and_get_probability(fits_file = skymap_file, start_time_ = start_time, end_time_ = end_time)

                        t6 = time.time()

                        # solution3 = schedule_single_visit() 
                        MILP_2_prob = solution3.objective_value
                        GWEMOPT_prob.append(probability_)
                        MILP1_prob.append(MILP_1_prob)
                        MILP2_prob.append(MILP_2_prob)
                        GWEMOPT_time.append(t6-t5)
                        
                        print(f"\n #################Probability from GWEMOPT: {probability_}  ###########")
                        print(f'\n #################Probability from revisiting: {MILP_1_prob} ###########')
                        print(f'\n #################Probability from single visit: {MILP_2_prob} ###########')
                    except Exception as e:
                        print(f"Error: {e}")
            else:
                print("No solution for scheduler problem")
    else:
        print("no solution found for coverage problem")
    
exp_time_list = [180]
for z in range(len(exp_time_list)):
    for i in range(len(filelist)):
        print('scheduling event number',i)
        scheduler(os.path.join(directory_path, filelist[i]),i,exp_time_list[z])
        
milp1_values = np.array(MILP1_prob) * 100 
milp2_values = np.array(MILP2_prob) * 100 
gwemopt_values = np.array(GWEMOPT_prob) * 100 
positions = np.arange(len(MILP2_prob))


fig, ax = plt.subplots(figsize=(10, 8))

bar_width = 0.25  # Width of each bar
positions = np.arange(len(MILP1_prob))

# Plot each set of values with appropriate offsets
ax.bar(positions, milp1_values, width=bar_width, label='MILP1 Probability', color='#8884d8')
ax.bar(positions + bar_width, milp2_values, width=bar_width, label='MILP2 Probability', color='#ff7f50')
ax.bar(positions + 2 * bar_width, gwemopt_values, width=bar_width, label='GWEMOPT Probability', color='#82ca9d')

# Adjust the ticks and labels
ax.set_ylabel('Probability (%)')
ax.set_xlabel('Trial Number')
ax.set_title('MILP vs GWEMOPT Probability Comparison for Cadence 30min and Exp_time 180s')
ax.set_xticks(positions + bar_width)
ax.set_xticklabels([f'{i+1}' for i in positions])

# Add legend and grid
ax.legend()
ax.grid(True, alpha=0.3)

# Adjust layout and save the plot
plt.tight_layout()
bar_plot_filename = '/u/ywagh/MILP_vs_GWEMOPT_Probability_Comparison.png'
plt.savefig(bar_plot_filename, dpi=300, bbox_inches='tight')