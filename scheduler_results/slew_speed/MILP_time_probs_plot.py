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
# from gurobipy import GRB,Env, Model
# env = Env(params={"WLSACCESSID": "c33d3a26-e111-4b61-b670-f9b9be9b4395",
#                   "WLSSECRET": "001698e8-35d4-450f-8fbe-b63f304b8f2b",
#                   "LICENSEID": 2533050})

directory_path = "/u/ywagh/test_skymaps/"
filelist = sorted([f for f in os.listdir(directory_path) if f.endswith('.gz')])
#probelm setup

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
exposure_time = 300 * u.second

hpx = HEALPix(nside=256, frame=ICRS())

coverage_problem_time = []
scheduler_time = []
solved_scheduler_list = []

def scheduler(skymap_file,n):
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
    k = int(np.floor((max_end - min_start)/(2*exposure_time.to(u.day))))
    print(k," number of exposures could be taken tonight")
    
    t2 =time.time()
    print("problem setup",n, "completed")
    m1 = Model('max coverage problem')
    footprint
    field_vars = m1.binary_var_list(len(footprints), name='field')
    pixel_vars = m1.binary_var_list(hpx.npix, name='pixel')

    footprints_healpix_inverse = [[] for _ in range(hpx.npix)]

    for field, pixels in enumerate(footprints_healpix):
        for pixel in pixels:
            footprints_healpix_inverse[pixel].append(field)

    for i_pixel, i_fields in enumerate(footprints_healpix_inverse):
        m1.add_constraint(m1.sum(field_vars[i] for i in i_fields) >= pixel_vars[i_pixel])

    m1.add_constraint(m1.sum(field_vars) <= k)
    m1.maximize(m1.dot(pixel_vars, prob))

    solution1 = m1.solve(log_output=True)
    t3 = time.time()
    if solution1:
        print("optimization completed")
        coverage_problem_time.append(t3-t2)
        total_prob_covered = solution1.objective_value

        print("Total probability covered:",total_prob_covered)
        
        '''we are adding a limit here, that if the total probability covered throughout the night is 
        less than 0.1, we won't be solving it for scheduling further [only for tests]'''
        
        if total_prob_covered>=0.1:            
            # selected_fields_ID = [i for i, v in enumerate(field_vars) if v.solution_value == 1]
            selected_fields = observable_fields[[solution1.get_value(v) == 1 for v in field_vars]]

            # selected_fields = observable_fields[selected_fields_ID]

            separation_matrix = selected_fields['coord'][:,np.newaxis].separation(selected_fields['coord'][np.newaxis,:])
            

            plot_filename = os.path.basename(skymap_file)

            def slew_time(separation):
                return np.where(separation <= (slew_speed**2 / slew_accel),np.sqrt(2 * separation / slew_accel),
                            (2 * slew_speed / slew_accel) + (separation - slew_speed**2 / slew_accel) / slew_speed)

            slew_times = slew_time(separation_matrix).value

            m2 = Model("Telescope timings")
            footprints_selected = np.moveaxis(get_footprint(selected_fields['coord']).cartesian.xyz.value, 0, -1)
            footprints_healpix_selected = [
                np.unique(np.concatenate([hp.query_polygon(hpx.nside, v) for v in footprint]))
                for footprint in tqdm(footprints_selected)]

            probabilities = []

            for field_index in range(len(footprints_healpix_selected)):
                probability_field = np.sum(prob[footprints_healpix_selected[field_index]])
                probabilities.append(probability_field)
            print("worked for",len(probabilities),"fields")

            selected_fields['probabilities'] = probabilities

            delta = exposure_time.to_value(u.day)
            M = (selected_fields['end_time'].max() - selected_fields['start_time'].min()).to_value(u.day).item()
            
            tc = [m2.continuous_var(
                    lb=(row['start_time'] - start_time).to_value(u.day),
                    ub=(row['end_time'] - start_time - exposure_time).to_value(u.day),
                    name=f'start_times_{i}'
                ) for i, row in enumerate(selected_fields)]
            
            slew_time_max = np.max(slew_times) * u.second
            slew_time_max_ = slew_time_max.to_value(u.day)

            slew_time_value = slew_times*u.second
            slew_time_day = slew_time_value.to_value(u.day)
            
            x = m2.binary_var_list(len(tc), name='selected field')
            s = [[m2.binary_var(name=f's_{i}_{j}') for j in range(i)] for i in range(len(tc))]



            #non-overlaping fields
            for i in range(len(tc)):
                for j in range(i):
                    # Non-overlap and max start time gap using Big M
                    m2.add_constraint(tc[i] >= tc[j] + delta * x[j] + slew_time_day[i][j] - M * (1 - s[i][j]),ctname=f'non_overlap_gap_1_{i}_{j}')
                    m2.add_constraint(tc[j] >= tc[i] + delta * x[i] + slew_time_day[i][j] - M * s[i][j],ctname=f'non_overlap_gap_2_{i}_{j}')
            w = 0.1 #weight factor for scheduling

            # for i in range(len(tc) - 1):
            #     m2.add_constraint(
            #         tc[i+1] - tc[i] >= delta + slew_time_max_,
            #         ctname=f'adjacent_field_constraint_{i}'
            #     )
            # m2.maximize(
            #     m2.sum(probabilities[i] * x[i] for i in range(len(tc))) 
            #     - w * m2.sum(slew_times[i][j] * s[i][j] for i in range(len(tc)) for j in range(i))
            # )
            m2.maximize(m2.sum(x[i] for i in range(len(selected_fields))))
            m2.parameters.timelimit = 60
            solution2 = m2.solve(log_output=True)
            t4 = time.time()
            if solution2:
                solved_scheduler_list.append(plot_filename)
                print("Optimization completed")
                scheduler_time.append(t4-t3)
                scheduled_start_times = [solution2.get_value(var) for var in tc]
                scheduled_fields = QTable(selected_fields)
                scheduled_fields['scheduled_start_time'] = Time(scheduled_start_times, format='mjd')
                scheduled_fields['scheduled_start_time'].format = 'iso'
                scheduled_fields['scheduled_end_time'] = scheduled_fields['scheduled_start_time'] + exposure_time
                scheduled_fields = scheduled_fields[np.asarray([solution2.get_value(var) for var in x], dtype=bool)]
                scheduled_fields.sort('scheduled_start_time')

                fig, ax = plt.subplots()
                ax.hlines(
                    np.arange(len(scheduled_fields)), 
                    scheduled_fields['scheduled_start_time'].mjd, 
                    scheduled_fields['scheduled_end_time'].mjd, 
                    colors='blue', linewidth=2)
                for i in range(len(scheduled_fields)):
                    ax.vlines(
                        scheduled_fields['scheduled_start_time'][i].mjd, 
                        ymin=i - 2, ymax=i + 2, 
                        color='black', linewidth=0.5, linestyle='-')
                    ax.vlines(
                        scheduled_fields['scheduled_end_time'][i].mjd, 
                        ymin=i - 2, ymax=i + 2, 
                        color='black', linewidth=0.5, linestyle='-')
                ax.set_yticks(np.arange(len(scheduled_fields)))
                ax.set_yticklabels(scheduled_fields['field_id'].astype(str))
                ax.set_xlabel('Observation time (MJD)')
                ax.set_ylabel('Field ID')
                save_path = '/u/ywagh/scheduler_results/slew_time_primary'
                full_path = os.path.join(save_path, f'slew_time_plot_{n}.png')
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                
                scheduled_fields['cumulative_probability'] = np.cumsum(scheduled_fields['probabilities'])

                field_indices = np.arange(len(scheduled_fields))  # x-axis (field indices)
                field_ids = scheduled_fields['field_id'].astype(str)  # Field IDs for x-ticks
                cumulative_prob = scheduled_fields['cumulative_probability']  # Cumulative probability for left y-axis

                fig, ax1 = plt.subplots(figsize=(10, 8))

                ax1.step(field_indices, cumulative_prob, where='mid', color='blue', label='Cumulative Probability')
                ax1.set_xlabel('Field Index')
                ax1.set_ylabel('Cumulative Probability', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')

                ax1.set_xticks(field_indices)
                ax1.set_xticklabels(field_ids, rotation=45, ha='right')

                ax1.legend(loc='upper left')

                plt.title('Cumulative Probability per Field')
                plt.tight_layout()
                save_path = '/u/ywagh/scheduler_results/slew_time_primary'
                full_path = os.path.join(save_path, f'cum_prob_{n}.png')
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
            else:
                print("No solution for scheduler problem")
    else:
        print("no solution found for coverage problem")
            

for i in range(len(filelist)):
    print('scheduling event number',i)
    scheduler(os.path.join(directory_path, filelist[i]),i)

print("the coverage problem time is :",coverage_problem_time,"seconds")
print("tghe scheduler took ",scheduler_time,"seconds to solve")

x_axis = range(len(coverage_problem_time))
plt.figure(figsize=(10, 8))
plt.bar(x_axis, coverage_problem_time, color='blue', edgecolor='black')
plt.xlabel('File Index')
plt.ylabel('Time to Optimize (seconds)')
plt.title('Optimization Time per skymap')
bar_plot_filename = '/u/ywagh/Coverage_optimization_times_slew.png'  
plt.savefig(bar_plot_filename, dpi=300, bbox_inches='tight')

x_axis = range(len(scheduler_time))
plt.figure(figsize=(10, 8))
plt.bar(x_axis, scheduler_time, color='blue', edgecolor='black')
plt.xlabel('File Index')
plt.ylabel('Time to Optimize (seconds)')
plt.title('Optimization Time per skymap')
bar_plot_filename = '/u/ywagh/Scheduler_optimization_times_slew.png' 
plt.savefig(bar_plot_filename, dpi=300, bbox_inches='tight')


# image_folder = '/u/ywagh/scheduler_results/slew_speed'
# pdf_filename = 'ztf_coverage_plots_combined.pdf'
# pdf_full_path = os.path.join(image_folder, pdf_filename)

# png_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

# with PdfPages(pdf_full_path) as pdf:
#     for i in range(0, len(png_files), 2):
#         fig, axs = plt.subplots(1, 2, figsize=(10, 6))  # 1 row, 2 columns layout

#         coverage_img_path = os.path.join(image_folder, f'cum_prob_{i//2}.png')
#         if os.path.exists(coverage_img_path):
#             coverage_img = imread(coverage_img_path)
#             axs[0].imshow(coverage_img)
#             axs[0].axis('off')
#             axs[0].set_title('Cumulative Probability Plot')

#         scheduler_img_path = os.path.join(image_folder, f'slew_time_plot_{i//2}.png')
#         if os.path.exists(scheduler_img_path):
#             scheduler_img = imread(scheduler_img_path)
#             axs[1].imshow(scheduler_img)
#             axs[1].axis('off')
#             axs[1].set_title('Scheduler Plot')
#         pdf.savefig(fig)
#         plt.close(fig)  

# print(f"All images saved to {pdf_full_path}")
with open('solved_scheduler_list.json', 'w') as f:
    json.dump(solved_scheduler_list, f)