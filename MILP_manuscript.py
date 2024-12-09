import astroplan
from astropy.coordinates import ICRS, SkyCoord, AltAz, get_moon, EarthLocation, get_body
from astropy import units as u
from astropy.utils.data import download_file
from astropy.table import Table, QTable, join
from astropy.time import Time, TimeDelta
from astropy_healpix import *
from ligo.skymap import plot
from ligo.skymap.io import read_sky_map
import healpy as hp
import os
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import datetime as dt
import pickle
import pandas as pd
from docplex.mp.model import Model

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
warnings.simplefilter('ignore', astroplan.TargetNeverUpWarning)
warnings.simplefilter('ignore', astroplan.TargetAlwaysUpWarning)

# directory_path = "/u/ywagh/test_skymaps/S240422ed.fits"
# skymap, metadata = read_sky_map(os.path.join(directory_path))

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

#******************************************************************************
skymap, metadata = read_sky_map(os.path.join(directory_path, filelist[6]))

plot_filename = os.path.basename(filelist[6])
# plot_filename = 'S240422ed'
# ci
#******************************************************************************

event_time = Time(metadata['gps_time'], format='gps').utc
event_time.format = 'iso'

event_time = Time(metadata['gps_time'], format='gps').utc
event_time.format = 'iso'
print('event time:',event_time)
observer = astroplan.Observer.at_site('Palomar')
night_horizon = -18 * u.deg
if observer.is_night(event_time, horizon=night_horizon):
    start_time = event_time
else:
    start_time = observer.sun_set_time(
        event_time, horizon=night_horizon, which='next')

# Find the latest possible end time of observations: the time of sunrise.
end_time = observer.sun_rise_time(
    start_time, horizon=night_horizon, which='next')

min_airmass = 2.5 * u.dimensionless_unscaled
airmass_horizon = (90 * u.deg - np.arccos(1 / min_airmass))
targets = field_grid['coord']

# Find the time that each field rises and sets above an airmass of 2.5.
target_start_time = Time(np.where(
    observer.target_is_up(start_time, targets, horizon=airmass_horizon),
    start_time,
    observer.target_rise_time(start_time, targets, which='next', horizon=airmass_horizon)))
target_start_time.format = 'iso'

# Find the time that each field sets below the airmass limit. If the target
# is always up (i.e., it's circumpolar) or if it sets after surnsise,
# then set the end time to sunrise.
target_end_time = observer.target_set_time(
    target_start_time, targets, which='next', horizon=airmass_horizon)
target_end_time[
    (target_end_time.mask & ~target_start_time.mask) | (target_end_time > end_time)
] = end_time
target_end_time.format = 'iso'
# Select fields that are observable for long enough for at least one exposure
##############################################################################
exposure_time = 180 * u.second
exposure_time_day = exposure_time.to_value(u.day)

num_visits = 2
num_filters = 2

cadence = 60         #minutes
cadence_days = cadence / (60 * 24)
##############################################################################
field_grid['start_time'] = target_start_time
field_grid['end_time'] = target_end_time
observable_fields = field_grid[target_end_time - target_start_time >= exposure_time]

# print(observable_fields)
hpx = HEALPix(nside=256, frame=ICRS())

footprint = np.moveaxis(
    get_footprint(SkyCoord(0 * u.deg, 0 * u.deg)).cartesian.xyz.value, 0, -1)
footprint_healpix = np.unique(np.concatenate(
    [hp.query_polygon(hpx.nside, v, nest=(hpx.order == 'nested')) for v in footprint]))

'''
# computing the footprints of every ZTF field as HEALPix indices. Downsampling skymap to same resolution.
'''
footprints = np.moveaxis(get_footprint(observable_fields['coord']).cartesian.xyz.value, 0, -1)
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
k = np.floor(k/(num_visits*num_filters))
print(k," number of exposures could be taken tonight")

print("problem setup completed")

m1 = Model('max coverage problem')

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
print(f"number fo fields observed should be less than {k}")

solution = m1.solve(log_output=True)

print("optimization completed")
total_prob_covered = solution.objective_value

print("Total probability covered:",total_prob_covered)

selected_fields_ID = [i for i, v in enumerate(field_vars) if v.solution_value == 1]
print(len(selected_fields_ID), "fields selected")
selected_fields = observable_fields[selected_fields_ID]
# print(selected_fields)

separation_matrix = selected_fields['coord'][:,np.newaxis].separation(selected_fields['coord'][np.newaxis,:])

def slew_time(separation):
   return np.where(separation <= (slew_speed**2 / slew_accel),
                   np.sqrt(2 * separation / slew_accel),
                   (2 * slew_speed / slew_accel) + (separation - slew_speed**2 / slew_accel) / slew_speed)

slew_times = slew_time(separation_matrix).value

slew_time_value = slew_times*u.second
slew_time_day = slew_time_value.to_value(u.day)

m2 = Model("Telescope timings")

observer_location = EarthLocation.of_site('Palomar')

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

x = [[m2.binary_var(name=f"x_{i}_visit_{v}") 
      for v in range(num_visits*num_filters)] 
      for i in range(len(selected_fields))]

tc = [[m2.continuous_var(
    lb=(row['start_time'] - start_time).to_value(u.day),
    ub=(row['end_time'] - start_time - exposure_time).to_value(u.day),
    name=f"start_time_field_{i}_visit_{v}")
    for v in range(num_visits*num_filters)] 
    for i, row in enumerate(selected_fields)]

visit_transition_times = [m2.continuous_var(lb=0,ub=M,name=f"visit_transition_{v}") for v in range(num_visits*num_filters-1)]  

# Cadence constraints
for i in range(len(selected_fields)):
    for v in range(1, num_visits*num_filters):
        m2.add_constraint(tc[i][v] - tc[i][v-1] >= cadence_days * (x[i][v] + x[i][v-1] - 1),
            ctname=f"cadence_constraint_field_{i}_visits_{v}")

#non-overlapping
for v in range(num_visits*num_filters):
    for i in range(len(selected_fields)):
        for j in range(i):
            m2.add_constraint(tc[i][v] + delta * x[i][v] + slew_time_day[i][j] - tc[j][v] <= M * (2 - x[i][v] - x[j][v]),
                              ctname=f"non_overlapping_cross_fields_{i}_{j}_visits_{v}")
            m2.add_constraint(tc[j][v] + delta * x[j][v] + slew_time_day[i][j] - tc[i][v] <= M * (-1 + x[i][v] + x[j][v]),
                ctname=f"non_overlapping_cross_fields_{j}_{i}_visits_{v}")

# Isolating visits
for v in range(1, num_visits*num_filters):
    for i in range(len(selected_fields)):
        m2.add_constraint(tc[i][v-1] + delta * x[i][v-1] <= visit_transition_times[v-1],
            ctname=f"visit_end_{i}_visit_{v-1}")
        m2.add_constraint(tc[i][v] >= visit_transition_times[v-1],
            ctname=f"visit_start_{i}_visit_{v}")

# Isolating visits
# for v in range(1, num_visits*num_filters):
#     prev_visit_end = m2.max([tc[i][v-1] + 2 * delta * x[i][v-1] for i in range(len(selected_fields))])
#     for i in range(len(selected_fields)):
#         m2.add_constraint(tc[i][v] >= prev_visit_end,
#             ctname=f"visit_sequence_field_{i}_visit_{v}")

m2.maximize(m2.sum([probabilities[i] * x[i][v]
                    for i in range(len(selected_fields))
                    for v in range(num_visits*num_filters)]))

m2.parameters.timelimit = 60
solution = m2.solve(log_output=True)

scheduled_fields_by_visit = []
for v in range(num_visits * num_filters):
    visit_fields = [i for i in range(len(selected_fields)) if solution.get_value(x[i][v]) == 1]
    scheduled_fields_by_visit.append(visit_fields)

scheduled_fields = selected_fields.copy()

scheduled_tc = []
for v in range(num_visits * num_filters):
    visit_times = []
    for i in range(len(selected_fields)):
        if i in scheduled_fields_by_visit[v]:
            visit_times.append(solution.get_value(tc[i][v]))
        else:
            visit_times.append(np.nan) 
    scheduled_tc.append(visit_times)

scheduled_tc = np.array(scheduled_tc).T  

for i in range(num_visits * num_filters):
    scheduled_fields[f"Scheduled_start_filt_times_{i}"] = scheduled_tc[:, i]

for v in range(num_visits * num_filters):
    scheduled_fields[f"Selected_in_visit_{v}"] = [1 if i in scheduled_fields_by_visit[v] else 0 
                                                 for i in range(len(selected_fields))]
    
n_visits = num_visits * num_filters  

fig, axes = plt.subplots(n_visits, 1, figsize=(8, 3 * n_visits), sharex=True)

for i in range(n_visits):
    start_col = f'Scheduled_start_filt_times_{i}'
    end_col = f'Scheduled_end_filt_times_{i}'
    
    # Convert start times to MJD and set format
    scheduled_fields[start_col] = Time(scheduled_fields[start_col], format='mjd')
    scheduled_fields[start_col].format = 'iso'
    scheduled_fields[end_col] = scheduled_fields[start_col] + exposure_time_day
    
    # Sort fields by end time for better visualization
    scheduled_fields.sort(end_col)
    
    # Get the start and end times for the vertical lines
    first_start_time = scheduled_fields[start_col].mjd[0]
    last_end_time = scheduled_fields[end_col].mjd[-1]
    
    ax = axes[i]  
    # Plot observation time intervals as horizontal lines
    ax.hlines(
        np.arange(len(scheduled_fields)),
        scheduled_fields[start_col].mjd,
        scheduled_fields[end_col].mjd,
        colors='blue',
        linewidth=2
    )
    # Plot small vertical lines at start and end times of each interval
    for j in range(len(scheduled_fields)):
        ax.vlines(
            scheduled_fields[start_col][j].mjd,
            ymin=j - 0.2,
            ymax=j + 0.2,
            color='black',
            linewidth=0.5,
            linestyle='-'
        )
        ax.vlines(
            scheduled_fields[end_col][j].mjd,
            ymin=j - 0.2,
            ymax=j + 0.2,
            color='black',
            linewidth=0.5,
            linestyle='-'
        )
    
    # Plot big vertical lines at the start of the first field and end of the last field
    ax.axvline(first_start_time, color='red', linestyle='--', linewidth=1.5, label='Start of First Field')
    ax.axvline(last_end_time, color='green', linestyle='--', linewidth=1.5, label='End of Last Field')
    
    # Add labels and title
    ax.set_yticks(np.arange(len(scheduled_fields)))
    ax.set_yticklabels(scheduled_fields['field_id'].astype(str))
    ax.set_ylabel('Field ID')
    ax.set_title(f'Observation Schedule for Visit {i + 1}')
    ax.legend(loc='upper right')  # Add legend to distinguish vertical lines
    
axes[-1].set_xlabel('Observation time (MJD)')

plt.tight_layout()
# plt.title(f'Total Cumulative Probability per Field:{total_cum_prob}')
plt.savefig('revisit_plots.png', dpi=300)
plt.show()
