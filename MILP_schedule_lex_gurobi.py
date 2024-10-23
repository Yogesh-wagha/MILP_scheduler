import astroplan
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
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import datetime as dt
import pickle
import pandas as pd
from docplex.mp.model import Model
# from gurobipy import GRB,Env, Model_
# env = Env(params={"WLSACCESSID": "c33d3a26-e111-4b61-b670-f9b9be9b4395",
#                   "WLSSECRET": "001698e8-35d4-450f-8fbe-b63f304b8f2b",
#                   "LICENSEID": 2533050})
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
warnings.simplefilter('ignore', astroplan.TargetNeverUpWarning)
warnings.simplefilter('ignore', astroplan.TargetAlwaysUpWarning)
directory_path = "/u/ywagh/test_skymaps/"
filelist = sorted([f for f in os.listdir(directory_path) if f.endswith('.gz')])
print(filelist)

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

skymap, metadata = read_sky_map(os.path.join(directory_path, filelist[2]))
# print("SkyMap loaded")
plot_filename = os.path.basename(filelist[12])

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
# sequence of 1800 seconds.
exposure_time = 1800 * u.second
field_grid['start_time'] = target_start_time
field_grid['end_time'] = target_end_time
observable_fields = field_grid[target_end_time - target_start_time >= exposure_time]

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

# k = max number of 1800s exposures 
min_start = min(observable_fields['start_time'])
max_end =max(observable_fields['end_time'])
min_start.format = 'jd'
max_end.format = 'jd'
k = int(np.floor((max_end - min_start)/exposure_time.to(u.day)))


print("problem setup completed")
# m = Model('real telescope',env=env)
m = Model('real_telescope')

field_vars = m.binary_var_list(len(footprints), name='field')
pixel_vars = m.binary_var_list(hpx.npix, name='pixel')
y_vars = [[m.binary_var(name=f'y_{i}_{j}') for j in range(len(footprints))] for i in range(len(footprints))]
start_times = m.continuous_var_list(len(footprints), name='start_time')
exp_time = exposure_time.value
durations = [exp_time for _ in range(len(footprints))]

footprints_healpix_inverse = [[] for _ in range(hpx.npix)]

for field, pixels in enumerate(footprints_healpix):
  for pixel in pixels:
      footprints_healpix_inverse[pixel].append(field)


#ensure that if the pixel is covered at least one of the fields containing that fields must be selected for observation\n",

for i_pixel, i_fields in enumerate(footprints_healpix_inverse):
  m.add_constraint(m.sum(field_vars[i] for i in i_fields) >= pixel_vars[i_pixel])
#2. number of fields selected should be less than what could be observed  throughout the night
m.add_constraint(m.sum(field_vars) <= k)
print("number fo fields observed should be less than k")
# 3 movement constraint (both the fields i and j are to be selected for observation if y[i,j]=1)
# 4 non-overlapping fields
# 5 symmetric observations (manouever from i to j, should not be differentiated from manouever from j to i)

separation_matrix = observable_fields['coord'][:,np.newaxis].separation(observable_fields['coord'][np.newaxis,:])

def slew_time(separation):
    return np.where(
        separation <= (slew_speed**2 / slew_accel), 
        np.sqrt(2 * separation / slew_accel), 
        (2 * slew_speed / slew_accel) + (separation - slew_speed**2 / slew_accel) / slew_speed
        )

slew_times = slew_time(separation_matrix).value
print("slew time calculated")
M = 1e6

# for i in range(len(footprints)):
#     for j in range(len(footprints)):
#         if i != j:
#             m.add_constraint(start_times[i] + durations[i] <= start_times[j] + M * (1 - y_vars[i][j]),f'overlap_{i}_{j}')
#             m.add_constraint(start_times[j] + durations[j] <= start_times[i] + M * y_vars[i][j],f'overlap_{j}_{i}')

#Differentiating movement from field i to j and vice versa
# for i in range(len(footprints)):
#     for j in range(len(footprints)):
#         if i != j:
#             m.add_constraint(y_vars[i][j] + y_vars[j][i] <= 1, f'direction_constraint_{i}_{j}')"

# move_constraints = [
#     (
#         m.add_constraint(y_vars[i][j] <= field_vars[i], f'move_if_selected_1_{i}_{j}'),
#         m.add_constraint(y_vars[i][j] <= field_vars[j], f'move_if_selected_2_{i}_{j}')
#     )
#     for i in range(len(footprints)) for j in range(len(footprints)) if i != j]

[
    (
        m.add_constraint(y_vars[i][j] <= field_vars[i], f'move_if_selected_1_{i}_{j}'),
        m.add_constraint(y_vars[i][j] <= field_vars[j], f'move_if_selected_2_{i}_{j}')
    )
    for i in range(len(footprints)) for j in range(len(footprints)) if i != j]

print("fields are selected, field movemnt constraint")
# overlap_constraints = [
#     m.add_constraint(start_times[i] + durations[i] <= start_times[j] + M * (1 - y_vars[i][j]), f'overlap_{i}_{j}')
#     for i in range(len(footprints)) for j in range(i+1, len(footprints))]

[
    m.add_constraint(start_times[i] + durations[i] <= start_times[j] + M * (1 - y_vars[i][j]), f'overlap_{i}_{j}')
    for i in range(len(footprints)) for j in range(i+1, len(footprints))]
print("overlaping constraint defined")
# reverse_overlap_constraints = [
#   m.add_constraint(start_times[j] + durations[j] <= start_times[i] + M * y_vars[i][j], f'overlap_{j}_{i}')
#   for i in range(len(footprints)) for j in range(i+1, len(footprints))
  # ]

[
  m.add_constraint(start_times[j] + durations[j] <= start_times[i] + M * y_vars[i][j], f'overlap_{j}_{i}')
  for i in range(len(footprints)) for j in range(i+1, len(footprints))
  ]
print("reverse overlaping constraint defined")
objective = m.sum(prob[i] * field_vars[i] for i in range(len(footprints))) - m.sum(slew_times[i][j] * y_vars[i][j] for i in range(len(footprints)) for j in range(len(footprints)) if i != j)
print("objective func defined")
m.maximize(objective)
m.set_time_limit(600)

solution = m.solve()
if solution:
  print("Solution found")
  # Extract the selected fields and their start times
  selected_fields = [i for i in range(len(footprints)) if solution.get_value(field_vars[i]) == 1]
  selected_start_times = [solution.get_value(start_times[i]) for i in selected_fields]
  
  # Print the results
  for field, start_time in zip(selected_fields, selected_start_times):
      print(f"Field {field} is selected for observation, start time: {start_time} seconds")
  else:
    print("No solution found")

 
