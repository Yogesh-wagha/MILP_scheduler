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

skymap, metadata = read_sky_map(os.path.join(directory_path, filelist[1]))
# print("SkyMap loaded")
plot_filename = os.path.basename(filelist[1])

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
print("problem setup completed")
# m = Model('real telescope',env=env)
m = Model('real_telescope')

# field_vars = [m.addVar(vtype=GRB.BINARY) for _ in range(len(footprints))]
# pixel_vars = [m.addVar(vtype=GRB.BINARY) for _ in range(hpx.npix)]

field_vars = m.binary_var_list(len(footprints), name='field')
pixel_vars = m.binary_var_list(hpx.npix, name='pixel')

footprints_healpix_inverse = [[] for _ in range(hpx.npix)]

for field, pixels in enumerate(footprints_healpix):
    for pixel in pixels:
        footprints_healpix_inverse[pixel].append(field)

for i_pixel, i_fields in enumerate(footprints_healpix_inverse):
     m.add_constraint(m.sum(field_vars[i] for i in i_fields) >= pixel_vars[i_pixel])

m.add_constraint(m.sum(field_vars) <= 30)
m.maximize(m.dot(pixel_vars, prob))

# m.optimize()

solution = m.solve(log_output=True)
'''
print("optimization completed")
total_prob_covered = m.ObjVal
# print("Total probability covered:",total_prob_covered)

selected_fields = observable_fields[[v.x == 1 for v in field_vars]]


#plotting the ztf-coverage

# skymap_name = os.path.basename(filename)
# Compute the total probability covered
total_prob_covered = m.ObjVal

plt.figure(figsize=(10, 8))
ax = plt.axes(projection='astro mollweide')

for row in selected_fields:
    coords = SkyCoord(
        [ew_total, -ew_total, -ew_total, ew_total],
        [ns_total, ns_total, -ns_total, -ns_total],
        frame=row['coord'].skyoffset_frame()
    ).icrs
    ax.add_patch(plt.Polygon(
        np.column_stack((coords.ra.deg, coords.dec.deg)),
        alpha=0.5,
        facecolor='lightgray',
        edgecolor='black',
        transform=ax.get_transform('world')
    ))

ax.grid()
ax.imshow_hpx(prob, cmap='cylon')

plt.text(0.05, 0.95, f'Total Probability Covered: {total_prob_covered:.2f}', transform=ax.transAxes,
        fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.savefig(f'ztf_coverage_{plot_filename}.png', dpi=300, bbox_inches='tight')
'''
if solution:
    print("Optimization completed")
    total_prob_covered = solution.objective_value
    selected_fields = observable_fields[[solution.get_value(v) == 1 for v in field_vars]]

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='astro mollweide')

    for row in selected_fields:
        coords = SkyCoord(
            [ew_total, -ew_total, -ew_total, ew_total],
            [ns_total, ns_total, -ns_total, -ns_total],
            frame=row['coord'].skyoffset_frame()
        ).icrs
        ax.add_patch(plt.Polygon(
            np.column_stack((coords.ra.deg, coords.dec.deg)),
            alpha=0.5,
            facecolor='lightgray',
            edgecolor='black',
            transform=ax.get_transform('world')
        ))

    ax.grid()
    ax.imshow_hpx(prob, cmap='cylon')

    plt.text(0.05, 0.95, f'Total Probability Covered: {total_prob_covered:.2f}', 
             transform=ax.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(f'ztf_coverage_{plot_filename}.png', dpi=300, bbox_inches='tight')
else:
    print("No solution found")

'''
delta = exposure_time.to_value(u.day)
M = (selected_fields['end_time'].max() - selected_fields['start_time'].min()).to_value(u.day).item()

t = [m.addVar(
        lb=(row['start_time'] - start_time).to_value(u.day),
        ub=(row['end_time'] - start_time - exposure_time).to_value(u.day),
     ) for row in selected_fields]

x = [m.addVar(vtype=GRB.BINARY) for _ in range(len(t))]
s = [[m.addVar(vtype=GRB.BINARY) for j in range(i)] for i in range(len(t))]
for i in range(len(t)):
    for j in range(i):
        m.addConstr(t[i] + delta * x[i] - t[j] <= M * (1 - s[i][j]))
        m.addConstr(t[j] + delta * x[j] - t[i] <= M * s[i][j])
m.setObjective(sum(x), GRB.MAXIMIZE)
m.optimize()

scheduled_fields = QTable(selected_fields)
selected_fields['scheduled_start_time'] = Time([v.x for v in t],format='mjd')
scheduled_fields['scheduled_start_time'].format = 'iso'
scheduled_fields['scheduled_end_time'] = scheduled_fields['scheduled_start_time'] + exposure_time
scheduled_fields = scheduled_fields[np.asarray([v.x for v in x], dtype =bool)]
scheduled_fields.sort('scheduled_start_time')
'''
delta = exposure_time.to_value(u.day)
M = (selected_fields['end_time'].max() - selected_fields['start_time'].min()).to_value(u.day).item()
m = Model('real_telescope')  #for docplex
m.parameters.timelimit = 60
t = [m.continuous_var(
        lb=(row['start_time'] - start_time).to_value(u.day),
        ub=(row['end_time'] - start_time - exposure_time).to_value(u.day),
        name=f't_{i}'
     ) for i, row in enumerate(selected_fields)]
x = [m.binary_var(name=f'x_{i}') for i in range(len(t))]
s = [[m.binary_var(name=f's_{i}_{j}') for j in range(i)] for i in range(len(t))]

for i in range(len(t)):
    for j in range(i):
        # Constraints for time sequencing between fields i and j
        m.add_constraint(t[i] + delta * x[i] - t[j] <= M * (1 - s[i][j]), ctname=f'constr1_{i}_{j}')
        m.add_constraint(t[j] + delta * x[j] - t[i] <= M * s[i][j], ctname=f'constr2_{i}_{j}')
m.maximize(m.sum(x))
solution = m.solve(log_output=True)

if solution:
    print("Optimization completed")
    scheduled_start_times = [solution.get_value(var) for var in t]
    scheduled_fields = QTable(selected_fields)
    scheduled_fields['scheduled_start_time'] = Time(scheduled_start_times, format='mjd')
    scheduled_fields['scheduled_start_time'].format = 'iso'
    scheduled_fields['scheduled_end_time'] = scheduled_fields['scheduled_start_time'] + exposure_time
    scheduled_fields = scheduled_fields[np.asarray([solution.get_value(var) for var in x], dtype=bool)]
    scheduled_fields.sort('scheduled_start_time')
else:
    print("No solution found")
fig, ax = plt.subplots()
ax.hlines(
    np.arange(len(scheduled_fields)), 
    scheduled_fields['scheduled_start_time'].mjd, 
    scheduled_fields['scheduled_end_time'].mjd,    
    colors='blue', linewidth=2) 
ax.set_yticks(np.arange(len(scheduled_fields)))
ax.set_yticklabels(scheduled_fields['field_id'].astype(str))
ax.set_xlabel('Observation time (MJD)')
ax.set_ylabel('Field ID')
plt.savefig('scheduler.png', dpi=300, bbox_inches='tight')
