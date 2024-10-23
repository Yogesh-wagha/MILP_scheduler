
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
import pandas as pd
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
warnings.simplefilter('ignore', astroplan.TargetNeverUpWarning)
warnings.simplefilter('ignore', astroplan.TargetAlwaysUpWarning)
from gurobipy import GRB,Env, Model
env = Env(params={"WLSACCESSID": "c33d3a26-e111-4b61-b670-f9b9be9b4395",
                  "WLSSECRET": "001698e8-35d4-450f-8fbe-b63f304b8f2b",
                  "LICENSEID": 2533050})

directory_path = "/u/ywagh/test_skymaps/"
filelist = sorted([f for f in os.listdir(directory_path) if f.endswith('.gz')])
#probelm setup
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

observer = astroplan.Observer.at_site('Palomar')
night_horizon = -18 * u.deg
min_airmass = 2.5 * u.dimensionless_unscaled
airmass_horizon = (90 * u.deg - np.arccos(1 / min_airmass))

targets = field_grid['coord']
exposure_time = 1800 * u.second

hpx = HEALPix(nside=256, frame=ICRS())

optimization_times = []

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

    footprints = np.moveaxis(get_footprint(observable_fields['coord']).cartesian.xyz.value, 0, -1)
    footprints_healpix = [
        np.unique(np.concatenate([hp.query_polygon(hpx.nside, v) for v in footprint]))
        for footprint in tqdm(footprints)]
    prob = hp.ud_grade(skymap, hpx.nside, power=-2)

    t2 =time.time()
    print("problem setup",n, "completed")
    m = Model('real telescope',env=env)
    field_vars = [m.addVar(vtype=GRB.BINARY) for _ in range(len(footprints))]
    pixel_vars = [m.addVar(vtype=GRB.BINARY) for _ in range(hpx.npix)]
    footprints_healpix_inverse = [[] for _ in range(hpx.npix)]
    for field, pixels in enumerate(footprints_healpix):
        for pixel in pixels:
            footprints_healpix_inverse[pixel].append(field)
    for i_pixel, i_fields in enumerate(footprints_healpix_inverse):
        m.addConstr(sum(field_vars[i] for i in i_fields) >= pixel_vars[i_pixel])
    m.addConstr(sum(field_vars) <= 30)
    m.setObjective(np.dot(pixel_vars, prob), GRB.MAXIMIZE)
    m.optimize()
    print("optimization",n, "completed")
    t3 = time.time()
    time_to_optimize = t3 - t2
    optimization_times.append(time_to_optimize)
    total_prob_covered = m.ObjVal
    selected_fields = observable_fields[[v.x == 1 for v in field_vars]]

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
    plot_filename = os.path.basename(skymap_file)
    ax.grid()
    ax.imshow_hpx(prob, cmap='cylon')
    plt.text(0.05, 0.95, f'Total Probability Covered: {total_prob_covered:.2f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    save_path = '/u/ywagh/scheduler_results/no_slew_speed'
    full_path = os.path.join(save_path, f'ztf_coverage_no_slew_{n}.png')
    plt.savefig(full_path, dpi=300, bbox_inches='tight')

# m.setParam('TimeLimit', 60)
# delta = exposure_time.to_value(u.day)
# M = (selected_fields['end_time'].max() - selected_fields['start_time'].min()).to_value(u.day).item()

# t = [m.addVar(
#         lb=(row['start_time'] - start_time).to_value(u.day),
#         ub=(row['end_time'] - start_time - exposure_time).to_value(u.day),
#      ) for row in selected_fields]

# x = [m.addVar(vtype=GRB.BINARY) for _ in range(len(t))]
# s = [[m.addVar(vtype=GRB.BINARY) for j in range(i)] for i in range(len(t))]
# for i in range(len(t)):
#     for j in range(i):
#         m.addConstr(t[i] + delta * x[i] - t[j] <= M * (1 - s[i][j]))
#         m.addConstr(t[j] + delta * x[j] - t[i] <= M * s[i][j])
# m.setObjective(sum(x), GRB.MAXIMIZE)
# m.optimize()

# scheduled_fields = QTable(selected_fields)
# selected_fields['scheduled_start_time'] = Time([v.x for v in t],format='mjd')
# scheduled_fields['scheduled_start_time'].format = 'iso'
# scheduled_fields['scheduled_end_time'] = scheduled_fields['scheduled_start_time'] + exposure_time
# scheduled_fields = scheduled_fields[np.asarray([v.x for v in x], dtype =bool)]
# scheduled_fields.sort('scheduled_start_time')

for i in range(len(filelist)):
    print('scheduling event number',i)
    scheduler(os.path.join(directory_path, filelist[i]),i)

'''
plotting the ztf-coverage, while plotting the tiles that are being obsevred later as lighter tiles
'''
x_axis = range(len(optimization_times))
plt.figure(figsize=(10, 8))
plt.bar(x_axis, optimization_times, color='blue', edgecolor='black')
plt.xlabel('File Index')
plt.ylabel('Time to Optimize (seconds)')
plt.title('Optimization Time per skymap')
bar_plot_filename = 'Scheduler_optimization_times_bar_no_slew.png'
plt.savefig(bar_plot_filename, dpi=300, bbox_inches='tight')


image_folder = '/u/ywagh/scheduler_results/no_slew_speed'
pdf_filename = 'ztf_coverage_plots_combined.pdf'
pdf_full_path = os.path.join(image_folder, pdf_filename)
png_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
with PdfPages(pdf_full_path) as pdf:
    for png_file in sorted(png_files):
        img_path = os.path.join(image_folder, png_file)
        fig = figure()
        img = imread(img_path)
        imshow(img)
        axis('off')
        
        pdf.savefig(fig)
        plt.close(fig)  
print(f"All images saved to {pdf_full_path}")

