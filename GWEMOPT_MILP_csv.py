from dataclasses import dataclass
from typing import List, Dict, Tuple
import time
import os
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
from matplotlib.pyplot import imread, figure, imshow, axis
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import datetime as dt
import pandas as pd
import warnings
from docplex.mp.model import Model
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
warnings.simplefilter('ignore', astroplan.TargetNeverUpWarning)
warnings.simplefilter('ignore', astroplan.TargetAlwaysUpWarning)

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

@dataclass
class SchedulerResult:
    cumulative_probability: float
    num_fields: float
    coverage: float  # Coverage percentage
    computation_time: float

def extract_field_numbers(output_text: str) -> Tuple[int, int]:
    """
    Extract the number of observed fields and total fields from output text.
    Example: "14/15 fields were observed at least twice" -> (14, 15)
    
    Returns:
        Tuple[int, int]: (observed_fields, total_fields)
    """
    pattern = r"(\d+)/(\d+) fields were observed"
    match = re.search(pattern, output_text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0

def run_gwemopt_with_params(fits_file: str, start_time_: Time, end_time_: Time, 
                           filters: str = "g,i,r", exposure_times: str = "180.0,180.0,180.0") -> SchedulerResult:
    """Run GWEMOPT and extract relevant metrics"""
    start_time_1 = time.time()
    tobs = (end_time_-start_time_)
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
        "--Tobs", f"0.0,{tobs}" 
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr
        
        # Extract metrics
        prob_matches = re.findall(r"Cumultative probability: ([\d.]+)", output)
        probability = float(prob_matches[-1]) if prob_matches else 0.0
        
        observed_fields, total_fields = extract_field_numbers(output)
        
        coverage_matches = re.findall(r"Cumultative area: ([\d.]+)", output)
        coverage = float(coverage_matches[-1]) if coverage_matches else 0.0
        
        computation_time = time.time() - start_time_1
        
        return SchedulerResult(
            cumulative_probability=probability,
            num_fields=total_fields,
            coverage=coverage,
            computation_time=computation_time
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"GWEMOPT command failed: {e}")

def dummy(skymap_file):
    skymap, metadata = read_sky_map(skymap_file)
    event_time = Time(metadata['gps_time'], format='gps').utc
    event_time.format = 'iso'
    if observer.is_night(event_time, horizon=night_horizon):
        start_time = event_time
    else:
        start_time = observer.sun_set_time(
            event_time, horizon=night_horizon, which='next')
    end_time = observer.sun_rise_time(
        start_time, horizon=night_horizon, which='next')
    return start_time,end_time

def run_milp_scheduler(skymap_file: str, num_revisits: int, exp_time: float = 180.0) -> SchedulerResult:
    """Run MILP scheduler with specified number of revisits"""
    cadence = 30         #minutes
    cadence_days = cadence / (60 * 24)
    hpx = HEALPix(nside=256, frame=ICRS())
    exposure_time = exp_time * u.second
    exposure_time_day = exposure_time.to_value(u.day)
    skymap, metadata = read_sky_map(skymap_file)
    # print("SkyMap",n, "loaded")
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
    k = int(np.floor((max_end - min_start)/(num_revisits*exposure_time.to(u.day))))
    start_time_2 = time.time()
    print(k," number of exposures for revisit available")
    m1 = Model('max coverage problem for revisit')
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
    solution1 = m1.solve(log_output=False)
    
    if solution1:
        total_prob_covered = solution1.objective_value
        if total_prob_covered>=0.05:
            selected_fields = observable_fields[[solution1.get_value(v) == 1 for v in field_vars]]
            delta = exposure_time.to_value(u.day)
            limit_duration = ((end_time-start_time).value*(1-1/(num_revisits))) + delta
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
                for v in range(num_revisits)] 
                for i in range(len(selected_fields))]

            tc = [[m2.continuous_var(
                lb=(row['start_time'] - start_time).to_value(u.day),
                ub=(row['end_time'] - start_time - exposure_time).to_value(u.day),
                name=f"start_time_field_{i}_visit_{v}")
                for v in range(num_revisits)] 
                for i, row in enumerate(selected_fields)]

            visit_transition_times = [m2.continuous_var(
                lb=0,ub=M,name=f"visit_transition_{v}")
                                    for v in range(num_revisits-1)]  

            # for i in range(len(selected_fields)):
            #     priority_factor = priority_weights[i]
            #     for v in range(num_revisits):
            #         # Tighter time windows for high-priority fields
            #         if priority_factor > 1.5:  # High priority fields
            #             m2.add_constraint(
            #                 tc[i][v] <= (selected_fields['end_time'][i] - start_time - exposure_time).to_value(u.day) * 0.8,
            #                 ctname=f"priority_time_window_upper_{i}_{v}"
            #             )

            # Isolating visits
            for v in range(1, num_revisits):
                for i in range(len(selected_fields)):
                    m2.add_constraint(tc[i][v-1] + delta * x[i][v-1] <= visit_transition_times[v-1],
                        ctname=f"visit_end_{i}_visit_{v-1}")
                    m2.add_constraint(tc[i][v] >= visit_transition_times[v-1],
                        ctname=f"visit_start_{i}_visit_{v}")

            # Cadence constraints
            for i in range(len(selected_fields)):
                for v in range(1, num_revisits):
                    m2.add_constraint(tc[i][v] - tc[i][v-1] >= (cadence_days+delta) * (x[i][v] + x[i][v-1] - 1),
                        ctname=f"cadence_constraint_field_{i}_visits_{v}")

            #non-overlapping
            for v in range(num_revisits):
                for i in range(len(selected_fields)):
                    for j in range(i):
                        m2.add_constraint(tc[i][v] + delta * x[i][v] + slew_time_day[i][j] - tc[j][v] <= M * (2 - x[i][v] - x[j][v]),
                                        ctname=f"non_overlapping_cross_fields_{i}_{j}_visits_{v}")
                        m2.add_constraint(tc[j][v] + delta * x[j][v] + slew_time_day[i][j] - tc[i][v] <= M * (-1 + x[i][v] + x[j][v]),
                            ctname=f"non_overlapping_cross_fields_{j}_{i}_visits_{v}")

            # objective = m2.sum([
            #     probabilities[i] * priority_weights[i] * x[i][v]
            #     for i in range(len(selected_fields))
            #     for v in range(num_revisits)
            # ])

            # high_prob_threshold = np.percentile(probabilities, 75)
            # high_prob_fields = [i for i, p in enumerate(probabilities) if p >= high_prob_threshold]

            # if high_prob_fields:
            #     m2.add_constraint(
            #         m2.sum([x[i][v] for i in high_prob_fields for v in range(num_revisits)]) >= 
            #         len(high_prob_fields) * num_revisits * 0.5,  # At least 50% coverage of high-prob fields
            #         ctname="high_probability_coverage"
            #     )
            # m2.maximize(objective)

            m2.parameters.timelimit = 120  # Increased time limit
            m2.parameters.mip.tolerances.mipgap = 0.01
            m2.parameters.emphasis.mip = 2
            m2.parameters.mip.strategy.variableselect = 4  # Strong branching
            m2.parameters.mip.strategy.probe = 3  # Aggressive probing
            m2.maximize(m2.sum([probabilities_revisit[i] * x[i][v]
                                for i in range(len(selected_fields))
                                for v in range(num_revisits)]))
            solution2 = m2.solve(log_output=False)
            if solution2:
                cumulative_probability_ = solution2.objective_value/(num_revisits)
                scheduled_fields = selected_fields[[solution2.get_value(v) == 1 for v in x]]
                coverage_percentage = 1729
            else:
                print("no solution to revisit problem for revisists = {num_revisits}")
    # Your existing MILP scheduler code here
    # Modify to handle different numbers of revisits
    # Make sure to calculate coverage percentage
    else:
        print("no solution to coverage problem")
    # Example return (replace with actual values from your MILP implementation):
    return SchedulerResult(
        cumulative_probability=cumulative_probability_,
        num_fields=len(scheduled_fields),
        coverage=coverage_percentage,
        computation_time=time.time() - start_time_2,
    )

def process_skymap(skymap_file: str, skymap_number: int, revisit_scenarios: List[int]) -> dict:
    """Process a single skymap and return results for all methods"""
    results = {
        'skymap_number': skymap_number,
        'skymap_file': os.path.basename(skymap_file)
    }
    for num_revisits in revisit_scenarios:
        milp_result = run_milp_scheduler(skymap_file, num_revisits)
        results.update({
            f'milp_{num_revisits}visit_probability': milp_result.cumulative_probability if milp_result.cumulative_probability is not None else np.nan,
            f'milp_{num_revisits}visit_fields': milp_result.num_fields if milp_result.num_fields is not None else np.nan,
            f'milp_{num_revisits}visit_coverage': milp_result.coverage if milp_result.coverage is not None else np.nan,
            f'milp_{num_revisits}visit_time': milp_result.computation_time if milp_result.computation_time is not None else np.nan
            })
    # Run GWEMOPT
    start_time_global,end_time_global = dummy(skymap_file)
    gwemopt_result = run_gwemopt_with_params(skymap_file, start_time_=start_time_global,end_time_=end_time_global)
    results.update({
        'gwemopt_probability': gwemopt_result.cumulative_probability,
        'gwemopt_fields': gwemopt_result.num_fields,
        'gwemopt_coverage': gwemopt_result.coverage,
        'gwemopt_time': gwemopt_result.computation_time
    })
    
    # Run MILP for each revisit scenario
    
    
    return results

def create_results_csv(directory_path: str, output_file: str = 'scheduling_results.csv'):
    """Process all skymaps and create CSV file with results"""
    skymap_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.gz')])
    revisit_scenarios = [1, 2, 3]  # Test 1, 2, and 3 revisits
    
    all_results = []
    for i, skymap_file in enumerate(skymap_files):
        print(f"Processing skymap {i+1}/{len(skymap_files)}: {skymap_file}")
        try:
            results = process_skymap(
                os.path.join(directory_path, skymap_file),
                i + 1,
                revisit_scenarios
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error processing {skymap_file}: {e}")
            continue
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Print summary statistics
    print("\nSummary Statistics:")
    methods = ['gwemopt'] + [f'milp_{n}visit' for n in revisit_scenarios]
    for method in methods:
        print(f"\n{method.upper()} Statistics:")
        print(f"Average probability: {df[f'{method}_probability'].mean():.4f}")
        print(f"Average fields used: {df[f'{method}_fields'].mean():.1f}")
        print(f"Average coverage: {df[f'{method}_coverage'].mean():.4f}")
        print(f"Average computation time: {df[f'{method}_time'].mean():.2f} seconds")

if __name__ == "__main__":
    directory_path = "/u/ywagh/test_skymaps"
    create_results_csv(directory_path)