import subprocess
import re
from astropy.time import Time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd
import time
import os

@dataclass
class SchedulerResult:
    cumulative_probability: float
    num_fields: float
    coverage: float  # Coverage percentage
    computation_time: float

def run_gwemopt_with_params(fits_file: str, start_time_: Time, end_time_: Time, 
                           filters: str = "g,i,r", exposure_times: str = "180.0,180.0,180.0") -> SchedulerResult:
    """Run GWEMOPT and extract relevant metrics"""
    start_time = time.time()
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
        
        # Extract metrics
        prob_matches = re.findall(r"Cumultative probability: ([\d.]+)", output)
        probability = float(prob_matches[-1]) if prob_matches else 0.0
        
        field_matches = re.findall(r"Number of fields scheduled: (\d+)", output)
        num_fields = int(field_matches[-1]) if field_matches else 0
        
        coverage_matches = re.findall(r"Coverage: ([\d.]+)", output)
        coverage = float(coverage_matches[-1]) if coverage_matches else 0.0
        
        computation_time = time.time() - start_time
        
        return SchedulerResult(
            cumulative_probability=probability,
            num_fields=num_fields,
            coverage=coverage,
            computation_time=computation_time
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"GWEMOPT command failed: {e}")

def run_milp_scheduler(skymap_file: str, num_revisits: int, exp_time: float = 180.0) -> SchedulerResult:
    """Run MILP scheduler with specified number of revisits"""
    start_time = time.time()
    
    # Your existing MILP scheduler code here
    # Modify to handle different numbers of revisits
    # Make sure to calculate coverage percentage
    
    # Example return (replace with actual values from your MILP implementation):
    return SchedulerResult(
        cumulative_probability=total_prob_covered,
        num_fields=len(selected_fields),
        coverage=coverage_percentage,
        computation_time=time.time() - start_time
    )

def process_skymap(skymap_file: str, skymap_number: int, revisit_scenarios: List[int]) -> dict:
    """Process a single skymap and return results for all methods"""
    results = {
        'skymap_number': skymap_number,
        'skymap_file': os.path.basename(skymap_file)
    }
    
    # Run GWEMOPT
    gwemopt_result = run_gwemopt_with_params(skymap_file)
    results.update({
        'gwemopt_probability': gwemopt_result.cumulative_probability,
        'gwemopt_fields': gwemopt_result.num_fields,
        'gwemopt_coverage': gwemopt_result.coverage,
        'gwemopt_time': gwemopt_result.computation_time
    })
    
    # Run MILP for each revisit scenario
    for num_revisits in revisit_scenarios:
        milp_result = run_milp_scheduler(skymap_file, num_revisits)
        results.update({
            f'milp_{num_revisits}visit_probability': milp_result.cumulative_probability,
            f'milp_{num_revisits}visit_fields': milp_result.num_fields,
            f'milp_{num_revisits}visit_coverage': milp_result.coverage,
            f'milp_{num_revisits}visit_time': milp_result.computation_time
        })
    
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
    directory_path = "/u/ywagh/test_skymaps/"
    create_results_csv(directory_path)