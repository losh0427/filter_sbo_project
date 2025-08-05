import os
import shutil
import time
import csv
import numpy as np
import glob
from typing import List, Tuple, Optional
import torch


# ============================================================================
# File Management Module
# ============================================================================

def move_files_data1_to_data(source_dir, destination_dir, initial_count, final_count):
    """
    Move files from Data1 directory to Data directory
    Maintains file moving logic compatible with original pipeline
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        
    moved_count = 0
    for i in range(initial_count, final_count):
        # Move input files
        input_file = os.path.join(source_dir, f"input{i}.txt")
        if os.path.exists(input_file):
            shutil.move(input_file, destination_dir)
            moved_count += 1
            
        # Move output files
        output_file = os.path.join(source_dir, f"output{i}.fld")
        if os.path.exists(output_file):
            shutil.move(output_file, destination_dir)
            moved_count += 1
            
    print(f"Moved {moved_count} files from {source_dir} to {destination_dir}")
    return moved_count


def manage_file_counters(file_record_path):
    """
    Manage file counters
    Returns (final_count, initial_count)
    """
    if not os.path.exists(file_record_path):
        # Initialize file
        with open(file_record_path, 'w', encoding='utf-8') as f:
            f.write("0\n0")
        return 0, 0
        
    with open(file_record_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) >= 2:
            final_count = int(lines[0].strip())
            initial_count = int(lines[1].strip())
        else:
            final_count, initial_count = 0, 0
            
    return final_count, initial_count


def update_file_counters(file_record_path, final_count, initial_count):
    """Update file counters"""
    with open(file_record_path, 'w', encoding='utf-8') as f:
        f.write(f"{final_count}\n{initial_count}")


def execute_data_archiving(source_dir, destination_dir, file_record_path):
    """
    Execute data archiving process
    """
    final_count, initial_count = manage_file_counters(file_record_path)
    moved_count = move_files_data1_to_data(source_dir, destination_dir, initial_count, final_count)
    return final_count, initial_count


# ============================================================================
# Parameter Processing Module
# ============================================================================

def get_parameter_bounds():
    """
    Get 10-dimensional parameter bounds
    Based on definitions in dataflow_spec.md
    """
    bounds = [
        [100.0, 200.0],   # top_width
        [200.0, 400.0],   # bottom_width
        [50.0, 70.0],     # height
        [300.0, 400.0],   # length
        [0.1, 19.9],      # tooth_depth
        [0.05, 50.0],     # tooth_length
        [0.05, 50.0],     # tooth_width
        [0.05, 100.0],    # groove_length
        [0.05, 100.0],    # groove_width
        [1.0, 2.0]        # extension_factor
    ]
    return torch.tensor(bounds, dtype=torch.float64)


def validate_parameters(params):
    """
    Validate parameters against boundary constraints
    """
    bounds = get_parameter_bounds()
    params = torch.tensor(params, dtype=torch.float64) if not isinstance(params, torch.Tensor) else params
    
    if params.dim() == 1:
        params = params.unsqueeze(0)
        
    valid_flags = []
    for i, param_set in enumerate(params):
        valid = True
        for j, val in enumerate(param_set):
            if val < bounds[j, 0] or val > bounds[j, 1]:
                valid = False
                break
        valid_flags.append(valid)
        
    return valid_flags


def clip_parameters_to_bounds(params):
    """
    Clip parameters to valid range
    """
    bounds = get_parameter_bounds()
    params = torch.tensor(params, dtype=torch.float64) if not isinstance(params, torch.Tensor) else params
    
    if params.dim() == 1:
        params = params.unsqueeze(0)
        
    clipped_params = torch.zeros_like(params)
    for i, param_set in enumerate(params):
        for j, val in enumerate(param_set):
            clipped_params[i, j] = torch.clamp(val, bounds[j, 0], bounds[j, 1])
            
    return clipped_params.squeeze() if clipped_params.shape[0] == 1 else clipped_params


# ============================================================================
# File I/O Processing Module
# ============================================================================

def write_input_file(filepath, parameters):
    """
    Write input file
    Parameters: 10-dimensional vector
    """
    if not isinstance(parameters, (list, np.ndarray, torch.Tensor)):
        raise ValueError("Parameters must be in list, array or tensor format")
        
    # Convert to list format
    if isinstance(parameters, torch.Tensor):
        parameters = parameters.tolist()
    elif isinstance(parameters, np.ndarray):
        parameters = parameters.tolist()
        
    # Write to file
    with open(filepath, 'w') as f:
        for i, param in enumerate(parameters):
            f.write(str(param))
            if i < len(parameters) - 1:
                f.write(" ")


def wait_for_output_file(filepath, check_interval=3, max_wait_time=7200):
    """
    Wait for output file to be generated
    Uses while loop + time.sleep, no timeout interruption
    """
    print(f"Waiting for file: {filepath}")
    start_time = time.time()
    
    while not os.path.exists(filepath):
        time.sleep(check_interval)
        
    # Wait a bit more after file exists to ensure writing is complete
    time.sleep(5)
    
    elapsed_time = time.time() - start_time
    print(f"File {os.path.basename(filepath)} is ready, wait time: {elapsed_time:.2f}s")
    return elapsed_time


def parse_fld_file(fld_filepath):
    """
    Parse .fld file and extract electric field data
    Based on calculate_objective_value_from_fld logic in functions.py
    """
    if not os.path.exists(fld_filepath):
        raise FileNotFoundError(f"FLD file does not exist: {fld_filepath}")
    
    field_data = []
    try:
        with open(fld_filepath, 'r') as f:
            lines = f.readlines()
            
        # Skip header and parse data
        data_started = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith('%') or line.startswith('#'):
                continue
                
            if not data_started:
                try:
                    # Try to parse as floating point data
                    values = line.split()
                    float_values = [float(val) for val in values]
                    data_started = True
                    field_data.append(float_values)
                except ValueError:
                    continue
            else:
                try:
                    values = line.split()
                    float_values = [float(val) for val in values]
                    field_data.append(float_values)
                except ValueError:
                    continue
                    
    except Exception as e:
        raise RuntimeError(f"Failed to parse FLD file {fld_filepath}: {str(e)}")
    
    return np.array(field_data)


def calculate_objective_value(fld_filepath):
    """
    Calculate objective function value from .fld file
    Returns exit plane electric field average value
    """
    try:
        field_data = parse_fld_file(fld_filepath)
        
        if field_data.size == 0:
            raise ValueError("Electric field data is empty")
        
        # Flatten all data
        all_values = field_data.flatten()
        
        # Filter NaN values
        valid_values = all_values[~np.isnan(all_values)]
        
        if valid_values.size == 0:
            raise ValueError("No valid electric field data")
        
        # Calculate average
        objective_value = np.mean(valid_values)
        
        return float(objective_value)
        
    except Exception as e:
        print(f"Failed to calculate objective function value {fld_filepath}: {str(e)}")
        return None


# ============================================================================
# Data Loading Module
# ============================================================================

def load_historical_data(data_filepath):
    """
    Load historical experimental data
    Returns (X, y) where X is parameter matrix, y is objective function value vector
    """
    if not os.path.exists(data_filepath):
        print(f"Data file does not exist: {data_filepath}")
        return None, None
        
    X_data = []
    y_data = []
    
    try:
        with open(data_filepath, 'r') as f:
            for line_num, line in enumerate(f.readlines(), 1):
                try:
                    parts = line.strip().split()
                    if len(parts) < 11:  # Need at least 10 parameters + 1 objective value
                        continue
                        
                    # First 10 are parameters
                    params = [float(x) for x in parts[:10]]
                    # Last one is objective function value
                    objective = float(parts[-1])
                    
                    X_data.append(params)
                    y_data.append(objective)
                    
                except ValueError as e:
                    print(f"Skipping invalid data line {line_num}: {str(e)}")
                    continue
                    
        if len(X_data) == 0:
            print("No valid data loaded")
            return None, None
            
        X = torch.tensor(X_data, dtype=torch.float64)
        y = torch.tensor(y_data, dtype=torch.float64)
        
        print(f"Loaded historical data: {X.shape[0]} samples, {X.shape[1]} parameter dimensions")
        return X, y
        
    except Exception as e:
        print(f"Failed to load data: {str(e)}")
        return None, None


def append_data_to_file(data_filepath, params, objective_value):
    """
    Append new data to data file
    """
    try:
        # Ensure correct parameter format
        if isinstance(params, torch.Tensor):
            params = params.tolist()
        elif isinstance(params, np.ndarray):
            params = params.tolist()
            
        # Build data line
        data_line = ' '.join([str(p) for p in params]) + f' {objective_value}\n'
        
        # Append to file
        with open(data_filepath, 'a') as f:
            f.write(data_line)
            
        print(f"New data appended to {data_filepath}")
        
    except Exception as e:
        print(f"Failed to append data: {str(e)}")


# ============================================================================
# Experiment Recording Module
# ============================================================================

def record_iteration_time(time_filepath, elapsed_time):
    """Record iteration time"""
    with open(time_filepath, 'a') as f:
        f.write(f"{elapsed_time}\n")


def record_model_performance(model_time_path, iteration, total_time, build_time, search_time, dataset_size, sample_num):
    """Record model performance statistics"""
    record_str = (f"iteration {iteration}, total_time {total_time:.2f}, "
                 f"build_time {build_time:.2f}, search_time {search_time:.2f}, "
                 f"dataset_num {dataset_size}, sample_num {sample_num}\n")
    
    with open(model_time_path, 'a') as f:
        f.write(record_str)
    
    print(record_str.strip())


def get_current_best_result(data_filepath):
    """
    Get current best result from data file
    Returns (best_params, best_value, best_index)
    """
    X, y = load_historical_data(data_filepath)
    if X is None or y is None:
        return None, None, -1
        
    best_idx = torch.argmin(y)  
    best_params = X[best_idx]
    best_value = y[best_idx].item()
    
    return best_params, best_value, best_idx.item()


# ============================================================================
# Utility Functions
# ============================================================================

def ensure_directory_exists(dir_path):
    """Ensure directory exists"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")


def cleanup_temp_files(temp_dir, pattern="*.tmp"):
    """Clean up temporary files"""
    temp_files = glob.glob(os.path.join(temp_dir, pattern))
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except:
            pass


def format_time_duration(seconds):
    """Format time display"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}min"
    else:
        return f"{seconds/3600:.2f}h"


def print_iteration_summary(iteration, best_value, current_value, dataset_size):
    """Print iteration summary"""
    print(f"\n=== Iteration {iteration} Summary ===")
    print(f"Current best value: {best_value:.6f}")
    print(f"Current objective value: {current_value:.6f}")
    print(f"Dataset size: {dataset_size}")
    print("=" * 30)


# ============================================================================
# Random Parameter Generation
# ============================================================================

def generate_random_parameters(n_samples=300, config_file=None):
    """
    Generate random parameter sets within bounds
    Returns list of 10-dimensional parameter vectors
    """
    bounds = get_parameter_bounds()
    
    parameter_sets = []
    for i in range(n_samples):
        # Generate random parameters within bounds
        random_params = []
        for j in range(bounds.shape[0]):
            min_val, max_val = bounds[j, 0].item(), bounds[j, 1].item()
            random_val = np.random.uniform(min_val, max_val)
            random_params.append(random_val)
        parameter_sets.append(random_params)
    
    return parameter_sets


# ============================================================================
# Data Integration Functions
# ============================================================================

def integrate_input_output_to_dataset(data_dir, dataset_file_path):
    """
    Integrate input{i}.txt and output{i}.fld files into dataset.txt
    Returns number of successfully processed entries
    """
    if not os.path.exists(data_dir):
        print(f"Data directory does not exist: {data_dir}")
        return 0
    
    # Ensure output directory exists
    dataset_dir = os.path.dirname(dataset_file_path)
    if dataset_dir and not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    processed_count = 0
    dataset_entries = []
    
    # Find all input files
    input_files = []
    for filename in os.listdir(data_dir):
        if filename.startswith("input") and filename.endswith(".txt"):
            # Extract number from filename
            try:
                num_str = filename[5:-4]  # Remove "input" and ".txt"
                file_num = int(num_str)
                input_files.append((file_num, filename))
            except ValueError:
                continue
    
    # Sort by file number
    input_files.sort(key=lambda x: x[0])
    
    print(f"Found {len(input_files)} input files")
    
    for file_num, input_filename in input_files:
        input_filepath = os.path.join(data_dir, input_filename)
        output_filepath = os.path.join(data_dir, f"output{file_num}.fld")
        
        # Check if both files exist
        if not os.path.exists(output_filepath):
            print(f"Warning: Missing output file for input{file_num}")
            continue
        
        try:
            # Read parameters from input file
            with open(input_filepath, 'r') as f:
                param_line = f.read().strip()
                params = [float(x) for x in param_line.split()]
            
            if len(params) != 10:
                print(f"Warning: Invalid parameter count in input{file_num}: {len(params)}")
                continue
            
            # Calculate objective value from output file
            objective_value = calculate_objective_value(output_filepath)
            
            if objective_value is None:
                print(f"Warning: Failed to calculate objective for output{file_num}")
                continue
            
            # Create dataset entry
            dataset_line = ' '.join([str(p) for p in params]) + f' {objective_value}'
            dataset_entries.append(dataset_line)
            processed_count += 1
            
            print(f"Processed: input{file_num}.txt -> objective: {objective_value:.6f}")
            
        except Exception as e:
            print(f"Error processing input{file_num}: {str(e)}")
            continue
    
    # Write dataset file
    if dataset_entries:
        with open(dataset_file_path, 'w') as f:
            for entry in dataset_entries:
                f.write(entry + '\n')
        
        print(f"Dataset created: {dataset_file_path}")
        print(f"Total entries: {processed_count}")
    else:
        print("No valid entries to write")
    
    return processed_count


def get_next_input_number(data_dir):
    """
    Get the next available input file number (original function)
    """
    max_num = -1
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.startswith("input") and filename.endswith(".txt"):
                try:
                    num_str = filename[5:-4]  # Remove "input" and ".txt"
                    file_num = int(num_str)
                    max_num = max(max_num, file_num)
                except ValueError:
                    continue
    
    return max_num + 1


# ============================================================================
# Shared Counter Management Module  
# ============================================================================

def read_iteration_counter(counter_file_path):
    """
    Read current iteration counter from file
    Returns the next input file number to be generated
    """
    try:
        if not os.path.exists(counter_file_path):
            # Initialize counter file if it doesn't exist
            with open(counter_file_path, 'w') as f:
                f.write("0")
            return 0
            
        with open(counter_file_path, 'r') as f:
            counter_str = f.read().strip()
            if not counter_str:
                return 0
            return int(counter_str)
            
    except (ValueError, IOError) as e:
        print(f"Warning: Failed to read counter file {counter_file_path}: {str(e)}")
        print("Using default counter value: 0")
        return 0


def update_iteration_counter(counter_file_path, next_number):
    """
    Update iteration counter to specified number
    Uses atomic write (temp file + rename) for safety
    """
    try:
        # Ensure directory exists
        counter_dir = os.path.dirname(counter_file_path)
        if counter_dir and not os.path.exists(counter_dir):
            os.makedirs(counter_dir)
        
        # Atomic write: write to temp file first
        temp_file_path = counter_file_path + ".tmp"
        with open(temp_file_path, 'w') as f:
            f.write(str(next_number))
        
        # Rename temp file to actual file (atomic on most systems)
        if os.path.exists(counter_file_path):
            os.remove(counter_file_path)
        os.rename(temp_file_path, counter_file_path)
        
        print(f"Updated iteration counter to: {next_number}")
        
    except (IOError, OSError) as e:
        print(f"Warning: Failed to update counter file {counter_file_path}: {str(e)}")
        # Clean up temp file if it exists
        temp_file_path = counter_file_path + ".tmp"
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass


def get_next_input_number_from_counter(counter_file_path):
    """
    Get next input number from shared counter file
    This replaces the original get_next_input_number function for pipeline usage
    """
    return read_iteration_counter(counter_file_path)