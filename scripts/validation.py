# -*- coding: ascii -*-
# Validation functions for cavity filter optimization

import os
import numpy as np
from functions import (PARAMETER_BOUNDS, PARAMETER_ORDER, PARAMETER_CONFIG,
                      collect_all_input_files, collect_all_fld_files,
                      load_parameter_config, get_parameter_bounds_from_config,
                      get_parameter_order_from_config)

def validate_parameter_bounds(param_vector, config_file=None):
    """Validate parameter vector against bounds from config or default"""
    if len(param_vector) != 10:
        return False, f"Parameter vector must have 10 elements, got {len(param_vector)}"
    
    # Load parameter bounds and order
    if config_file is not None:
        config = load_parameter_config(config_file)
        parameter_bounds = get_parameter_bounds_from_config(config)
        parameter_order = get_parameter_order_from_config(config)
    else:
        parameter_bounds = PARAMETER_BOUNDS
        parameter_order = PARAMETER_ORDER
    
    # Create bounds list in correct order
    bounds_list = []
    param_names = []
    for param_name in parameter_order:
        if param_name in parameter_bounds:
            bounds_list.append(parameter_bounds[param_name])
            param_names.append(param_name)
        else:
            return False, f"Parameter {param_name} not found in bounds configuration"
    
    violations = []
    for i, (value, bounds, name) in enumerate(zip(param_vector, bounds_list, param_names)):
        if not (bounds[0] <= value <= bounds[1]):
            violations.append(f"{name}={value} not in range [{bounds[0]}, {bounds[1]}]")
    
    if violations:
        return False, "; ".join(violations)
    
    return True, "All parameters within bounds"

def validate_geometric_constraints(param_vector):
    """Validate geometric constraints for trapezoidal cavity"""
    if len(param_vector) < 4:
        return False, "Not enough parameters for geometric validation"
    
    top_width = param_vector[0]
    bottom_width = param_vector[1]
    height = param_vector[2]
    length = param_vector[3]
    
    constraints = []
    
    # Geometric constraint: bottom_width should be larger than top_width
    if bottom_width <= top_width:
        constraints.append(f"bottom_width ({bottom_width}) must be > top_width ({top_width})")
    
    # Reasonable aspect ratios
    if height / top_width > 2.0:
        constraints.append(f"Height/top_width ratio ({height/top_width:.2f}) too large")
    
    if length / bottom_width < 0.5:
        constraints.append(f"Length/bottom_width ratio ({length/bottom_width:.2f}) too small")
    
    # Tooth constraints
    if len(param_vector) >= 10:
        tooth_depth = param_vector[4]
        tooth_length = param_vector[5]
        tooth_width = param_vector[6]
        groove_length = param_vector[7]
        groove_width = param_vector[8]
        extension_factor = param_vector[9]
        
        # Tooth depth should not exceed reasonable fraction of height
        if tooth_depth > height * 0.8:
            constraints.append(f"tooth_depth ({tooth_depth}) too large for height ({height})")
        
        # Minimum feature size constraints
        min_feature_size = 0.05
        if tooth_length < min_feature_size:
            constraints.append(f"tooth_length ({tooth_length}) below minimum feature size")
        if tooth_width < min_feature_size:
            constraints.append(f"tooth_width ({tooth_width}) below minimum feature size")
        if groove_length < min_feature_size:
            constraints.append(f"groove_length ({groove_length}) below minimum feature size")
        if groove_width < min_feature_size:
            constraints.append(f"groove_width ({groove_width}) below minimum feature size")
    
    if constraints:
        return False, "; ".join(constraints)
    
    return True, "All geometric constraints satisfied"

def validate_fld_file_format(fld_file_path):
    """Validate .fld file format and basic content"""
    if not os.path.exists(fld_file_path):
        return False, f"File does not exist: {fld_file_path}"
    
    try:
        with open(fld_file_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            return False, "Empty file"
        
        # Check for data lines
        data_lines = 0
        for line in lines:
            line = line.strip()
            if line and not line.startswith('%') and not line.startswith('#'):
                try:
                    values = line.split()
                    float_values = [float(val) for val in values]
                    data_lines += 1
                except ValueError:
                    continue
        
        if data_lines == 0:
            return False, "No valid data lines found"
        
        return True, f"Valid .fld file with {data_lines} data lines"
        
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

def validate_objective_value_range(objective_value):
    """Validate objective value is in reasonable range"""
    if objective_value is None:
        return False, "Objective value is None"
    
    if not isinstance(objective_value, (int, float)):
        return False, f"Objective value must be numeric, got {type(objective_value)}"
    
    if np.isnan(objective_value) or np.isinf(objective_value):
        return False, "Objective value is NaN or infinite"
    
    # Reasonable range for electric field magnitude (V/m)
    if objective_value < 0:
        return False, f"Objective value cannot be negative: {objective_value}"
    
    if objective_value > 10000:  # Very high field strength
        return False, f"Objective value seems too high: {objective_value}"
    
    return True, f"Objective value {objective_value} is in reasonable range"

def check_simulation_completeness(data_dir="../Data1"):
    """Check completeness of simulation results"""
    input_files = collect_all_input_files(data_dir)
    fld_files = collect_all_fld_files(data_dir)
    
    print(f"Checking simulation completeness:")
    print(f"  Input files: {len(input_files)}")
    print(f"  Output .fld files: {len(fld_files)}")
    
    # Extract iteration numbers
    input_iterations = set()
    for input_file in input_files:
        basename = os.path.basename(input_file)
        iteration = int(basename[5:-4])  # Extract from "input{i}.txt"
        input_iterations.add(iteration)
    
    fld_iterations = set()
    for fld_file in fld_files:
        basename = os.path.basename(fld_file)
        iteration = int(basename[6:-4])  # Extract from "output{i}.fld"
        fld_iterations.add(iteration)
    
    # Find missing outputs
    missing_outputs = input_iterations - fld_iterations
    extra_outputs = fld_iterations - input_iterations
    
    print(f"  Missing outputs: {len(missing_outputs)}")
    if missing_outputs:
        print(f"    Missing iterations: {sorted(list(missing_outputs))}")
    
    print(f"  Extra outputs: {len(extra_outputs)}")
    if extra_outputs:
        print(f"    Extra iterations: {sorted(list(extra_outputs))}")
    
    # Validate file formats
    valid_count = 0
    invalid_files = []
    
    for fld_file in fld_files:
        is_valid, message = validate_fld_file_format(fld_file)
        if is_valid:
            valid_count += 1
        else:
            invalid_files.append((os.path.basename(fld_file), message))
    
    print(f"  Valid .fld files: {valid_count}/{len(fld_files)}")
    if invalid_files:
        print("  Invalid files:")
        for filename, error in invalid_files:
            print(f"    {filename}: {error}")
    
    # Overall completeness
    completeness_ratio = len(fld_iterations) / len(input_iterations) if input_iterations else 0
    print(f"  Completeness: {completeness_ratio:.1%}")
    
    return {
        "input_count": len(input_files),
        "output_count": len(fld_files),
        "missing_outputs": list(missing_outputs),
        "extra_outputs": list(extra_outputs),
        "valid_fld_count": valid_count,
        "invalid_files": invalid_files,
        "completeness_ratio": completeness_ratio
    }

def validate_dataset_file(dataset_file_path):
    """Validate dataset.txt file format and content"""
    if not os.path.exists(dataset_file_path):
        return False, f"Dataset file does not exist: {dataset_file_path}"
    
    try:
        with open(dataset_file_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            return False, "Empty dataset file"
        
        valid_entries = 0
        invalid_entries = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            try:
                values = [float(x) for x in line.split()]
                
                # Should have 11 values: 10 parameters + 1 objective value
                if len(values) != 11:
                    invalid_entries.append(f"Line {i+1}: Expected 11 values, got {len(values)}")
                    continue
                
                # Validate parameter bounds
                param_vector = values[:10]
                objective_value = values[10]
                
                param_valid, param_msg = validate_parameter_bounds(param_vector)
                if not param_valid:
                    invalid_entries.append(f"Line {i+1}: {param_msg}")
                    continue
                
                obj_valid, obj_msg = validate_objective_value_range(objective_value)
                if not obj_valid:
                    invalid_entries.append(f"Line {i+1}: {obj_msg}")
                    continue
                
                valid_entries += 1
                
            except ValueError as e:
                invalid_entries.append(f"Line {i+1}: Parse error - {str(e)}")
        
        total_entries = len([line for line in lines if line.strip()])
        
        if invalid_entries:
            error_summary = f"Dataset validation: {valid_entries}/{total_entries} valid entries"
            error_details = "; ".join(invalid_entries[:5])  # Show first 5 errors
            if len(invalid_entries) > 5:
                error_details += f" ... and {len(invalid_entries)-5} more errors"
            return False, f"{error_summary}. Errors: {error_details}"
        
        return True, f"Valid dataset with {valid_entries} entries"
        
    except Exception as e:
        return False, f"Error reading dataset file: {str(e)}"

def run_full_validation(data_dir="../Data1", dataset_file_path=None):
    """Run comprehensive validation of all data"""
    print("=" * 60)
    print("FULL VALIDATION REPORT")
    print("=" * 60)
    
    # Check simulation completeness
    completeness = check_simulation_completeness(data_dir)
    
    # Validate dataset if provided
    if dataset_file_path is None:
        dataset_file_path = "../Data/dataset.txt"  # Default to Data folder
        
    if os.path.exists(dataset_file_path):
        print(f"\nDataset validation:")
        dataset_valid, dataset_msg = validate_dataset_file(dataset_file_path)
        print(f"  {dataset_msg}")
    
    print("=" * 60)
    
    return completeness