import os
import json
import random
import numpy as np

# Default parameter bounds (fallback if JSON not available)
DEFAULT_PARAMETER_BOUNDS = {
    "top_width": [100.0, 200.0],        # mm
    "bottom_width": [200.0, 400.0],     # mm  
    "height": [50.0, 70.0],             # mm
    "length": [300.0, 400.0],           # mm
    "tooth_depth": [0.1, 19.9],         # mm
    "tooth_length": [0.05, 50.0],       # mm
    "tooth_width": [0.05, 50.0],        # mm
    "groove_length": [0.05, 100.0],     # mm
    "groove_width": [0.05, 100.0],      # mm
    "extension_factor": [1.0, 2.0]      # dimensionless
}

def load_parameter_config(config_file="parameters_config.json"):
    """Load parameter configuration from JSON file"""
    try:
        # Look in current directory first, then in scripts directory
        possible_paths = [
            config_file,
            os.path.join("scripts", config_file),
            os.path.join(os.path.dirname(__file__), config_file)
        ]
        
        config_path = None
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        else:
            print(f"Warning: Config file {config_file} not found in any expected location")
            return None
    except Exception as e:
        print(f"Error loading config file {config_file}: {str(e)}")
        return None

def get_parameter_bounds_from_config(config=None):
    """Convert JSON config to old format for backward compatibility"""
    if config is None:
        config = load_parameter_config()
    
    if config is None:
        return DEFAULT_PARAMETER_BOUNDS
    
    bounds = {}
    parameter_bounds = config.get("parameter_bounds", {})
    
    for param_name, param_info in parameter_bounds.items():
        bounds[param_name] = [param_info["lower_bound"], param_info["upper_bound"]]
    
    return bounds

def get_parameter_order_from_config(config=None):
    """Get parameter order from config"""
    if config is None:
        config = load_parameter_config()
    
    if config is None:
        return list(DEFAULT_PARAMETER_BOUNDS.keys())
    
    return config.get("parameter_order", list(DEFAULT_PARAMETER_BOUNDS.keys()))

def generate_random_parameters(n_samples, parameter_bounds=None, config_file=None):
    """Generate random parameter sets within bounds"""
    if parameter_bounds is None:
        # Try to load from config file if specified
        if config_file is not None:
            config = load_parameter_config(config_file)
            parameter_bounds = get_parameter_bounds_from_config(config)
            parameter_order = get_parameter_order_from_config(config)
        else:
            # Load parameter bounds from config
            parameter_config = load_parameter_config()
            parameter_bounds = get_parameter_bounds_from_config(parameter_config)
            parameter_order = get_parameter_order_from_config(parameter_config)
    else:
        # Use parameter order from global config or default
        parameter_order = get_parameter_order_from_config()
    
    parameter_sets = []
    
    # Create bounds list in correct order
    bounds_list = []
    for param_name in parameter_order:
        if param_name in parameter_bounds:
            bounds_list.append(parameter_bounds[param_name])
        else:
            raise ValueError(f"Parameter {param_name} not found in bounds")
    
    for i in range(n_samples):
        param_vector = []
        for bounds in bounds_list:
            value = random.uniform(bounds[0], bounds[1])
            param_vector.append(value)
        parameter_sets.append(param_vector)
    
    return parameter_sets

def parse_fld_file(fld_file_path):
    """Parse .fld file and extract field data"""
    if not os.path.exists(fld_file_path):
        raise FileNotFoundError(f"FLD file not found: {fld_file_path}")
    
    field_data = []
    try:
        with open(fld_file_path, 'r') as f:
            lines = f.readlines()
            
        # Skip header lines and parse data
        data_started = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith('%') or line.startswith('#'):
                continue
                
            # Look for data section
            if not data_started:
                try:
                    # Try to parse as float data
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
        raise RuntimeError(f"Error parsing FLD file {fld_file_path}: {str(e)}")
    
    return np.array(field_data)

def calculate_objective_value_from_fld(fld_file_path):
    """Calculate objective value by averaging all values in .fld file (excluding NaN)"""
    try:
        field_data = parse_fld_file(fld_file_path)
        
        if field_data.size == 0:
            raise ValueError("Empty field data")
        
        # Flatten all data to get all numerical values
        all_values = field_data.flatten()
        
        # Filter out NaN values
        valid_values = all_values[~np.isnan(all_values)]
        
        if valid_values.size == 0:
            raise ValueError("No valid (non-NaN) values found in field data")
        
        # Calculate average of all valid values
        objective_value = np.mean(valid_values)
        
        return float(objective_value)
        
    except Exception as e:
        print(f"Error calculating objective value from {fld_file_path}: {str(e)}")
        return None

def ensure_directory_exists(dir_path):
    """Ensure directory exists, create if not"""
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

def integrate_input_output_to_dataset(data_dir="../Data1", dataset_file_path=None):
    """Integrate input and output files to create dataset.txt"""
    if dataset_file_path is None:
        dataset_file_path = "../Data/dataset.txt"  # Default to Data folder
    
    # Collect all input and fld files
    input_files = []
    fld_files = []
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    for filename in os.listdir(data_dir):
        if filename.startswith("input") and filename.endswith(".txt"):
            input_files.append(os.path.join(data_dir, filename))
        elif filename.startswith("output") and filename.endswith(".fld"):
            fld_files.append(os.path.join(data_dir, filename))
    
    # Sort by iteration number
    input_files.sort(key=lambda x: int(os.path.basename(x)[5:-4]))
    fld_files.sort(key=lambda x: int(os.path.basename(x)[6:-4]))
    
    if len(input_files) == 0:
        raise ValueError(f"No input files found in {data_dir}")
    
    if len(fld_files) == 0:
        raise ValueError(f"No .fld files found in {data_dir}")
    
    print(f"Found {len(input_files)} input files and {len(fld_files)} output files")
    
    # Create dataset entries
    dataset_entries = []
    processed_count = 0
    
    for input_file in input_files:
        # Extract iteration number
        basename = os.path.basename(input_file)
        iteration = int(basename[5:-4])  # Extract number from "input{i}.txt"
        
        # Find corresponding fld file
        fld_file_path = os.path.join(data_dir, f"output{iteration}.fld")
        
        if os.path.exists(fld_file_path):
            try:
                # Read input parameters
                with open(input_file, 'r') as f:
                    param_line = f.read().strip()
                    param_vector = [float(x) for x in param_line.split()]
                
                # Calculate objective value
                objective_value = calculate_objective_value_from_fld(fld_file_path)
                
                if objective_value is not None:
                    # Create dataset entry: [param1, param2, ..., param10, objective_value]
                    dataset_entry = param_vector + [objective_value]
                    dataset_entries.append(dataset_entry)
                    processed_count += 1
                else:
                    print(f"Warning: Could not calculate objective value for iteration {iteration}")
                    
            except Exception as e:
                print(f"Error processing iteration {iteration}: {str(e)}")
        else:
            print(f"Warning: Missing output file for iteration {iteration}")
    
    # Save dataset
    ensure_directory_exists(os.path.dirname(dataset_file_path))
    
    with open(dataset_file_path, 'w') as f:
        for entry in dataset_entries:
            entry_str = ' '.join([str(val) for val in entry])
            f.write(entry_str + '\n')
    
    print(f"Created dataset with {processed_count} entries: {dataset_file_path}")
    return processed_count

def update_single_data_to_dataset(param_vector, fld_file_path, dataset_file_path=None, append=True):
    """Update a single data entry to dataset.txt
    
    Args:
        param_vector: List of 10 parameters
        fld_file_path: Path to the corresponding .fld file
        dataset_file_path: Path to dataset file (default: "../Data/dataset.txt")
        append: If True, append to existing dataset; if False, overwrite
    
    Returns:
        bool: True if successful, False otherwise
    """
    if dataset_file_path is None:
        dataset_file_path = "../Data/dataset.txt"
    
    try:
        # Validate input
        if len(param_vector) != 10:
            print(f"Error: Parameter vector must have 10 elements, got {len(param_vector)}")
            return False
        
        if not os.path.exists(fld_file_path):
            print(f"Error: FLD file not found: {fld_file_path}")
            return False
        
        # Calculate objective value
        objective_value = calculate_objective_value_from_fld(fld_file_path)
        
        if objective_value is None:
            print(f"Error: Could not calculate objective value from {fld_file_path}")
            return False
        
        # Create dataset entry
        dataset_entry = param_vector + [objective_value]
        entry_str = ' '.join([str(val) for val in dataset_entry])
        
        # Ensure directory exists
        ensure_directory_exists(os.path.dirname(dataset_file_path))
        
        # Write to dataset file
        mode = 'a' if append else 'w'
        with open(dataset_file_path, mode) as f:
            f.write(entry_str + '\n')
        
        print(f"Successfully added entry to dataset: {os.path.basename(dataset_file_path)}")
        print(f"  Parameters: {param_vector}")
        print(f"  Objective value: {objective_value}")
        
        return True
        
    except Exception as e:
        print(f"Error updating dataset: {str(e)}")
        return False