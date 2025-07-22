# Enhanced common utility functions for cavity filter optimization

import os
import time
import random
import numpy as np
import json
import matplotlib.pyplot as plt
import re

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

# Load parameter bounds from config
PARAMETER_CONFIG = load_parameter_config()
PARAMETER_BOUNDS = get_parameter_bounds_from_config(PARAMETER_CONFIG)
PARAMETER_ORDER = get_parameter_order_from_config(PARAMETER_CONFIG)

def generate_random_parameters(n_samples, parameter_bounds=None, config_file=None):
    """Generate random parameter sets within bounds"""
    if parameter_bounds is None:
        # Try to load from config file if specified
        if config_file is not None:
            config = load_parameter_config(config_file)
            parameter_bounds = get_parameter_bounds_from_config(config)
            parameter_order = get_parameter_order_from_config(config)
        else:
            parameter_bounds = PARAMETER_BOUNDS
            parameter_order = PARAMETER_ORDER
    else:
        # Use parameter order from global config or default
        parameter_order = PARAMETER_ORDER
    
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

def save_input_files(parameter_sets, input_dir="../Data1"):
    """Save parameter sets to input{i}.txt files"""
    ensure_directory_exists(input_dir)
    
    for i, param_vector in enumerate(parameter_sets):
        input_file_path = os.path.join(input_dir, f"input{i}.txt")
        with open(input_file_path, 'w') as f:
            # Write as space-separated values
            param_str = ' '.join([str(param) for param in param_vector])
            f.write(param_str)
    
    print(f"Generated {len(parameter_sets)} input files in {input_dir}")

def parse_fld_file_enhanced(fld_file_path):
    """Enhanced .fld file parsing with detailed grid information"""
    if not os.path.exists(fld_file_path):
        raise FileNotFoundError(f"FLD file not found: {fld_file_path}")
    
    try:
        with open(fld_file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header information
        header_info = {}
        if len(lines) > 0:
            header_line = lines[0].strip()
            # Extract grid bounds using regex
            min_match = re.search(r'Min:\s*\[([^\]]+)\]', header_line)
            max_match = re.search(r'Max:\s*\[([^\]]+)\]', header_line)
            
            if min_match and max_match:
                try:
                    min_coords = [float(x.replace('mm', '').strip()) for x in min_match.group(1).split(',')]
                    max_coords = [float(x.replace('mm', '').strip()) for x in max_match.group(1).split(',')]
                    header_info = {
                        'min_coords': min_coords,
                        'max_coords': max_coords,
                        'x_range': [min_coords[0], max_coords[0]],
                        'y_range': [min_coords[1], max_coords[1]], 
                        'z_range': [min_coords[2], max_coords[2]]
                    }
                except:
                    pass
        
        # Parse data (skip header lines)
        field_data = []
        data_started = False
        for line in lines[2:]:  # Skip first 2 lines (header + column names)
            line = line.strip()
            if not line or line.startswith('%') or line.startswith('#'):
                continue
                
            try:
                values = line.split()
                if len(values) >= 4:  # Expecting X, Y, Z, |E|
                    x, y, z, mag_e = float(values[0]), float(values[1]), float(values[2]), float(values[3])
                    field_data.append([x, y, z, mag_e])
            except ValueError:
                continue
        
        if len(field_data) == 0:
            raise ValueError("No valid field data found")
        
        field_array = np.array(field_data)
        
        # Organize data
        result = {
            'x': field_array[:, 0],
            'y': field_array[:, 1], 
            'z': field_array[:, 2],
            'mag_e': field_array[:, 3],
            'header_info': header_info,
            'total_points': len(field_array)
        }
        
        # Detect grid structure for rear plane (assuming constant X)
        unique_x = np.unique(result['x'])
        if len(unique_x) == 1:  # Rear plane with constant X
            unique_y = np.unique(result['y'])
            unique_z = np.unique(result['z'])
            result['grid_info'] = {
                'is_rear_plane': True,
                'x_plane': unique_x[0],
                'unique_y': unique_y,
                'unique_z': unique_z,
                'ny': len(unique_y),
                'nz': len(unique_z),
                'grid_complete': len(unique_y) * len(unique_z) == len(field_array)
            }
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Error parsing enhanced FLD file {fld_file_path}: {str(e)}")

def detect_valid_cavity_region(field_data, method="hybrid", field_threshold_percentile=1):
    """Detect valid cavity region for objective calculation
    
    Args:
        field_data: Dict from parse_fld_file_enhanced()
        method: "nan_only", "threshold_only", "hybrid" 
        field_threshold_percentile: Percentile threshold for low field filtering
    
    Returns:
        valid_mask: Boolean array indicating valid points for calculation
        stats: Dictionary with detection statistics
    """
    
    mag_e = field_data['mag_e']
    
    # Method 1: NaN and infinite value filtering
    nan_mask = ~np.isnan(mag_e) & ~np.isinf(mag_e) & (mag_e >= 0)
    
    # Method 2: Low field threshold filtering  
    threshold_mask = np.ones_like(mag_e, dtype=bool)
    if np.sum(nan_mask) > 0:  # Only if we have valid data
        valid_values = mag_e[nan_mask]
        if len(valid_values) > 0:
            threshold = np.percentile(valid_values, field_threshold_percentile)
            threshold_mask = mag_e > threshold
    
    # Combine methods based on selection
    if method == "nan_only":
        valid_mask = nan_mask
    elif method == "threshold_only":
        valid_mask = nan_mask & threshold_mask  # Still need basic nan filtering
    elif method == "hybrid":
        valid_mask = nan_mask & threshold_mask
    else:
        raise ValueError(f"Unknown detection method: {method}")
    
    # Calculate statistics
    stats = {
        'total_points': len(mag_e),
        'nan_filtered_points': np.sum(nan_mask),
        'threshold_filtered_points': np.sum(valid_mask),
        'valid_ratio': np.sum(valid_mask) / len(mag_e),
        'field_threshold': np.percentile(mag_e[nan_mask], field_threshold_percentile) if np.sum(nan_mask) > 0 else 0,
        'method_used': method
    }
    
    return valid_mask, stats

def calculate_enhanced_objective_value(fld_file_path, method="hybrid", field_threshold_percentile=1):
    """Calculate enhanced objective value using valid cavity region detection
    
    Args:
        fld_file_path: Path to .fld file
        method: Detection method ("nan_only", "threshold_only", "hybrid")
        field_threshold_percentile: Percentile threshold for filtering
    
    Returns:
        objective_value: Enhanced objective value (mean of valid region)
        detailed_stats: Dictionary with calculation details
    """
    
    try:
        # Parse field data
        field_data = parse_fld_file_enhanced(fld_file_path)
        
        # Detect valid region
        valid_mask, detection_stats = detect_valid_cavity_region(
            field_data, method=method, field_threshold_percentile=field_threshold_percentile
        )
        
        if np.sum(valid_mask) == 0:
            return None, {"error": "No valid cavity region detected", "detection_stats": detection_stats}
        
        # Calculate objective value on valid region
        valid_mag_e = field_data['mag_e'][valid_mask]
        objective_value = float(np.mean(valid_mag_e))
        
        # Detailed statistics
        detailed_stats = {
            'objective_value': objective_value,
            'valid_field_mean': objective_value,
            'valid_field_std': float(np.std(valid_mag_e)),
            'valid_field_min': float(np.min(valid_mag_e)),
            'valid_field_max': float(np.max(valid_mag_e)),
            'detection_stats': detection_stats,
            'grid_info': field_data.get('grid_info', {}),
            'valid_mask': valid_mask  # For visualization
        }
        
        return objective_value, detailed_stats
        
    except Exception as e:
        return None, {"error": f"Error calculating enhanced objective: {str(e)}"}

def visualize_valid_region(fld_file_path, method="hybrid", field_threshold_percentile=1, save_path=None):
    """Visualize the valid cavity region used for objective calculation"""
    
    try:
        # Calculate enhanced objective to get valid mask
        objective_value, stats = calculate_enhanced_objective_value(
            fld_file_path, method=method, field_threshold_percentile=field_threshold_percentile
        )
        
        if objective_value is None:
            print(f"Error: {stats.get('error', 'Unknown error')}")
            return
        
        # Parse field data for visualization
        field_data = parse_fld_file_enhanced(fld_file_path)
        valid_mask = stats['valid_mask']
        
        # Check if it's a rear plane (2D visualization)
        if not field_data.get('grid_info', {}).get('is_rear_plane', False):
            print("Warning: Not a rear plane - 3D visualization not implemented")
            return
        
        # Prepare 2D grid data
        grid_info = field_data['grid_info']
        unique_y = grid_info['unique_y']
        unique_z = grid_info['unique_z']
        ny, nz = grid_info['ny'], grid_info['nz']
        
        if not grid_info['grid_complete']:
            print("Warning: Incomplete grid data - visualization may be incorrect")
        
        # Reshape data
        Y = field_data['y'].reshape(ny, nz)
        Z = field_data['z'].reshape(ny, nz)
        MAG_E = field_data['mag_e'].reshape(ny, nz)
        VALID_MASK = valid_mask.reshape(ny, nz)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Original field data
        im1 = axes[0].imshow(MAG_E.T, extent=[Y.min()*1000, Y.max()*1000, Z.min()*1000, Z.max()*1000], 
                            aspect='auto', origin='lower', cmap='jet')
        axes[0].set_title(f'Original Field Data\n(All {field_data["total_points"]} points)')
        axes[0].set_xlabel('Y (mm)')
        axes[0].set_ylabel('Z (mm)')
        plt.colorbar(im1, ax=axes[0], label='|E| (V/m)')
        
        # Plot 2: Valid region mask
        im2 = axes[1].imshow(VALID_MASK.T.astype(float), extent=[Y.min()*1000, Y.max()*1000, Z.min()*1000, Z.max()*1000], 
                            aspect='auto', origin='lower', cmap='RdYlBu', vmin=0, vmax=1)
        axes[1].set_title(f'Valid Region Mask\n({stats["detection_stats"]["threshold_filtered_points"]} valid points)')
        axes[1].set_xlabel('Y (mm)')
        axes[1].set_ylabel('Z (mm)')
        plt.colorbar(im2, ax=axes[1], label='Valid (1) / Invalid (0)')
        
        # Plot 3: Field data with valid region only
        MAG_E_VALID = MAG_E.copy()
        MAG_E_VALID[~VALID_MASK] = np.nan  # Set invalid points to NaN for visualization
        
        im3 = axes[2].imshow(MAG_E_VALID.T, extent=[Y.min()*1000, Y.max()*1000, Z.min()*1000, Z.max()*1000], 
                            aspect='auto', origin='lower', cmap='jet')
        axes[2].set_title(f'Enhanced Calculation Region\nObjective = {objective_value:.3e} V/m')
        axes[2].set_xlabel('Y (mm)')
        axes[2].set_ylabel('Z (mm)')
        plt.colorbar(im3, ax=axes[2], label='|E| (V/m)')
        
        plt.tight_layout()
        
        # Print statistics
        print(f"\n📊 Enhanced Objective Calculation Results:")
        print(f"   Original method (all points): {np.mean(field_data['mag_e']):.3e} V/m")
        print(f"   Enhanced method ({method}): {objective_value:.3e} V/m") 
        print(f"   Valid points: {stats['detection_stats']['threshold_filtered_points']}/{stats['detection_stats']['total_points']} ({stats['detection_stats']['valid_ratio']:.1%})")
        print(f"   Field threshold: {stats['detection_stats']['field_threshold']:.3e} V/m")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Visualization saved: {save_path}")
        
        plt.show()
        
        return objective_value, stats
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        return None, None

# Legacy functions (preserved for compatibility)
def parse_fld_file(fld_file_path):
    """Parse .fld file and extract field data (legacy version)"""
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
    """Calculate objective value (rear plane |E| average) from .fld file (legacy version)"""
    try:
        field_data = parse_fld_file(fld_file_path)
        
        if field_data.size == 0:
            raise ValueError("Empty field data")
        
        # Assuming the .fld file contains [x, y, z, |E|] or similar
        # Extract |E| values (typically the last column)
        if field_data.ndim == 2 and field_data.shape[1] >= 4:
            e_magnitude = field_data[:, -1]  # Last column is |E|
        else:
            # If single column, assume it's |E| values
            e_magnitude = field_data.flatten()
        
        # Calculate area-weighted average (simple average for now)
        objective_value = np.mean(e_magnitude)
        
        return float(objective_value)
        
    except Exception as e:
        print(f"Error calculating objective value from {fld_file_path}: {str(e)}")
        return None

def collect_all_input_files(data_dir="../Data1"):
    """Collect all input{i}.txt files and return sorted list"""
    if not os.path.exists(data_dir):
        return []
    
    input_files = []
    for filename in os.listdir(data_dir):
        if filename.startswith("input") and filename.endswith(".txt"):
            input_files.append(os.path.join(data_dir, filename))
    
    # Sort by iteration number
    input_files.sort(key=lambda x: int(os.path.basename(x)[5:-4]))
    return input_files

def collect_all_fld_files(data_dir="../Data1"):
    """Collect all output{i}.fld files and return sorted list"""
    if not os.path.exists(data_dir):
        return []
    
    fld_files = []
    for filename in os.listdir(data_dir):
        if filename.startswith("output") and filename.endswith(".fld"):
            fld_files.append(os.path.join(data_dir, filename))
    
    # Sort by iteration number  
    fld_files.sort(key=lambda x: int(os.path.basename(x)[6:-4]))
    return fld_files

def integrate_input_output_to_dataset(data_dir="../Data1", dataset_file_path=None, use_enhanced_method=True):
    """Integrate input and output files to create dataset.txt"""
    if dataset_file_path is None:
        dataset_file_path = "../Data/dataset.txt"  # Default to Data folder
    
    input_files = collect_all_input_files(data_dir)
    fld_files = collect_all_fld_files(data_dir)
    
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
                if use_enhanced_method:
                    objective_value, _ = calculate_enhanced_objective_value(fld_file_path)
                else:
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
    
    method_str = "enhanced" if use_enhanced_method else "legacy"
    print(f"Created dataset with {processed_count} entries using {method_str} method: {dataset_file_path}")
    return processed_count

def ensure_directory_exists(dir_path):
    """Ensure directory exists, create if not"""
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

def monitor_file_creation(file_path, timeout=3600):
    """Monitor file creation with timeout"""
    start_time = time.time()
    while not os.path.exists(file_path):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for file: {file_path}")
        time.sleep(1)
    return True

def wait_for_simulation_completion(data_dir="../Data1", expected_count=300, timeout=72000):
    """Wait for all simulation outputs to be completed"""
    start_time = time.time()
    
    while True:
        fld_files = collect_all_fld_files(data_dir)
        current_count = len(fld_files)
        
        print(f"Progress: {current_count}/{expected_count} simulations completed")
        
        if current_count >= expected_count:
            print("All simulations completed!")
            break
            
        if time.time() - start_time > timeout:
            print(f"Timeout reached. Only {current_count}/{expected_count} completed.")
            break
            
        time.sleep(10)  # Check every 10 seconds
    
    return current_count

# =====================================================================
# TESTING AND DEMONSTRATION
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED OBJECTIVE CALCULATION TESTING")
    print("=" * 60)
    
    # Test file path - UPDATE THIS WITH YOUR ACTUAL .FLD FILE PATH
    test_fld_file = "../Data1/output0.fld"
    
    # Check if test file exists
    if not os.path.exists(test_fld_file):
        print(f"❌ Test file not found: {test_fld_file}")
        print("Please update the 'test_fld_file' variable with your actual .fld file path")
        exit(1)
    
    print(f"📁 Testing with file: {test_fld_file}")
    print()
    
    # Test 1: Compare methods
    print("🔬 Test 1: Comparing calculation methods")
    print("-" * 40)
    
    methods = ["nan_only", "threshold_only", "hybrid"]
    results = {}
    
    for method in methods:
        print(f"Testing method: {method}")
        obj_val, stats = calculate_enhanced_objective_value(test_fld_file, method=method)
        if obj_val is not None:
            results[method] = obj_val
            print(f"  Result: {obj_val:.3e} V/m")
            print(f"  Valid points: {stats['detection_stats']['threshold_filtered_points']}/{stats['detection_stats']['total_points']} ({stats['detection_stats']['valid_ratio']:.1%})")
        else:
            print(f"  Error: {stats.get('error', 'Unknown error')}")
        print()
    
    # Test 2: Compare with legacy method
    print("⚖️  Test 2: Enhanced vs Legacy comparison")
    print("-" * 40)
    
    legacy_result = calculate_objective_value_from_fld(test_fld_file)
    if legacy_result is not None:
        print(f"Legacy method (all points): {legacy_result:.3e} V/m")
        
        if "hybrid" in results:
            improvement = ((legacy_result - results["hybrid"]) / legacy_result) * 100
            print(f"Enhanced method (hybrid): {results['hybrid']:.3e} V/m")
            print(f"Difference: {improvement:+.2f}% {'(reduction)' if improvement > 0 else '(increase)'}")
    
    print()
    
    # Test 3: Visualization
    print("🎨 Test 3: Creating visualization")
    print("-" * 40)
    
    print("Generating visualization plots...")
    obj_val, stats = visualize_valid_region(
        test_fld_file, 
        method="hybrid", 
        field_threshold_percentile=1,
        save_path="enhanced_objective_analysis.png"
    )
    
    if obj_val is not None:
        print("✅ Visualization completed successfully!")
        print(f"Final enhanced objective value: {obj_val:.3e} V/m")
    else:
        print("❌ Visualization failed")
    
    print()
    print("=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)