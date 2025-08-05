#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cavity Filter Global Optimization Pipeline - Fixed Version
AI-driven parameter optimization using Bayesian optimization with GP surrogate models
Simplified version with immediate file processing and clean state management
"""

import os
import time
import argparse
import torch
import numpy as np
import shutil

# Import custom modules
from models import GPSurrogateModel, AcquisitionFunction, ModelTrainer
import functions as fn

# ============================================================================
# SEARCH TIME CONFIGURATION - Phase 1 Implementation
# ============================================================================

class SearchTimeConfig:
    """Simple search time configuration"""
    SEARCH_TIME_BUDGET = 180        # 3 minutes for search (adjustable)
    EARLY_STOP_THRESHOLD = 10       # Stop after N consecutive no-improvements  
    MIN_IMPROVEMENT = 1e-6          # Minimum improvement threshold
    SINGLE_OPT_TIMEOUT = 30         # Single optimization timeout (seconds)
    LOG_FILE = "log.txt"            # Unified log file name


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cavity Filter Global Optimization')
    parser.add_argument('-p', '--path', 
                       help='Working directory path', 
                       default='C:/Users/USER/Desktop/wave/project/')
    parser.add_argument('-d', '--data', 
                       help='Data file name', 
                       default='dataset')
    parser.add_argument('--max_iterations', 
                       help='Maximum number of iterations', 
                       type=int, default=300)
    parser.add_argument('--acquisition_func', 
                       help='Acquisition function type', 
                       choices=['EI', 'UCB', 'MES'], 
                       default='EI')
    return parser.parse_args()


def initialize_paths_and_directories(base_path, data_name):
    """
    Initialize paths and directories for the optimization process
    
    Args:
        base_path: Base working directory path
        data_name: Dataset file name (without extension)
    
    Returns:
        dict: Dictionary containing all necessary file paths
    """
    paths = {
        'base_path': base_path,
        'data_path': os.path.join(base_path, 'Data', f'{data_name}.txt'),
        'data1_dir': os.path.join(base_path, 'Data1'),
        'data_dir': os.path.join(base_path, 'Data'),
        'time_record': os.path.join(base_path, 'round_time.txt'),
        'model_time_record': os.path.join(base_path, 'model_time.txt'),
        'counter_file': os.path.join(base_path, 'current_input_counter.txt')
    }
    
    # Ensure necessary directories exist
    for dir_path in [paths['data1_dir'], paths['data_dir']]:
        fn.ensure_directory_exists(dir_path)
    
    return paths


def verify_initialization(paths):
    """
    Verify that the system is properly initialized
    
    Args:
        paths: Dictionary containing file paths
    
    Returns:
        tuple: (success, error_message, initial_data_count)
    """
    # Check if dataset exists and is valid
    if not os.path.exists(paths['data_path']):
        return False, f"Dataset file not found: {paths['data_path']}", 0
    
    # Load and verify dataset
    X, y = fn.load_historical_data(paths['data_path'])
    if X is None or y is None:
        return False, "Dataset file exists but contains no valid data", 0
    
    if X.shape[0] < 2:
        return False, f"Dataset contains only {X.shape[0]} samples, need at least 2 for GP training", X.shape[0]
    
    # Check counter file
    if not os.path.exists(paths['counter_file']):
        return False, f"Counter file not found: {paths['counter_file']}", X.shape[0]
    
    # Verify Data1 is clean (ready for new iterations)
    if os.path.exists(paths['data1_dir']):
        existing_files = [f for f in os.listdir(paths['data1_dir']) 
                         if f.startswith(('input', 'output')) and 
                         (f.endswith('.txt') or f.endswith('.fld'))]
        if existing_files:
            return False, f"Data1 is not clean, contains {len(existing_files)} files. Run integrate_existing_data.py first.", X.shape[0]
    
    return True, "System properly initialized", X.shape[0]


def get_search_time_simple(config=None):
    """
    Simple search time allocation - Phase 1 Implementation
    
    Args:
        config: SearchTimeConfig instance (optional)
    
    Returns:
        int: Search time budget in seconds
    """
    if config is None:
        return SearchTimeConfig.SEARCH_TIME_BUDGET
    return config.SEARCH_TIME_BUDGET


def generate_candidate_simple_with_early_stop(gp_model, acq_func, bounds, max_time, config=None):
    """
    Simplified candidate generation with early stopping - Phase 1 Implementation
    
    Args:
        gp_model: Trained GP surrogate model
        acq_func: Acquisition function instance
        bounds: Parameter bounds tensor
        max_time: Maximum search time in seconds
        config: SearchTimeConfig instance (optional)
    
    Returns:
        tuple: (best_candidate, best_acq_value, elapsed_time, search_attempts)
    """
    if config is None:
        config = SearchTimeConfig()
    
    start_time = time.time()
    best_candidate = None
    best_acq_value = -float('inf')
    no_improvement_count = 0
    search_attempts = 0
    
    print(f"  Starting simplified search with {max_time}s budget...")
    print(f"  Early stop after {config.EARLY_STOP_THRESHOLD} consecutive no-improvements")
    
    # Strategy 1: Initial optimization attempt
    try:
        candidate, acq_value = acq_func.optimize(
            bounds=bounds, 
            q=1,
            num_restarts=3,
            raw_samples=10
        )
        best_candidate = candidate[0]
        best_acq_value = acq_value.item() if hasattr(acq_value, 'item') else acq_value
        search_attempts = 1
        print(f"  Initial optimization successful: {best_acq_value:.6f}")
        
    except Exception as e:
        print(f"  Initial optimization failed: {str(e)}")
        # Fallback to random candidate
        n_dims = bounds.shape[0]
        random_point = torch.rand(n_dims, dtype=torch.float64)
        best_candidate = bounds[:, 0] + random_point * (bounds[:, 1] - bounds[:, 0])
        best_acq_value = 0.0
        search_attempts = 1
        print(f"  Using random fallback candidate")
    
    # Early stopping search loop
    while (time.time() - start_time) < max_time:
        # Check single optimization timeout
        single_opt_start = time.time()
        
        try:
            # Single optimization attempt with minimal parameters for speed
            candidate, acq_value = acq_func.optimize(
                bounds=bounds, 
                q=1,
                num_restarts=1,  # Minimal for speed
                raw_samples=5    # Minimal for speed
            )
            
            search_attempts += 1
            current_acq_value = acq_value.item() if hasattr(acq_value, 'item') else acq_value
            
            # Check for improvement
            if current_acq_value > best_acq_value + config.MIN_IMPROVEMENT:
                best_candidate = candidate[0]
                best_acq_value = current_acq_value
                no_improvement_count = 0
                print(f"  Better candidate found (attempt {search_attempts}): {best_acq_value:.6f}")
            else:
                no_improvement_count += 1
                
            # Early stopping condition
            if no_improvement_count >= config.EARLY_STOP_THRESHOLD:
                print(f"  Early stop: {no_improvement_count} consecutive no-improvements")
                break
                
            # Single optimization timeout check
            single_opt_time = time.time() - single_opt_start
            if single_opt_time > config.SINGLE_OPT_TIMEOUT:
                print(f"  Single optimization timeout ({single_opt_time:.1f}s)")
                break
                
        except Exception as e:
            search_attempts += 1
            no_improvement_count += 1
            # Don't break on individual failures, just count as no improvement
            if no_improvement_count >= config.EARLY_STOP_THRESHOLD:
                print(f"  Early stop due to consecutive failures")
                break
    
    elapsed_time = time.time() - start_time
    
    # Final safety check
    if best_candidate is None:
        print(f"  Emergency fallback: generating random candidate")
        n_dims = bounds.shape[0]
        random_point = torch.rand(n_dims, dtype=torch.float64)
        best_candidate = bounds[:, 0] + random_point * (bounds[:, 1] - bounds[:, 0])
        best_acq_value = 0.0
    
    # Ensure candidate is within bounds
    best_candidate = fn.clip_parameters_to_bounds(best_candidate)
    
    print(f"  Search completed: {search_attempts} attempts in {elapsed_time:.1f}s")
    print(f"      Final acquisition value: {best_acq_value:.6f}")
    
    return best_candidate, best_acq_value, elapsed_time, search_attempts


def execute_hfss_evaluation_and_wait(candidate_point, paths):
    """
    Execute HFSS evaluation with shared counter management and wait for completion
    
    Args:
        candidate_point: Optimized parameter vector
        paths: Dictionary containing file paths
    
    Returns:
        tuple: (success, objective_value, wait_time, input_number) or (False, None, 0, None) if failed
    """
    try:
        # Get current input number from shared counter
        input_number = fn.get_next_input_number_from_counter(paths['counter_file'])
        
        # Write input file
        input_filepath = os.path.join(paths['data1_dir'], f'input{input_number}.txt')
        fn.write_input_file(input_filepath, candidate_point)
        print(f"  Generated input file: input{input_number}.txt")
        
        # CRITICAL: Update counter immediately after writing input file
        # This ensures HFSS script and pipeline stay synchronized
        fn.update_iteration_counter(paths['counter_file'], input_number + 1)
        print(f"  Updated counter to: {input_number + 1}")
        
        # Wait for corresponding output file
        output_filepath = os.path.join(paths['data1_dir'], f'output{input_number}.fld')
        print(f"  Waiting for HFSS to generate: output{input_number}.fld")
        print(f"  This iteration will not proceed until HFSS completes...")
        
        wait_time = fn.wait_for_output_file(output_filepath)
        print(f"  HFSS completed! Output file ready after {wait_time:.2f}s")
        
        # Calculate objective function value
        objective_value = fn.calculate_objective_value(output_filepath)
        
        if objective_value is None:
            print(f"  Warning: Unable to calculate objective value, using default 0.0")
            objective_value = 0.0
        else:
            print(f"  Objective function value: {objective_value:.6f}")
        
        return True, objective_value, wait_time, input_number
        
    except Exception as e:
        print(f"  ❌ HFSS evaluation failed: {str(e)}")
        return False, None, 0, None


def integrate_and_archive_files(input_number, objective_value, candidate_point, paths):
    """
    Integrate the new data point and archive files immediately
    
    Args:
        input_number: Input file number
        objective_value: Calculated objective value
        candidate_point: Parameter vector
        paths: Dictionary containing file paths
    
    Returns:
        bool: Success status
    """
    try:
        # Step 1: Append new data to dataset
        print(f"  Integrating data into dataset...")
        fn.append_data_to_file(paths['data_path'], candidate_point, objective_value)
        
        # Step 2: Move completed files to archive immediately
        print(f"  Archiving completed files...")
        
        input_source = os.path.join(paths['data1_dir'], f'input{input_number}.txt')
        output_source = os.path.join(paths['data1_dir'], f'output{input_number}.fld')
        
        input_dest = os.path.join(paths['data_dir'], f'input{input_number}.txt')
        output_dest = os.path.join(paths['data_dir'], f'output{input_number}.fld')
        
        # Move input file
        if os.path.exists(input_source):
            if os.path.exists(input_dest):
                os.remove(input_dest)  # Remove existing file in archive
            shutil.move(input_source, input_dest)
            print(f"    Moved input{input_number}.txt to archive")
        else:
            print(f"    Warning: input{input_number}.txt not found for archiving")
        
        # Move output file
        if os.path.exists(output_source):
            if os.path.exists(output_dest):
                os.remove(output_dest)  # Remove existing file in archive
            shutil.move(output_source, output_dest)
            print(f"    Moved output{input_number}.fld to archive")
        else:
            print(f"    Warning: output{input_number}.fld not found for archiving")
        
        # Step 3: Verify Data1 is clean
        remaining_files = []
        if os.path.exists(paths['data1_dir']):
            remaining_files = [f for f in os.listdir(paths['data1_dir']) 
                              if f.startswith(('input', 'output')) and 
                              (f.endswith('.txt') or f.endswith('.fld'))]
        
        if remaining_files:
            print(f"    Warning: {len(remaining_files)} files still in Data1")
        else:
            print(f"    Data1 is clean and ready for next iteration")
        
        return True
        
    except Exception as e:
        print(f"  Failed to integrate and archive files: {str(e)}")
        return False


def single_iteration_cycle(iteration, paths, gp_model, acq_func, bounds):
    """
    Execute single optimization iteration cycle with immediate processing
    Updated for Phase 1 & 2: Simplified search + Unified logging
    
    Args:
        iteration: Current iteration number
        paths: Dictionary containing file paths
        gp_model: GP surrogate model instance
        acq_func: Acquisition function instance
        bounds: Parameter bounds tensor
    
    Returns:
        tuple: (success, objective_value)
    """
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration} - STARTING")
    print(f"{'='*60}")
    iteration_start_time = time.time()
    
    # Initialize variables for logging
    model_train_time = 0.0
    search_elapsed = 0.0
    search_attempts = 0
    wait_time = 0.0
    
    # Step 1: Load historical data and train model
    print("STEP 1: LOADING DATA AND TRAINING GP MODEL")
    print("-" * 40)
    model_start_time = time.time()
    
    X, y = fn.load_historical_data(paths['data_path'])
    if X is None or y is None:
        print("  No historical data available")
        print("  Counter NOT updated due to iteration failure")
        return False, 0.0
    
    print(f"  Loaded {X.shape[0]} historical samples")
    
    # Train GP model
    trainer = ModelTrainer()
    success = trainer.train_with_retry(gp_model, X, y)
    if not success:
        print("  GP model training failed")
        print("  Counter NOT updated due to iteration failure")
        return False, 0.0
        
    model_train_time = time.time() - model_start_time
    print(f"  Model training completed in {model_train_time:.2f}s")
    
    # Step 2: Build acquisition function
    print("\nSTEP 2: BUILDING ACQUISITION FUNCTION")
    print("-" * 40)
    
    # Diagnose data before building acquisition function
    print(f"  Dataset diagnosis:")
    print(f"    Samples: {X.shape[0]}, Dimensions: {X.shape[1]}")
    print(f"    Objective range: [{y.min():.6f}, {y.max():.6f}]")
    print(f"    Objective std: {y.std():.6f}")
    
    best_f = y.min()  # Current best value (minimize objective for cavity filter)
    
    try:
        acq_func.build(gp_model, best_f=best_f, bounds=bounds)
        print(f"  Acquisition function built (current best: {best_f:.6f})")
    except Exception as e:
        print(f"  Acquisition function building failed: {str(e)}")
        print("  Counter NOT updated due to iteration failure")
        return False, 0.0
    
    # Step 3: Generate candidate points with simplified search
    print("\nSTEP 3: OPTIMIZING ACQUISITION FUNCTION (SIMPLIFIED)")
    print("-" * 40)
    search_time = get_search_time_simple()
    print(f"  Search budget: {search_time}s")
    
    candidate_point, acq_value, search_elapsed, search_attempts = generate_candidate_simple_with_early_stop(
        gp_model, acq_func, bounds, search_time
    )
    print(f"  Generated candidate point (acquisition value: {acq_value:.6f})")
    
    # Step 4: HFSS evaluation - WAIT FOR COMPLETION
    print("\nSTEP 4: HFSS EVALUATION AND WAITING")
    print("-" * 40)
    print("  CRITICAL: This iteration WAITS for HFSS completion!")
    
    success, objective_value, wait_time, input_number = execute_hfss_evaluation_and_wait(
        candidate_point, paths
    )
    
    if not success:
        print("  HFSS evaluation failed, skipping this iteration")
        print("  Counter NOT updated due to iteration failure")
        return False, 0.0
    
    print(f"  HFSS evaluation completed in {wait_time:.2f}s")
    
    # Step 5: IMMEDIATE integration and archiving
    print("\nSTEP 5: IMMEDIATE DATA INTEGRATION AND ARCHIVING")
    print("-" * 40)
    
    archive_success = integrate_and_archive_files(
        input_number, objective_value, candidate_point, paths
    )
    
    if not archive_success:
        print("  Warning: Archiving failed, but continuing...")
    else:
        print("  Data integrated and files archived successfully")
    
    # Get updated best results for logging
    best_params, best_value, best_idx = fn.get_current_best_result(paths['data_path'])
    
    # Step 6: Record statistics with unified logging - Phase 2 Implementation
    print("\nSTEP 6: RECORDING STATISTICS")
    print("-" * 40)
    
    total_time = time.time() - iteration_start_time
    
    # Check if this iteration improved the best result
    improvement = objective_value < best_f if best_params is not None else True
    
    # Prepare unified log data
    log_data = {
        'iteration': iteration,
        'gp_time': model_train_time,
        'search_time': search_elapsed,
        'hfss_time': wait_time,
        'total_time': total_time,
        'obj_value': objective_value,
        'best_value': best_value if best_params is not None else objective_value,
        'dataset_size': X.shape[0] + 1,
        'improvement': improvement,
        'search_attempts': search_attempts
    }
    
    # Record to unified log
    log_path = os.path.join(paths['base_path'], SearchTimeConfig.LOG_FILE)
    fn.record_to_unified_log(log_path, log_data)
    
    # Keep legacy functions for backward compatibility (with deprecation warnings)
    fn.record_model_performance(
        paths['model_time_record'], iteration, total_time, 
        model_train_time, search_elapsed, X.shape[0] + 1, 1
    )
    fn.record_iteration_time(paths['time_record'], wait_time)
    
    # Step 7: Display results
    print("\nSTEP 7: ITERATION SUMMARY")
    print("-" * 40)
    
    if best_params is not None:
        improvement_status = "NEW BEST!" if objective_value < best_value else "No improvement"
        print(f"  Current iteration result: {objective_value:.6f}")
        print(f"  Overall best result: {best_value:.6f} (at entry {best_idx + 1})")
        print(f"  Status: {improvement_status}")
        print(f"  Total dataset size: {X.shape[0] + 1} samples")
        print(f"  Processed input file: input{input_number}.txt")
        print(f"  Search attempts: {search_attempts}")
    
    # Step 8: Update counter ONLY after successful iteration completion
    print("\nSTEP 8: UPDATING COUNTER AFTER SUCCESSFUL ITERATION")
    print("-" * 40)
    next_counter = input_number + 1
    fn.update_iteration_counter(paths['counter_file'], next_counter)
    print(f"  Counter updated to: {next_counter}")
    print(f"  Next iteration will use: input{next_counter}.txt")
    
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration} COMPLETED - {total_time:.2f}s total")
    print(f"  HFSS time: {wait_time:.2f}s | Model time: {model_train_time:.2f}s | Search time: {search_elapsed:.2f}s")
    print(f"  Search attempts: {search_attempts} | Time budget: {search_time}s")
    print(f"  Data1 status: Clean and ready for next iteration")
    print(f"  Counter status: Updated to {next_counter}")
    print(f"{'='*60}")
    
    return True, objective_value


def diagnose_optimization_failure(gp_model, bounds):
    """
    Diagnose why acquisition function optimization might be failing
    """
    print("\nDIAGNOSING OPTIMIZATION FAILURE")
    print("-" * 40)
    
    try:
        X_train = gp_model.training_data['X']
        y_train = gp_model.training_data['y']
        
        print(f"Training data diagnosis:")
        print(f"  X shape: {X_train.shape}")
        print(f"  y shape: {y_train.shape}")
        print(f"  X range: [{X_train.min():.3f}, {X_train.max():.3f}]")
        print(f"  y range: [{y_train.min():.6f}, {y_train.max():.6f}]")
        print(f"  y variance: {y_train.var():.6f}")
        
        # Test GP prediction on a few random points
        print(f"Testing GP predictions:")
        n_test = 3
        test_X = torch.rand(n_test, bounds.shape[0], dtype=torch.float64)
        for i in range(bounds.shape[0]):
            test_X[:, i] = bounds[i, 0] + test_X[:, i] * (bounds[i, 1] - bounds[i, 0])
        
        mean, std = gp_model.predict(test_X)
        for i in range(n_test):
            print(f"  Test point {i+1}: mean={mean[i]:.6f}, std={std[i]:.6f}")
            
        # Check for numerical issues
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            print("  GP predictions contain NaN/Inf - numerical instability detected")
        elif std.min() < 1e-10:
            print("  GP uncertainty is extremely low - model may be overconfident")
        elif std.max() > 1e6:
            print("  GP uncertainty is extremely high - model may be poorly calibrated")
        else:
            print("  GP predictions appear numerically stable")
            
    except Exception as e:
        print(f"  Diagnosis failed: {e}")


def main():
    """Main execution function for the optimization pipeline"""
    
    # Parse command line arguments
    args = parse_arguments()
    print(f"Starting Cavity Filter Global Optimization")
    print(f"Working directory: {args.path}")
    print(f"Dataset: {args.data}, Max iterations: {args.max_iterations}")
    print(f"Acquisition function: {args.acquisition_func}")
    
    # Initialize paths and directories
    paths = initialize_paths_and_directories(args.path, args.data)
    print(f"\nPath configuration:")
    for key, value in paths.items():
        print(f"  {key}: {value}")
    
    # Verify system initialization
    print(f"\nVerifying system initialization...")
    init_success, init_message, data_count = verify_initialization(paths)
    
    if not init_success:
        print(f"INITIALIZATION FAILED: {init_message}")
        print(f"\nSOLUTION:")
        print(f"1. Make sure you have run integrate_existing_data.py first")
        print(f"2. Ensure dataset.txt exists with valid data")
        print(f"3. Ensure Data1 directory is clean")
        print(f"4. Check that current_input_counter.txt exists")
        return
    
    print(f"System properly initialized")
    print(f"  Dataset contains: {data_count} samples")
    print(f"  Counter file: Ready")
    print(f"  Data1 directory: Clean")
    
    # Show current counter status
    current_counter = fn.read_iteration_counter(paths['counter_file'])
    print(f"  Next input number: {current_counter}")
    
    # Initialize model components with configuration display
    print(f"\nInitializing AI model components...")
    print(f"Search Configuration - Phase 1 & 2:")
    print(f"  Search time budget: {SearchTimeConfig.SEARCH_TIME_BUDGET}s")
    print(f"  Early stop threshold: {SearchTimeConfig.EARLY_STOP_THRESHOLD} consecutive no-improvements")
    print(f"  Single optimization timeout: {SearchTimeConfig.SINGLE_OPT_TIMEOUT}s")
    print(f"  Unified log file: {SearchTimeConfig.LOG_FILE}")
    
    gp_model = GPSurrogateModel()
    acq_func = AcquisitionFunction(func_type=args.acquisition_func)
    bounds = fn.get_parameter_bounds()
    
    print(f"  Parameter space: {bounds.shape[0]} dimensions")
    print(f"  Acquisition function: {args.acquisition_func}")
    print(f"  GP surrogate model: Ready")
    
    # Main iteration loop
    print(f"\nSTARTING OPTIMIZATION ITERATIONS")
    print(f"Target: {args.max_iterations} iterations maximum")
    print(f"{'='*80}")
    
    start_time = time.time()
    successful_iterations = 0
    consecutive_failures = 0
    
    for iteration in range(args.max_iterations):
        try:
            success, current_objective = single_iteration_cycle(
                iteration, paths, gp_model, acq_func, bounds
            )
            
            if success:
                successful_iterations += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                print(f"Iteration {iteration} failed (consecutive failures: {consecutive_failures})")
                
                # If we have too many consecutive failures, provide diagnosis
                if consecutive_failures >= 3:
                    print(f"\n{consecutive_failures} consecutive failures detected!")
                    diagnose_optimization_failure(gp_model, bounds)
                    
                    user_input = input("\nContinue with optimization? (y/n): ").lower().strip()
                    if user_input != 'y':
                        print("Stopping optimization as requested by user")
                        break
                    consecutive_failures = 0  # Reset counter if user chooses to continue
            
        except KeyboardInterrupt:
            print(f"\nUser interrupted at iteration {iteration}")
            break
        except Exception as e:
            print(f"\nError in iteration {iteration}: {str(e)}")
            consecutive_failures += 1
            
            if consecutive_failures >= 5:
                print(f"Too many consecutive errors ({consecutive_failures}), stopping optimization")
                break
            else:
                print(f"Continuing to next iteration...")
                continue
    
    # Final statistics and summary
    total_elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETED")
    print(f"{'='*80}")
    print(f"  Total iterations attempted: {iteration + 1}")
    print(f"  Successful iterations: {successful_iterations}")
    print(f"  Success rate: {successful_iterations/(iteration+1)*100:.1f}%")
    print(f"  Total time elapsed: {total_elapsed/3600:.2f} hours")
    print(f"  Average time per iteration: {total_elapsed/(iteration+1):.2f} seconds")
    
    # Display final best results
    best_params, best_value, best_idx = fn.get_current_best_result(paths['data_path'])
    if best_params is not None:
        print(f"\nFINAL BEST RESULT:")
        print(f"  Best objective value: {best_value:.6f}")
        print(f"  Found at dataset entry: {best_idx + 1}")
        print(f"  Best parameters:")
        
        param_names = [
            "top_width", "bottom_width", "height", "length", "tooth_depth",
            "tooth_length", "tooth_width", "groove_length", "groove_width", "extension_factor"
        ]
        
        for name, value in zip(param_names, best_params):
            print(f"    {name:15}: {value:8.3f}")
    else:
        print(f"\nNo valid best result found")
    
    print(f"\nFILES LOCATION:")
    print(f"  Dataset: {paths['data_path']}")
    print(f"  Archive: {paths['data_dir']}")
    print(f"  Unified log: {os.path.join(paths['base_path'], SearchTimeConfig.LOG_FILE)}")
    print(f"  Legacy logs: {paths['model_time_record']}, {paths['time_record']}")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Set PyTorch to double precision (recommended for GP models)
    torch.set_default_dtype(torch.float64)
    
    # Execute main function
    main()