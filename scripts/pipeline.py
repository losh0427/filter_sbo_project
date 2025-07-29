# 在pipeline.py中需要修改的部分

# 1. 修改imports，添加計數器函數
from functions import (
    execute_data_archiving, update_file_counters, get_parameter_bounds,
    validate_parameters, clip_parameters_to_bounds, write_input_file,
    wait_for_output_file, calculate_objective_value, load_historical_data,
    append_data_to_file, record_iteration_time, record_model_performance,
    get_current_best_result, ensure_directory_exists, print_iteration_summary,
    read_iteration_counter, update_iteration_counter, get_next_input_number_from_counter  # 新增
)

# 2. 修改initialize_paths_and_directories函數，添加計數器檔案路徑
def initialize_paths_and_directories(base_path, data_name):
    """Initialize paths and directories"""
    paths = {
        'base_path': base_path,
        'data_path': os.path.join(base_path, f'{data_name}.txt'),
        'data1_dir': os.path.join(base_path, 'Data1'),
        'data_dir': os.path.join(base_path, 'Data'),
        'file_record': os.path.join(base_path, 'move_file.txt'),
        'time_record': os.path.join(base_path, 'round_time.txt'),
        'model_time_record': os.path.join(base_path, 'model_time.txt'),
        'counter_file': os.path.join(base_path, 'current_input_counter.txt')  # 新增計數器檔案
    }
    
    # Ensure necessary directories exist
    for dir_path in [paths['data1_dir'], paths['data_dir']]:
        ensure_directory_exists(dir_path)
    
    return paths

# 3. 修改execute_hfss_evaluation函數，使用計數器並更新
def execute_hfss_evaluation(candidate_point, paths):
    """
    Execute HFSS evaluation
    Output input file and wait for output results
    Uses shared counter for file numbering
    """
    # Get current input number from counter
    input_number = get_next_input_number_from_counter(paths['counter_file'])
    
    # Write input file
    input_filepath = os.path.join(paths['data1_dir'], f'input{input_number}.txt')
    write_input_file(input_filepath, candidate_point)
    print(f"  Parameter file output: input{input_number}.txt")
    
    # Update counter for next iteration  
    update_iteration_counter(paths['counter_file'], input_number + 1)
    
    # Wait for corresponding output file
    output_filepath = os.path.join(paths['data1_dir'], f'output{input_number}.fld')
    wait_time = wait_for_output_file(output_filepath)
    
    # Calculate objective function value
    objective_value = calculate_objective_value(output_filepath)
    
    if objective_value is None:
        print(f"  Warning: Unable to calculate objective function value, using default value 0.0")
        objective_value = 0.0
    else:
        print(f"  Objective function value: {objective_value:.6f}")
    
    return objective_value, wait_time, input_number

# 4. 修改single_iteration_cycle函數，移除input_counter參數
def single_iteration_cycle(iteration, paths, gp_model, acq_func, bounds):
    """
    Execute single iteration cycle
    """
    print(f"\n=== Iteration {iteration} Start ===")
    iteration_start_time = time.time()
    
    # Step 1: Load historical data and train model
    print("Step 1: Loading data and training GP model...")
    model_start_time = time.time()
    
    X, y = load_historical_data(paths['data_path'])
    if X is None or y is None:
        print("  No historical data, skipping this iteration")
        return 0.0
    
    # Train GP model
    trainer = ModelTrainer()
    success = trainer.train_with_retry(gp_model, X, y)
    if not success:
        print("  GP model training failed, skipping this iteration")
        return 0.0
        
    model_train_time = time.time() - model_start_time
    print(f"  Model training completed, time elapsed: {model_train_time:.2f}s")
    
    # Step 2: Build acquisition function
    print("Step 2: Building acquisition function...")
    best_f = y.max()  # Current best value
    acq_func.build(gp_model, best_f=best_f, bounds=bounds)
    
    # Step 3: Generate candidate points
    print("Step 3: Generating candidate points...")
    search_time = get_search_time_allocation(iteration)
    candidate_point, acq_value, search_elapsed = generate_candidate_points(
        gp_model, acq_func, bounds, search_time, iteration
    )
    
    # Step 4: HFSS evaluation (now with counter management)
    print("Step 4: Executing HFSS evaluation...")
    objective_value, wait_time, input_number = execute_hfss_evaluation(
        candidate_point, paths
    )
    
    # Step 5: Update dataset
    print("Step 5: Updating dataset...")
    append_data_to_file(paths['data_path'], candidate_point, objective_value)
    
    # Step 6: Record statistics
    total_time = time.time() - iteration_start_time
    record_model_performance(
        paths['model_time_record'], iteration, total_time, 
        model_train_time, search_elapsed, X.shape[0] + 1, 1
    )
    record_iteration_time(paths['time_record'], wait_time)
    
    # Step 7: Display current best results
    best_params, best_value, best_idx = get_current_best_result(paths['data_path'])
    if best_params is not None:
        print_iteration_summary(iteration, best_value, objective_value, X.shape[0] + 1)
        print(f"Best result at index: {best_idx + 1}")
        print(f"Current input file: input{input_number}.txt")
    
    print(f"=== Iteration {iteration} Completed, total time: {total_time:.2f}s ===")
    return objective_value

# 5. 修改main函數中的迭代循環部分
def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_arguments()
    print(f"Starting global optimization, working directory: {args.path}")
    print(f"Data file: {args.data}, max iterations: {args.max_iterations}")
    print(f"Acquisition function: {args.acquisition_func}")
    
    # Initialize paths and directories
    paths = initialize_paths_and_directories(args.path, args.data)
    print(f"Path configuration completed:")
    for key, value in paths.items():
        print(f"  {key}: {value}")
    
    # Initialize counter if needed
    current_counter = read_iteration_counter(paths['counter_file'])
    print(f"Current input counter: {current_counter}")
    
    # Execute data archiving
    print("\nExecuting data archiving...")
    final_count, initial_count = execute_data_archiving(
        paths['data1_dir'], paths['data_dir'], paths['file_record']
    )
    
    # Initialize model components
    print("\nInitializing model components...")
    gp_model = GPSurrogateModel()
    acq_func = AcquisitionFunction(func_type=args.acquisition_func)
    bounds = get_parameter_bounds()
    
    print(f"Parameter bounds: {bounds.shape}")
    print(f"Acquisition function type: {args.acquisition_func}")
    
    # Main iteration loop (移除input_counter相關邏輯)
    print(f"\nStarting main iteration loop, {args.max_iterations} iterations total...")
    start_time = time.time()
    
    for iteration in range(args.max_iterations):
        try:
            current_objective = single_iteration_cycle(
                iteration, paths, gp_model, acq_func, bounds
            )
            
            # Update file counter (keep original logic for compatibility)
            final_count += 1
            update_file_counters(paths['file_record'], final_count, initial_count)
            
        except KeyboardInterrupt:
            print(f"\nUser interrupted, stopping at iteration {iteration}")
            break
        except Exception as e:
            print(f"\nError in iteration {iteration}: {str(e)}")
            print("Continuing to next iteration...")
            continue
    
    # Final statistics
    total_elapsed = time.time() - start_time
    print(f"\n=== Optimization Completed ===")
    print(f"Total iterations: {iteration + 1}")
    print(f"Total time elapsed: {total_elapsed/3600:.2f} hours")
    
    # Display final best results
    best_params, best_value, best_idx = get_current_best_result(paths['data_path'])
    if best_params is not None:
        print(f"\nFinal best result:")
        print(f"  Best value: {best_value:.6f}")
        print(f"  Best parameters: {best_params.tolist()}")
        print(f"  Located at index: {best_idx + 1}")
    else:
        print("No valid best result found")