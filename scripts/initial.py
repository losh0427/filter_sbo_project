# -*- coding: ascii -*-
# Initial sampling pipeline for cavity filter optimization

import os
import time
import functions
import validation

def run_initial_sampling_pipeline(n_samples=300, 
                                 data_dir="../Data1",
                                 config_file=None):
    """Execute complete initial sampling pipeline"""
    
    print("=" * 60)
    print("INITIAL SAMPLING PIPELINE")
    print("=" * 60)
    print(f"Number of samples: {n_samples}")
    print(f"Data directory: {data_dir}")
    if config_file:
        print(f"Config file: {config_file}")
    print()
    
    try:
        # Stage 1: Generate random parameters
        print("Stage 1: Generating random parameter sets...")
        parameter_sets = functions.generate_random_parameters(
            n_samples=n_samples,
            config_file=config_file
        )
        print(f"Generated {len(parameter_sets)} parameter sets")
        
        # Stage 2: Save input files
        print("\nStage 2: Saving input files...")
        functions.save_input_files(parameter_sets, data_dir)
        
        # Stage 3: Validate generated inputs
        print("\nStage 3: Validating generated inputs...")
        validate_generated_inputs(data_dir, n_samples, config_file)
        
        # Stage 4: Instructions for HFSS simulation
        print("\nStage 4: Ready for HFSS simulation")
        print("=" * 40)
        print("NEXT STEPS:")
        print("1. Execute hfss_script.py in Ansys Electronics Desktop")
        print("2. Or run: python hfss_script.py")
        print("3. Wait for all simulations to complete")
        print("4. Run integration when simulations are done:")
        print(f"   python initial.py --integrate --data_dir {data_dir}")
        print("5. Dataset will be created in: ../Data/dataset.txt")
        print("=" * 40)
        
        return True
        
    except Exception as e:
        print(f"Error in initial sampling pipeline: {str(e)}")
        return False

def validate_generated_inputs(data_dir="../Data1", expected_count=300, config_file=None):
    """Validate generated input files"""
    input_files = functions.collect_all_input_files(data_dir)
    
    print(f"Validating {len(input_files)} input files...")
    
    if len(input_files) != expected_count:
        print(f"Warning: Expected {expected_count} files, found {len(input_files)}")
    
    valid_count = 0
    invalid_files = []
    
    for input_file in input_files:
        try:
            with open(input_file, 'r') as f:
                param_line = f.read().strip()
                param_vector = [float(x) for x in param_line.split()]
            
            # Validate parameter bounds
            bounds_valid, bounds_msg = validation.validate_parameter_bounds(param_vector, config_file)
            if not bounds_valid:
                invalid_files.append((os.path.basename(input_file), bounds_msg))
                continue
            
            # Validate geometric constraints  
            geom_valid, geom_msg = validation.validate_geometric_constraints(param_vector)
            if not geom_valid:
                invalid_files.append((os.path.basename(input_file), geom_msg))
                continue
            
            valid_count += 1
            
        except Exception as e:
            invalid_files.append((os.path.basename(input_file), f"Parse error: {str(e)}"))
    
    print(f"Validation results: {valid_count}/{len(input_files)} valid input files")
    
    if invalid_files:
        print("Invalid files:")
        for filename, error in invalid_files[:10]:  # Show first 10 errors
            print(f"  {filename}: {error}")
        if len(invalid_files) > 10:
            print(f"  ... and {len(invalid_files)-10} more errors")
    
    return valid_count == len(input_files)

def wait_for_simulation_completion(data_dir="../Data1", expected_count=300, timeout=72000):
    """Wait for all simulations to complete with progress monitoring"""
    print(f"\nWaiting for {expected_count} simulations to complete...")
    print("(This may take several hours)")
    print()
    
    start_time = time.time()
    last_count = 0
    
    while True:
        fld_files = functions.collect_all_fld_files(data_dir)
        current_count = len(fld_files)
        
        # Show progress
        elapsed = time.time() - start_time
        if current_count != last_count:
            progress_pct = (current_count / expected_count) * 100
            print(f"Progress: {current_count}/{expected_count} ({progress_pct:.1f}%) - "
                  f"Elapsed: {elapsed/3600:.1f}h")
            last_count = current_count
        
        if current_count >= expected_count:
            print("All simulations completed!")
            break
            
        if elapsed > timeout:
            print(f"Timeout reached. Only {current_count}/{expected_count} completed.")
            break
            
        time.sleep(30)  # Check every 30 seconds
    
    return current_count

def integrate_results(data_dir="../Data1", dataset_file_path=None):
    """Integrate simulation results into dataset"""
    print("\n" + "=" * 60)
    print("INTEGRATING SIMULATION RESULTS")
    print("=" * 60)
    
    if dataset_file_path is None:
        dataset_file_path = "../Data/dataset.txt"  # Default to Data folder
    
    try:
        # Check completeness
        print("Checking simulation completeness...")
        completeness = validation.check_simulation_completeness(data_dir)
        
        if completeness["completeness_ratio"] < 0.8:
            print(f"Warning: Only {completeness['completeness_ratio']:.1%} simulations completed")
            response = input("Continue with integration? (y/n): ")
            if response.lower() != 'y':
                print("Integration cancelled")
                return False
        
        # Integrate data
        print("\nIntegrating input and output data...")
        processed_count = functions.integrate_input_output_to_dataset(data_dir, dataset_file_path)
        
        # Validate final dataset
        print("\nValidating final dataset...")
        dataset_valid, dataset_msg = validation.validate_dataset_file(dataset_file_path)
        print(f"Dataset validation: {dataset_msg}")
        
        print("\n" + "=" * 60)
        print("INTEGRATION COMPLETE")
        print("=" * 60)
        print(f"Dataset created: {dataset_file_path}")
        print(f"Total entries: {processed_count}")
        print(f"Completeness: {completeness['completeness_ratio']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"Error during integration: {str(e)}")
        return False

def run_complete_pipeline(n_samples=300, 
                         data_root="./Data1",
                         wait_for_completion=True,
                         timeout=72000,
                         config_file=None):
    """Run complete pipeline from parameter generation to dataset creation"""
    
    # Setup paths
    input_dir = os.path.join(data_root, "inputs")
    output_dir = os.path.join(data_root, "outputs")
    dataset_file = os.path.join(data_root, "dataset.txt")
    
    # Stage 1: Generate inputs
    success = run_initial_sampling_pipeline(
        n_samples=n_samples,
        data_root=data_root,
        input_dir=input_dir,
        output_dir=output_dir,
        dataset_file=dataset_file,
        config_file=config_file
    )
    
    if not success:
        return False
    
    # Stage 2: Wait for simulations (optional)
    if wait_for_completion:
        print("\nAutomatically waiting for simulation completion...")
        completed_count = wait_for_simulation_completion(output_dir, n_samples, timeout)
        
        if completed_count < n_samples * 0.8:  # Less than 80% completed
            print(f"Warning: Only {completed_count}/{n_samples} simulations completed")
            return False
        
        # Stage 3: Integrate results
        return integrate_results(input_dir, output_dir, dataset_file)
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initial sampling pipeline for cavity filter optimization")
    parser.add_argument("--n_samples", type=int, default=300, help="Number of samples to generate")
    parser.add_argument("--data_dir", default="../Data1", help="Data directory path")
    parser.add_argument("--dataset_file", default="../Data/dataset.txt", help="Dataset file path")
    parser.add_argument("--config_file", default="parameters_config.json", help="Parameter config JSON file")
    
    # Modes
    parser.add_argument("--generate_only", action="store_true", help="Only generate input files")
    parser.add_argument("--integrate", action="store_true", help="Only integrate results")
    parser.add_argument("--complete", action="store_true", help="Run complete pipeline with waiting")
    parser.add_argument("--validate", action="store_true", help="Only run validation")
    
    # Options
    parser.add_argument("--timeout", type=int, default=72000, help="Timeout for waiting (seconds)")
    
    args = parser.parse_args()
    
    try:
        if args.integrate:
            # Integration mode
            integrate_results(args.data_dir, args.dataset_file)
            
        elif args.validate:
            # Validation mode
            validation.run_full_validation(args.data_dir, args.dataset_file)
            
        elif args.complete:
            # Complete pipeline with waiting
            run_complete_pipeline(
                n_samples=args.n_samples,
                data_dir=args.data_dir,
                wait_for_completion=True,
                timeout=args.timeout,
                config_file=args.config_file,
                dataset_file=args.dataset_file
            )
            
        else:
            # Default: generate inputs only
            run_initial_sampling_pipeline(
                n_samples=args.n_samples,
                data_dir=args.data_dir,
                config_file=args.config_file
            )
            
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise