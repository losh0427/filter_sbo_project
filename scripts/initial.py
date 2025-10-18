import os
import functions

def generate_input_files(n_samples=300, data_dir="../Data1", config_file=None):
    """Generate specified number of input{i}.txt files for HFSS simulation
    
    Args:
        n_samples: Number of input files to generate
        data_dir: Directory to save input files
        config_file: Parameter configuration JSON file path (optional)
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    print("=" * 60)
    print("GENERATING INPUT FILES")
    print("=" * 60)
    print(f"Number of samples: {n_samples}")
    print(f"Data directory: {data_dir}")
    if config_file:
        print(f"Config file: {config_file}")
    print()
    
    try:
        # Generate random parameter sets
        print("Generating random parameter sets...")
        parameter_sets = functions.generate_random_parameters(
            n_samples=n_samples,
            config_file=config_file
        )
        print(f"Generated {len(parameter_sets)} parameter sets")
        
        # Ensure data directory exists
        functions.ensure_directory_exists(data_dir)
        
        # Save input files
        print("Saving input files...")
        for i, param_vector in enumerate(parameter_sets):
            input_file_path = os.path.join(data_dir, f"input{i}.txt")
            with open(input_file_path, 'w') as f:
                # Write as space-separated values
                param_str = ' '.join([str(param) for param in param_vector])
                f.write(param_str)
        
        print(f"Successfully generated {len(parameter_sets)} input files in {data_dir}")
        print()
        print("=" * 40)
        print("NEXT STEPS:")
        print("1. Run HFSS simulation script: hfss_script.py")
        print("2. Wait for all simulations to complete")
        print("3. Run integration to create dataset:")
        print("   integrate_to_dataset()")
        print("=" * 40)
        
        return True
        
    except Exception as e:
        print(f"Error generating input files: {str(e)}")
        return False

def integrate_to_dataset(data_dir="../Data1", dataset_file_path=None):
    """Integrate input{i}.txt and output{i}.fld files into dataset.txt
    
    Args:
        data_dir: Directory containing input and output files
        dataset_file_path: Path for output dataset file (default: "../Data/dataset.txt")
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    print("=" * 60)
    print("INTEGRATING TO DATASET")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    
    if dataset_file_path is None:
        dataset_file_path = "../Data/dataset.txt"
    print(f"Dataset file: {dataset_file_path}")
    print()
    
    try:
        # Check if data directory exists
        if not os.path.exists(data_dir):
            print(f"Error: Data directory does not exist: {data_dir}")
            return False
        
        # Count available files
        input_count = 0
        output_count = 0
        
        for filename in os.listdir(data_dir):
            if filename.startswith("input") and filename.endswith(".txt"):
                input_count += 1
            elif filename.startswith("output") and filename.endswith(".fld"):
                output_count += 1
        
        print(f"Found {input_count} input files and {output_count} output files")
        
        if input_count == 0:
            print("Error: No input files found")
            return False
        
        if output_count == 0:
            print("Error: No output files found")
            return False
        
        # Integrate data using functions.py
        print("Integrating data...")
        processed_count = functions.integrate_input_output_to_dataset(data_dir, dataset_file_path)
        
        if processed_count > 0:
            print(f"Successfully created dataset with {processed_count} entries")
            print(f"Dataset saved to: {dataset_file_path}")
            print()
            print("=" * 40)
            print("INTEGRATION COMPLETE")
            print(f"Total entries: {processed_count}")
            print(f"Dataset file: {dataset_file_path}")
            print("=" * 40)
            return True
        else:
            print("Error: No valid data entries were processed")
            return False
        
    except Exception as e:
        print(f"Error during integration: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified initial sampling for cavity filter optimization")
    parser.add_argument("--n_samples", type=int, default=300, help="Number of input files to generate")
    parser.add_argument("--data_dir", default="../Data1", help="Data directory path")
    parser.add_argument("--dataset_file", default="../Data/dataset.txt", help="Dataset file path")
    parser.add_argument("--config_file", default=None, help="Parameter config JSON file")
    
    # Mode selection
    parser.add_argument("--generate", action="store_true", help="Generate input files")
    parser.add_argument("--integrate", action="store_true", help="Integrate to dataset")
    
    args = parser.parse_args()
    
    try:
        if args.integrate:
            # Integration mode
            success = integrate_to_dataset(args.data_dir, args.dataset_file)
            
        elif args.generate:
            # Generation mode
            success = generate_input_files(args.n_samples, args.data_dir, args.config_file)
            
        else:
            # Default: show help
            print("Please specify a mode:")
            print("  --generate    : Generate input files")
            print("  --integrate   : Integrate to dataset")
            print()
            print("Examples:")
            print("  python initial.py --generate --n_samples 500")
            print("  python initial.py --integrate --data_dir ../Data1")
            success = False
        
        if success:
            print("Operation completed successfully!")
        else:
            print("Operation failed!")
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


"""
usage:
# Generate 500 input files
python initial.py --generate --n_samples 500

# Integrate existing input/output files into dataset
python initial.py --integrate --data_dir ../Data1

# Specify dataset output location
python initial.py --integrate --dataset_file ../MyDataset/dataset.txt
"""
