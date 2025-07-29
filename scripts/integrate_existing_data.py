#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrate existing input/output files into dataset
"""

import os
import sys
from functions import integrate_input_output_to_dataset, ensure_directory_exists

def integrate_existing_data():
    """
    Integrate input0.txt to input29.txt and corresponding output files
    """
    print("=" * 60)
    print("INTEGRATING EXISTING DATA TO DATASET")
    print("=" * 60)
    
    # Set paths (scripts are in scripts/ folder)
    data1_dir = "../Data1"  # Your existing data directory
    data_dir = "../Data"    # Archive directory
    dataset_file = "../Data/dataset.txt"  # Output dataset file
    
    print(f"Data1 directory: {data1_dir}")
    print(f"Data archive directory: {data_dir}")
    print(f"Dataset file: {dataset_file}")
    
    # Ensure directories exist
    ensure_directory_exists(data_dir)
    
    # Check if Data1 exists
    if not os.path.exists(data1_dir):
        print(f"Error: {data1_dir} directory does not exist!")
        return False
    
    # Count existing files
    input_count = 0
    output_count = 0
    for filename in os.listdir(data1_dir):
        if filename.startswith("input") and filename.endswith(".txt"):
            input_count += 1
        elif filename.startswith("output") and filename.endswith(".fld"):
            output_count += 1
    
    print(f"Found {input_count} input files and {output_count} output files")
    
    if input_count == 0:
        print("Error: No input files found!")
        return False
    
    if output_count == 0:
        print("Error: No output files found!")
        return False
    
    # Integrate data
    print("\nIntegrating data...")
    processed_count = integrate_input_output_to_dataset(data1_dir, dataset_file)
    
    if processed_count > 0:
        print(f"\n✅ SUCCESS: Created dataset with {processed_count} entries")
        print(f"Dataset saved to: {dataset_file}")
        
        # Show first few lines of dataset
        print("\nFirst 3 entries of dataset:")
        with open(dataset_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 3:
                    parts = line.strip().split()
                    params = ' '.join(parts[:10])
                    objective = parts[10]
                    print(f"  Entry {i+1}: params=[{params[:50]}...] objective={objective}")
                else:
                    break
        
        return True
    else:
        print("❌ FAILED: No valid data entries were processed")
        return False

if __name__ == "__main__":
    success = integrate_existing_data()
    if success:
        print("\n" + "="*60)
        print("READY FOR PIPELINE TEST!")
        print("Next step: Run pipeline test with 1 iteration")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("INTEGRATION FAILED!")
        print("Please check your data files")
        print("="*60)