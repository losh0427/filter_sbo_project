#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrate existing input/output files into dataset
Modified version with automatic detection and immediate processing
"""

import os
import sys
import shutil
from functions import (
    integrate_input_output_to_dataset, 
    ensure_directory_exists, 
    update_iteration_counter,
    read_iteration_counter
)

def detect_available_file_pairs(data1_dir):
    """
    Auto-detect available input/output file pairs in Data1 directory
    
    Returns:
        dict: {
            'paired_files': [(num, input_file, output_file), ...],
            'orphaned_inputs': [input_files without outputs],
            'orphaned_outputs': [output_files without inputs],
            'total_pairs': int
        }
    """
    if not os.path.exists(data1_dir):
        return {
            'paired_files': [],
            'orphaned_inputs': [],
            'orphaned_outputs': [],
            'total_pairs': 0
        }
    
    # Find all input and output files
    input_files = {}  # {number: filename}
    output_files = {}  # {number: filename}
    
    for filename in os.listdir(data1_dir):
        if filename.startswith("input") and filename.endswith(".txt"):
            try:
                num_str = filename[5:-4]  # Remove "input" and ".txt"
                file_num = int(num_str)
                input_files[file_num] = filename
            except ValueError:
                continue
        elif filename.startswith("output") and filename.endswith(".fld"):
            try:
                num_str = filename[6:-4]  # Remove "output" and ".fld"
                file_num = int(num_str)
                output_files[file_num] = filename
            except ValueError:
                continue
    
    # Find paired files
    paired_files = []
    all_input_nums = set(input_files.keys())
    all_output_nums = set(output_files.keys())
    
    for num in sorted(all_input_nums & all_output_nums):
        paired_files.append((num, input_files[num], output_files[num]))
    
    # Find orphaned files
    orphaned_input_nums = all_input_nums - all_output_nums
    orphaned_output_nums = all_output_nums - all_input_nums
    
    orphaned_inputs = [input_files[num] for num in sorted(orphaned_input_nums)]
    orphaned_outputs = [output_files[num] for num in sorted(orphaned_output_nums)]
    
    return {
        'paired_files': paired_files,
        'orphaned_inputs': orphaned_inputs,
        'orphaned_outputs': orphaned_outputs,
        'total_pairs': len(paired_files)
    }

def integrate_existing_data():
    """
    Auto-detect and integrate existing input/output files into dataset
    Move all processed files from Data1 to Data archive
    Initialize shared counter for seamless pipeline continuation
    """
    print("=" * 60)
    print("AUTO-DETECTING AND INTEGRATING EXISTING DATA")
    print("=" * 60)
    
    # Set paths (scripts are in scripts/ folder)
    data1_dir = "../Data1"      # Working directory for HFSS
    data_dir = "../Data"        # Archive directory
    dataset_file = "../Data/dataset.txt"  # Output dataset file
    counter_file = "../current_input_counter.txt"  # Shared counter file
    
    print(f"Data1 directory (working): {data1_dir}")
    print(f"Data archive directory: {data_dir}")
    print(f"Dataset file: {dataset_file}")
    print(f"Counter file: {counter_file}")
    
    # Ensure directories exist
    ensure_directory_exists(data_dir)
    
    # Step 1: Auto-detect available files
    print(f"\n📋 STEP 1: AUTO-DETECTING FILES IN {data1_dir}")
    print("-" * 40)
    
    file_detection = detect_available_file_pairs(data1_dir)
    
    print(f"Found file pairs: {file_detection['total_pairs']}")
    print(f"Orphaned inputs: {len(file_detection['orphaned_inputs'])}")
    print(f"Orphaned outputs: {len(file_detection['orphaned_outputs'])}")
    
    if file_detection['total_pairs'] == 0:
        print("❌ No complete input/output pairs found!")
        if file_detection['orphaned_inputs']:
            print(f"   Found {len(file_detection['orphaned_inputs'])} input files without outputs:")
            for filename in file_detection['orphaned_inputs'][:5]:  # Show first 5
                print(f"     - {filename}")
        if file_detection['orphaned_outputs']:
            print(f"   Found {len(file_detection['orphaned_outputs'])} output files without inputs:")
            for filename in file_detection['orphaned_outputs'][:5]:  # Show first 5
                print(f"     - {filename}")
        print("\n💡 Make sure HFSS simulation has completed for all input files!")
        return False
    
    # Show detailed file pair information
    print(f"\n📝 Detected {file_detection['total_pairs']} complete pairs:")
    for i, (num, input_file, output_file) in enumerate(file_detection['paired_files']):
        if i < 5:  # Show first 5 pairs
            print(f"   {num:3d}: {input_file} ↔ {output_file}")
        elif i == 5:
            print(f"   ... and {len(file_detection['paired_files']) - 5} more pairs")
            break
    
    if file_detection['orphaned_inputs']:
        print(f"\n⚠️  Warning: {len(file_detection['orphaned_inputs'])} orphaned input files will be skipped")
    if file_detection['orphaned_outputs']:
        print(f"⚠️  Warning: {len(file_detection['orphaned_outputs'])} orphaned output files will be skipped")
    
    # Step 2: Integrate data using existing function
    print(f"\n🔄 STEP 2: INTEGRATING DATA TO DATASET")
    print("-" * 40)
    
    processed_count = integrate_input_output_to_dataset(data1_dir, dataset_file)
    
    if processed_count == 0:
        print("❌ FAILED: No valid data entries were processed")
        return False
    
    print(f"✅ SUCCESS: Created dataset with {processed_count} entries")
    print(f"Dataset saved to: {dataset_file}")
    
    # Step 3: Initialize shared counter for next input
    print(f"\n🔢 STEP 3: INITIALIZING SHARED COUNTER")
    print("-" * 40)
    
    # Set counter to the next available number after processed files
    max_processed_num = max([num for num, _, _ in file_detection['paired_files']], default=-1)
    next_counter = max_processed_num + 1
    
    update_iteration_counter(counter_file, next_counter)
    print(f"✅ Initialized shared counter to: {next_counter}")
    print(f"Next input file will be: input{next_counter}.txt")
    
    # Step 4: Move all processed files to archive
    print(f"\n📦 STEP 4: MOVING FILES TO ARCHIVE")
    print("-" * 40)
    
    moved_count = 0
    for num, input_file, output_file in file_detection['paired_files']:
        input_source = os.path.join(data1_dir, input_file)
        output_source = os.path.join(data1_dir, output_file)
        
        input_dest = os.path.join(data_dir, input_file)
        output_dest = os.path.join(data_dir, output_file)
        
        # Move input file
        try:
            if os.path.exists(input_source):
                if os.path.exists(input_dest):
                    os.remove(input_dest)  # Remove existing file in archive
                shutil.move(input_source, input_dest)
                moved_count += 1
                if moved_count <= 5:  # Show first 5 moves
                    print(f"   Moved {input_file} to archive")
        except Exception as e:
            print(f"   ⚠️  Failed to move {input_file}: {e}")
        
        # Move output file
        try:
            if os.path.exists(output_source):
                if os.path.exists(output_dest):
                    os.remove(output_dest)  # Remove existing file in archive
                shutil.move(output_source, output_dest)
                moved_count += 1
                if moved_count <= 10:  # Show first 5 pairs
                    print(f"   Moved {output_file} to archive")
        except Exception as e:
            print(f"   ⚠️  Failed to move {output_file}: {e}")
    
    print(f"📦 Moved {moved_count} files to archive")
    
    # Step 5: Clean up any remaining orphaned files (optional)
    print(f"\n🧹 STEP 5: CLEANING UP ORPHANED FILES")
    print("-" * 40)
    
    cleaned_count = 0
    
    # Move orphaned files to archive as well (for record keeping)
    for filename in file_detection['orphaned_inputs'] + file_detection['orphaned_outputs']:
        source_path = os.path.join(data1_dir, filename)
        dest_path = os.path.join(data_dir, filename)
        
        try:
            if os.path.exists(source_path):
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                shutil.move(source_path, dest_path)
                cleaned_count += 1
                print(f"   Moved orphaned file: {filename}")
        except Exception as e:
            print(f"   ⚠️  Failed to move orphaned file {filename}: {e}")
    
    if cleaned_count > 0:
        print(f"🧹 Moved {cleaned_count} orphaned files to archive")
    else:
        print("✨ No orphaned files to clean up")
    
    # Step 6: Verify Data1 is clean
    print(f"\n✅ STEP 6: VERIFICATION")
    print("-" * 40)
    
    remaining_files = []
    if os.path.exists(data1_dir):
        remaining_files = [f for f in os.listdir(data1_dir) 
                          if f.startswith(('input', 'output')) and 
                          (f.endswith('.txt') or f.endswith('.fld'))]
    
    if remaining_files:
        print(f"⚠️  Warning: {len(remaining_files)} files still in Data1:")
        for filename in remaining_files[:5]:
            print(f"     - {filename}")
    else:
        print("✅ Data1 directory is clean and ready for pipeline")
    
    # Show dataset summary
    print(f"\nDataset Summary:")
    print(f"  📁 Dataset file: {dataset_file}")
    print(f"  📊 Total entries: {processed_count}")
    print(f"  🔢 Next input counter: {next_counter}")
    print(f"  📂 Archive location: {data_dir}")
    
    # Show first few entries of dataset
    print(f"\nFirst 3 dataset entries:")
    try:
        with open(dataset_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 3:
                    parts = line.strip().split()
                    if len(parts) >= 11:
                        params = ' '.join(parts[:10])
                        objective = parts[10]
                        print(f"  Entry {i+1}: params=[{params[:50]}...] objective={objective}")
                else:
                    break
    except Exception as e:
        print(f"  Unable to preview dataset: {e}")
    
    return True

if __name__ == "__main__":
    try:
        success = integrate_existing_data()
        if success:
            print("\n" + "="*60)
            print("🎉 INTEGRATION COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("✅ Dataset created with all existing data")
            print("✅ Shared counter initialized for pipeline")
            print("✅ Data1 cleaned and ready for optimization")
            print("")
            print("🚀 READY FOR PIPELINE EXECUTION!")
            print("Next step: Run pipeline.py to start optimization iterations")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ INTEGRATION FAILED!")
            print("="*60)
            print("Please check the error messages above and ensure:")
            print("1. HFSS simulations have completed for all input files")
            print("2. Data1 directory contains matching input/output pairs")
            print("3. File permissions are correct")
            print("="*60)
    except KeyboardInterrupt:
        print("\n⚠️  Integration interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()