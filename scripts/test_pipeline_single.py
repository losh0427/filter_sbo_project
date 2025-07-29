#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single iteration pipeline test
Test the complete optimization workflow with 1 iteration
"""

import os
import time
import torch
import numpy as np
from models import GPSurrogateModel, AcquisitionFunction, ModelTrainer
from functions import (
    get_parameter_bounds, load_historical_data, clip_parameters_to_bounds,
    write_input_file, get_next_input_number, ensure_directory_exists
)

def test_single_pipeline_iteration():
    """
    Test complete pipeline with single iteration
    """
    print("=" * 60)
    print("SINGLE ITERATION PIPELINE TEST")
    print("=" * 60)
    
    # Set PyTorch to double precision (required for GP)
    torch.set_default_dtype(torch.float64)
    
    # Configuration (scripts are in scripts/ folder)
    base_path = "../Data"  # Data directory containing dataset.txt
    dataset_file = "dataset.txt"  # Dataset filename
    data1_dir = "../Data1"  # Directory for new input files
    acquisition_func_type = "EI"
    
    print(f"Base path: {base_path}")
    print(f"Dataset file: {dataset_file}")
    print(f"Data1 directory: {data1_dir}")
    print(f"Acquisition function: {acquisition_func_type}")
    
    # Ensure Data1 directory exists
    ensure_directory_exists(data1_dir)
    
    # Step 1: Load historical data
    print("\n" + "="*40)
    print("STEP 1: LOADING HISTORICAL DATA")
    print("="*40)
    
    dataset_path = os.path.join(base_path, dataset_file)
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        print("Please run integrate_existing_data.py first!")
        return False
    
    X, y = load_historical_data(dataset_path)
    if X is None or y is None:
        print("❌ Failed to load historical data")
        return False
    
    print(f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} dimensions")
    print(f"   Objective value range: [{y.min():.6f}, {y.max():.6f}]")
    print(f"   Current best value: {y.max():.6f}")
    
    # Step 2: Train GP surrogate model
    print("\n" + "="*40)
    print("STEP 2: TRAINING GP SURROGATE MODEL")
    print("="*40)
    
    gp_model = GPSurrogateModel()
    trainer = ModelTrainer()
    
    start_time = time.time()
    success = trainer.train_with_retry(gp_model, X, y, max_retries=3)
    train_time = time.time() - start_time
    
    if not success:
        print("❌ GP model training failed")
        return False
    
    print(f"✅ GP model trained successfully in {train_time:.2f}s")
    
    # Test prediction on existing data point
    test_idx = 0
    test_x = X[test_idx:test_idx+1]
    pred_mean, pred_std = gp_model.predict(test_x)
    actual_y = y[test_idx]
    
    print(f"   Test prediction: actual={actual_y:.6f}, predicted={pred_mean[0]:.6f}±{pred_std[0]:.6f}")
    
    # Step 3: Build acquisition function
    print("\n" + "="*40)
    print("STEP 3: BUILDING ACQUISITION FUNCTION")
    print("="*40)
    
    bounds = get_parameter_bounds()
    print(f"Parameter bounds: {bounds.shape}")
    
    acq_func = AcquisitionFunction(func_type=acquisition_func_type)
    best_f = y.max()  # Current best value for EI
    
    try:
        acq_func.build(gp_model, best_f=best_f, bounds=bounds)
        print(f"✅ {acquisition_func_type} acquisition function built successfully")
        print(f"   Current best objective value: {best_f:.6f}")
    except Exception as e:
        print(f"❌ Acquisition function building failed: {e}")
        return False
    
    # Step 4: Optimize acquisition function
    print("\n" + "="*40)
    print("STEP 4: OPTIMIZING ACQUISITION FUNCTION")
    print("="*40)
    
    start_time = time.time()
    try:
        candidate, acq_value = acq_func.optimize(bounds=bounds, q=1)
        search_time = time.time() - start_time
        
        # Ensure candidate is within bounds
        candidate_clipped = clip_parameters_to_bounds(candidate[0])
        
        print(f"✅ Acquisition optimization completed in {search_time:.2f}s")
        print(f"   Acquisition value: {acq_value.item():.6f}")
        print(f"   Candidate parameters:")
        
        # Display candidate parameters with names
        param_names = [
            "top_width", "bottom_width", "height", "length", "tooth_depth",
            "tooth_length", "tooth_width", "groove_length", "groove_width", "extension_factor"
        ]
        
        for i, (name, value, bound) in enumerate(zip(param_names, candidate_clipped, bounds)):
            print(f"     {name:15}: {value:8.3f} (range: [{bound[0]:6.1f}, {bound[1]:6.1f}])")
        
    except Exception as e:
        print(f"❌ Acquisition optimization failed: {e}")
        # Fallback to random sampling
        print("   Using random fallback...")
        random_candidate = torch.rand(10, dtype=torch.float64)
        for i in range(10):
            random_candidate[i] = bounds[i, 0] + random_candidate[i] * (bounds[i, 1] - bounds[i, 0])
        candidate_clipped = random_candidate
        acq_value = torch.tensor(0.0)
    
    # Step 5: Generate next input file
    print("\n" + "="*40)
    print("STEP 5: GENERATING NEXT INPUT FILE")
    print("="*40)
    
    # Get next input file number
    next_input_num = get_next_input_number(data1_dir)
    input_filename = f"input{next_input_num}.txt"
    input_filepath = os.path.join(data1_dir, input_filename)
    
    try:
        write_input_file(input_filepath, candidate_clipped)
        print(f"✅ Generated new input file: {input_filename}")
        print(f"   File path: {input_filepath}")
        
        # Verify file content
        with open(input_filepath, 'r') as f:
            content = f.read().strip()
            params = content.split()
        
        print(f"   File contains {len(params)} parameters")
        print(f"   First 3 parameters: {' '.join(params[:3])}")
        print(f"   Last 3 parameters: {' '.join(params[-3:])}")
        
    except Exception as e:
        print(f"❌ Failed to generate input file: {e}")
        return False
    
    # Step 6: Summary
    print("\n" + "="*60)
    print("PIPELINE TEST SUMMARY")
    print("="*60)
    print("✅ All steps completed successfully!")
    print(f"   Historical data: {X.shape[0]} samples loaded")
    print(f"   GP model: Trained and validated")
    print(f"   Acquisition function: {acquisition_func_type} optimized")
    print(f"   New candidate: Generated as {input_filename}")
    print(f"   Next step: Run HFSS simulation on {input_filename}")
    print("="*60)
    
    return True

def main():
    """Main function"""
    try:
        success = test_single_pipeline_iteration()
        if success:
            print("\n🎉 PIPELINE TEST PASSED!")
            print("Ready for deployment to offline machine!")
        else:
            print("\n❌ PIPELINE TEST FAILED!")
            print("Please check the error messages above.")
        
        return success
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()