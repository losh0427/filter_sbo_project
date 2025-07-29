# 將這些函數添加到你的 functions.py 文件末尾

# ============================================================================
# Shared Counter Management Module  
# ============================================================================

def read_iteration_counter(counter_file_path):
    """
    Read current iteration counter from file
    Returns the next input file number to be generated
    """
    try:
        if not os.path.exists(counter_file_path):
            # Initialize counter file if it doesn't exist
            with open(counter_file_path, 'w') as f:
                f.write("0")
            return 0
            
        with open(counter_file_path, 'r') as f:
            counter_str = f.read().strip()
            if not counter_str:
                return 0
            return int(counter_str)
            
    except (ValueError, IOError) as e:
        print(f"Warning: Failed to read counter file {counter_file_path}: {str(e)}")
        print("Using default counter value: 0")
        return 0

def update_iteration_counter(counter_file_path, next_number):
    """
    Update iteration counter to specified number
    Uses atomic write (temp file + rename) for safety
    """
    try:
        # Ensure directory exists
        counter_dir = os.path.dirname(counter_file_path)
        if counter_dir and not os.path.exists(counter_dir):
            os.makedirs(counter_dir)
        
        # Atomic write: write to temp file first
        temp_file_path = counter_file_path + ".tmp"
        with open(temp_file_path, 'w') as f:
            f.write(str(next_number))
        
        # Rename temp file to actual file (atomic on most systems)
        if os.path.exists(counter_file_path):
            os.remove(counter_file_path)
        os.rename(temp_file_path, counter_file_path)
        
        print(f"Updated iteration counter to: {next_number}")
        
    except (IOError, OSError) as e:
        print(f"Warning: Failed to update counter file {counter_file_path}: {str(e)}")
        # Clean up temp file if it exists
        temp_file_path = counter_file_path + ".tmp"
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

def get_next_input_number_from_counter(counter_file_path):
    """
    Get next input number from shared counter file
    This replaces the original get_next_input_number function for pipeline usage
    """
    return read_iteration_counter(counter_file_path)