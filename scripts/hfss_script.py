# -*- coding: ascii -*-
# HFSS 2020.2 automation - Enhanced Trapezoidal Cavity with Boundary Constraints

import os, datetime, time, ScriptEnv, math

# Initialize HFSS
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()

OUT_DIR = os.path.dirname(__file__)
FIELD_EXPORT_DIR = OUT_DIR
_timestamp = lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Global geometry data
GEOMETRY_DATA = {}

# =====================================================================
# ENHANCED FUNCTIONS FOR LOOP PROCESSING
# =====================================================================

def wait_for_input_file(iteration, data_dir="../Data1", timeout=3600):
    """Wait for input{iteration}.txt to appear"""
    input_file_path = os.path.join(data_dir, "input{}.txt".format(iteration))
    start_time = time.time()
    
    while not os.path.exists(input_file_path):
        if time.time() - start_time > timeout:
            raise TimeoutError("Timeout waiting for input file: {}".format(input_file_path))
        time.sleep(1)
    
    # Additional wait to ensure file is completely written
    time.sleep(0.5)
    return input_file_path

def parse_parameter_vector_from_file(input_file_path):
    """Parse 10-dimensional parameter vector from input file"""
    try:
        with open(input_file_path, 'r') as f:
            param_line = f.read().strip()
            param_vector = [float(x) for x in param_line.split()]
        
        if len(param_vector) != 10:
            raise ValueError("Expected 10 parameters, got {}".format(len(param_vector)))
        
        return param_vector
        
    except Exception as e:
        raise RuntimeError("Error parsing parameter file {}: {}".format(input_file_path, str(e)))

def convert_to_hfss_parameters(param_vector):
    """Convert parameter vector to HFSS parameter dictionaries"""
    # Parameters according to hfss_script_spec.md:
    # Index 0-3: trapezoidal parameters (top_width, bottom_width, height, length)
    # Index 4-9: groove parameters (tooth_depth, tooth_length, tooth_width, groove_length, groove_width, extension_factor)
    
    trapezoidal_params = {
        "top_width": param_vector[0],      # [100, 200] mm
        "bottom_width": param_vector[1],   # [200, 400] mm
        "height": param_vector[2],         # [50, 70] mm
        "length": param_vector[3],         # [300, 400] mm
        "wall_thickness": 2.0,             # 2.0 mm (fixed)
        "f0": 2.45,                        # 2.45 GHz (fixed)
    }
    
    groove_params = {
        "tooth_depth": param_vector[4],     # [0.1, 19.9] mm
        "tooth_length": param_vector[5],    # [0.05, 50.0] mm
        "tooth_width": param_vector[6],     # [0.05, 50.0] mm
        "groove_length": param_vector[7],   # [0.05, 100.0] mm
        "groove_width": param_vector[8],    # [0.05, 100.0] mm
        "extension_factor": param_vector[9], # [1.0, 2.0]
        "max_tooth_count": 600,             # 600 (fixed)
    }
    
    return trapezoidal_params, groove_params

def export_field_to_fld_file(des, iteration, data_dir="../Data1", freq_ghz=2.45):
    """Export rear plane field data to output{iteration}.fld file"""
    observe_data = GEOMETRY_DATA["observe_planes"]
    plane_width = observe_data["width"]
    plane_height = observe_data["height"]
    rear_x = observe_data["rear_x"]
    
    start_coords = [rear_x, 0.0, 0.0]
    end_coords = [rear_x, plane_width, plane_height]
    
    # Output file naming: output{iteration}.fld
    filename = "output{}.fld".format(iteration)
    filepath = os.path.join(data_dir, filename)
    
    # Ensure output directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    oModule = des.GetModule("FieldsReporter")
    
    try:
        oModule.CopyNamedExprToStack("Mag_E")
        oModule.ExportOnGrid(
            filepath,
            ["{}mm".format(start_coords[0]), "{}mm".format(start_coords[1]), "{}mm".format(start_coords[2])],
            ["{}mm".format(end_coords[0]), "{}mm".format(end_coords[1]), "{}mm".format(end_coords[2])],
            ["1.0mm", "1.0mm", "1.0mm"],
            "Setup1 : LastAdaptive",
            [
                "Freq:=", "{}GHz".format(freq_ghz),
                "Phase:=", "0deg"
            ],
            True,
            "Cartesian",
            ["0mm", "0mm", "0mm"],
            False
        )
        return {"success": True, "filepath": filepath, "filename": filename}
    except Exception as e:
        return {"success": False, "error": str(e), "filepath": filepath}

def main_simulation_loop(start_iter=0, end_iter=300, data_dir="../Data1"):
    """Main simulation loop - wait for input, simulate, output .fld"""
    
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    for iteration in range(start_iter, end_iter):
        try:
            # Step 1: Wait for input file
            input_file_path = wait_for_input_file(iteration, data_dir)
            
            # Step 2: Parse parameters
            param_vector = parse_parameter_vector_from_file(input_file_path)
            trapezoidal_params, groove_params = convert_to_hfss_parameters(param_vector)
            
            # Step 3: Run simulation
            result = run_single_simulation(trapezoidal_params, groove_params, iteration, data_dir)
            
            # Step 4: Cleanup project resources
            cleanup_current_project()
            
        except Exception as e:
            # Log error but continue with next iteration
            error_msg = "Error in iteration {}: {}".format(iteration, str(e))
            
            # Save error info to file
            error_file = os.path.join(data_dir, "error_{}.txt".format(iteration))
            try:
                with open(error_file, 'w') as f:
                    f.write(error_msg)
            except:
                pass
            
            continue

def cleanup_current_project():
    """Clean up current HFSS project to free resources"""
    try:
        oProject = oDesktop.GetActiveProject()
        if oProject is not None:
            project_name = oProject.GetName()
            oDesktop.CloseProject(project_name)
    except:
        pass

# =====================================================================
# GEOMETRY CALCULATION FUNCTIONS
# =====================================================================

def compute_trapezoids(a, b, h, d):
    """Compute trapezoidal coordinates"""
    delta = (b - a) / 2.0
    m = 2.0 * h / (b - a)
    L = (1 + m * m) ** 0.5
    c = d * L
    
    # Large trapezoid vertices
    xA = - (d + c) / m
    yA = -d
    xB = (b * m + d + c) / m
    yB = -d
    xD = (h + d - c) / m
    yD = h + d
    xC = (b * m - (h + d) + c) / m
    yC = h + d
    
    # Translation to origin
    tx = -xA
    ty = -yA
    A = (0.0, 0.0)
    B = (xB + tx, yB + ty)
    D = (xD + tx, yD + ty)
    C = (xC + tx, yC + ty)
    
    # Inner trapezoid
    x_mid = (0.0 + (xB + tx)) / 2.0
    x_off = x_mid - b / 2.0
    y_off = d
    A_s = (x_off, y_off)
    B_s = (x_off + b, y_off)
    D_s = (x_off + delta, y_off + h)
    C_s = (x_off + delta + a, y_off + h)
    
    return {
        'large': [A, B, C, D],
        'small': [A_s, B_s, C_s, D_s]
    }

def initialize_geometry_calculations(params):
    """Initialize geometry calculations"""
    global GEOMETRY_DATA
    
    current_trapezoids = compute_trapezoids(
        a=params["top_width"],
        b=params["bottom_width"],
        h=params["height"],
        d=params["wall_thickness"]
    )
    
    GEOMETRY_DATA = {
        "current_geometry": {
            "length": params["length"],
            "coords_large": current_trapezoids['large'],
            "coords_small": current_trapezoids['small']
        },
        "top_surface": {
            "center": (
                params["length"] / 2, 
                (current_trapezoids['small'][2][0] + current_trapezoids['small'][3][0]) / 2,
                current_trapezoids['small'][3][1]
            ),
            "bounds": {
                "x_min": 0.0,
                "x_max": params["length"],
                "y_min": current_trapezoids['small'][3][0],
                "y_max": current_trapezoids['small'][2][0],
                "z": current_trapezoids['small'][3][1]
            }
        },
        "observe_planes": {
            "width": current_trapezoids['large'][1][0],
            "height": current_trapezoids['large'][3][1],
            "front_x": 0.0,
            "rear_x": params["length"],
            "y_position": 0.0,
            "z_position": 0.0
        }
    }

def _new_project(tag):
    """Create new HFSS project"""
    proj = oDesktop.NewProject(tag)
    proj.InsertDesign("HFSS", "TrapezoidalCavity", "DrivenModal", "")
    des = proj.SetActiveDesign("TrapezoidalCavity")
    edt = des.SetActiveEditor("3D Modeler")
    return proj, des, edt

def create_trapezoidal_shell(edt, material_name="pec"):
    """Create trapezoidal outer shell"""
    large_coords = GEOMETRY_DATA["current_geometry"]["coords_large"]
    A, B, C, D = large_coords[0], large_coords[1], large_coords[2], large_coords[3]
    
    edt.CreatePolyline(
        [
            "NAME:PolylineParameters",
            "IsPolylineCovered:=", True,
            "IsPolylineClosed:=", True,
            [
                "NAME:PolylinePoints",
                ["NAME:PLPoint", "X:=", "0mm", "Y:=", "{}mm".format(A[0]), "Z:=", "{}mm".format(A[1])],
                ["NAME:PLPoint", "X:=", "0mm", "Y:=", "{}mm".format(B[0]), "Z:=", "{}mm".format(B[1])],
                ["NAME:PLPoint", "X:=", "0mm", "Y:=", "{}mm".format(C[0]), "Z:=", "{}mm".format(C[1])],
                ["NAME:PLPoint", "X:=", "0mm", "Y:=", "{}mm".format(D[0]), "Z:=", "{}mm".format(D[1])]
            ],
            [
                "NAME:PolylineSegments",
                ["NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 0, "NoOfPoints:=", 2],
                ["NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 1, "NoOfPoints:=", 2],
                ["NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 2, "NoOfPoints:=", 2],
                ["NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 3, "NoOfPoints:=", 2]
            ]
        ],
        [
            "NAME:Attributes",
            "Name:=", "TrapezoidalShell",
            "Flags:=", "",
            "Color:=", "(143 175 143)",
            "Transparency:=", 0,
            "PartCoordinateSystem:=", "Global",
            "UDMId:=", "",
            "MaterialValue:=", "\"{}\"".format(material_name),
            "SurfaceMaterialValue:=", "\"\"",
            "SolveInside:=", False,
            "ShellElement:=", False,
            "ShellElementThickness:=", "0mm",
            "IsMaterialEditable:=", True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:=", False
        ])
    
    edt.SweepAlongVector(
        [
            "NAME:Selections",
            "Selections:=", "TrapezoidalShell",
            "NewPartsModelFlag:=", "Model"
        ],
        [
            "NAME:VecSweepParameters",
            "DraftAngle:=", "0deg",
            "DraftType:=", "Round",
            "CheckFaceFaceIntersection:=", False,
            "SweepVectorX:=", "{}mm".format(GEOMETRY_DATA["current_geometry"]["length"]),
            "SweepVectorY:=", "0mm",
            "SweepVectorZ:=", "0mm"
        ])
    
    return "TrapezoidalShell"

def create_inner_cavity(edt):
    """Create inner cavity"""
    small_coords = GEOMETRY_DATA["current_geometry"]["coords_small"]
    A_s, B_s, C_s, D_s = small_coords[0], small_coords[1], small_coords[2], small_coords[3]
    
    edt.CreatePolyline(
        [
            "NAME:PolylineParameters",
            "IsPolylineCovered:=", True,
            "IsPolylineClosed:=", True,
            [
                "NAME:PolylinePoints",
                ["NAME:PLPoint", "X:=", "0mm", "Y:=", "{}mm".format(A_s[0]), "Z:=", "{}mm".format(A_s[1])],
                ["NAME:PLPoint", "X:=", "0mm", "Y:=", "{}mm".format(B_s[0]), "Z:=", "{}mm".format(B_s[1])],
                ["NAME:PLPoint", "X:=", "0mm", "Y:=", "{}mm".format(C_s[0]), "Z:=", "{}mm".format(C_s[1])],
                ["NAME:PLPoint", "X:=", "0mm", "Y:=", "{}mm".format(D_s[0]), "Z:=", "{}mm".format(D_s[1])]
            ],
            [
                "NAME:PolylineSegments",
                ["NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 0, "NoOfPoints:=", 2],
                ["NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 1, "NoOfPoints:=", 2],
                ["NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 2, "NoOfPoints:=", 2],
                ["NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", 3, "NoOfPoints:=", 2]
            ]
        ],
        [
            "NAME:Attributes",
            "Name:=", "TrapezoidalCavity",
            "Flags:=", "",
            "Color:=", "(255 0 0)",
            "Transparency:=", 0.5,
            "PartCoordinateSystem:=", "Global",
            "UDMId:=", "",
            "MaterialValue:=", "\"vacuum\"",
            "SurfaceMaterialValue:=", "\"\"",
            "SolveInside:=", True,
            "ShellElement:=", False,
            "ShellElementThickness:=", "0mm",
            "IsMaterialEditable:=", True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:=", False
        ])
    
    edt.SweepAlongVector(
        [
            "NAME:Selections",
            "Selections:=", "TrapezoidalCavity",
            "NewPartsModelFlag:=", "Model"
        ],
        [
            "NAME:VecSweepParameters",
            "DraftAngle:=", "0deg",
            "DraftType:=", "Round",
            "CheckFaceFaceIntersection:=", False,
            "SweepVectorX:=", "{}mm".format(GEOMETRY_DATA["current_geometry"]["length"]),
            "SweepVectorY:=", "0mm",
            "SweepVectorZ:=", "0mm"
        ])
    
    return "TrapezoidalCavity"

def create_complete_cavity(edt):
    """Create complete cavity"""
    outer_shell = create_trapezoidal_shell(edt)
    inner_cavity = create_inner_cavity(edt)
    
    edt.Subtract(
        [
            "NAME:Selections",
            "Blank Parts:=", outer_shell,
            "Tool Parts:=", inner_cavity
        ],
        [
            "NAME:SubtractParameters",
            "KeepOriginals:=", False
        ])
    
    return outer_shell

def calculate_tooth_positions(groove_params):
    """Calculate tooth positions with four quadrant symmetry"""
    top_surface_data = GEOMETRY_DATA["top_surface"]
    center_point = top_surface_data["center"]
    bounds = top_surface_data["bounds"]
    
    center_x, center_y, center_z = center_point
    
    # Tooth spacing = tooth size + gap
    tooth_pitch_x = groove_params["tooth_length"] + groove_params["groove_length"]
    tooth_pitch_y = groove_params["tooth_width"] + groove_params["groove_width"]
    
    # Available space in first quadrant
    available_x = bounds["x_max"] - center_x
    available_y = bounds["y_max"] - center_y
    
    # Apply extension factor
    extended_x = available_x * groove_params["extension_factor"]
    extended_y = available_y * groove_params["extension_factor"]
    
    # Calculate teeth count
    max_teeth_x = int(extended_x / tooth_pitch_x)
    max_teeth_y = int(extended_y / tooth_pitch_y)
    
    # Generate first quadrant positions
    quadrant_1_positions = []
    for i in range(max_teeth_x):
        for j in range(max_teeth_y):
            tooth_x = center_x + (i + 0.5) * tooth_pitch_x
            tooth_y = center_y + (j + 0.5) * tooth_pitch_y
            quadrant_1_positions.append((tooth_x, tooth_y, center_z))
    
    # Four quadrant symmetry
    all_positions = []
    for (x, y, z) in quadrant_1_positions:
        offset_x = x - center_x
        offset_y = y - center_y
        
        quadrants = [
            (center_x + offset_x, center_y + offset_y, z),
            (center_x - offset_x, center_y + offset_y, z),
            (center_x - offset_x, center_y - offset_y, z),
            (center_x + offset_x, center_y - offset_y, z),
        ]
        all_positions.extend(quadrants)
    
    return all_positions

def enforce_tooth_boundary_constraints(center_x, center_y, tooth_length, tooth_width):
    """Enforce boundary constraints for individual tooth"""
    bounds = GEOMETRY_DATA["top_surface"]["bounds"]
    
    # Calculate maximum allowed dimensions based on position and boundaries
    max_half_length_left = center_x - bounds["x_min"]
    max_half_length_right = bounds["x_max"] - center_x
    max_half_width_bottom = center_y - bounds["y_min"]  
    max_half_width_top = bounds["y_max"] - center_y
    
    # Calculate actual allowed dimensions
    max_length = min(tooth_length, max_half_length_left * 2, max_half_length_right * 2)
    max_width = min(tooth_width, max_half_width_bottom * 2, max_half_width_top * 2)
    
    # Ensure positive dimensions
    actual_length = max(0.05, max_length)  # Minimum 0.05mm
    actual_width = max(0.05, max_width)
    
    is_adjusted = (actual_length != tooth_length or actual_width != tooth_width)
    
    return {
        "length": actual_length,
        "width": actual_width,
        "is_adjusted": is_adjusted,
        "adjustment_info": {
            "original": [tooth_length, tooth_width],
            "adjusted": [actual_length, actual_width],
            "position": [center_x, center_y]
        }
    }

def create_teeth_array(edt, groove_params, material_name="pec"):
    """Create teeth array with boundary constraints"""
    tooth_positions = calculate_tooth_positions(groove_params)
    bounds = GEOMETRY_DATA["top_surface"]["bounds"]
    
    tooth_objects = []
    created_count = 0
    adjustment_log = []
    
    # Get top surface Z coordinate
    top_surface_z = bounds["z"]
    
    for idx, (center_x, center_y, center_z) in enumerate(tooth_positions):
        
        if created_count >= groove_params["max_tooth_count"]:
            break
        
        # Apply boundary constraints to tooth dimensions
        constrained = enforce_tooth_boundary_constraints(
            center_x, center_y,
            groove_params["tooth_length"], 
            groove_params["tooth_width"]
        )
        
        actual_tooth_length = constrained["length"]
        actual_tooth_width = constrained["width"]
        
        # Skip if tooth becomes too small
        if actual_tooth_length < 0.1 or actual_tooth_width < 0.1:
            continue
        
        # Calculate tooth position (centered)
        tooth_x_min = center_x - actual_tooth_length / 2
        tooth_y_min = center_y - actual_tooth_width / 2
        
        # Use top surface Z coordinate for tooth positioning
        tooth_z_position = top_surface_z - groove_params["tooth_depth"]
        
        # Create tooth with constrained dimensions
        tooth_id = "GrooveTooth_{}".format(created_count)
        
        edt.CreateBox(
            [
                "NAME:BoxParameters",
                "XPosition:=", "{}mm".format(tooth_x_min),
                "YPosition:=", "{}mm".format(tooth_y_min),
                "ZPosition:=", "{}mm".format(tooth_z_position),
                "XSize:=", "{}mm".format(math.floor(actual_tooth_length * 10000) / 10000),
                "YSize:=", "{}mm".format(math.floor(actual_tooth_width * 10000) / 10000),
                "ZSize:=", "{}mm".format(math.floor(groove_params["tooth_depth"] * 10000) / 10000)
            ],
            [
                "NAME:Attributes",
                "Name:=", tooth_id,
                "Flags:=", "",
                "Color:=", "(143 175 143)",
                "Transparency:=", 0,
                "PartCoordinateSystem:=", "Global",
                "UDMId:=", "",
                "MaterialValue:=", "\"{}\"".format(material_name),
                "SurfaceMaterialValue:=", "\"\"",
                "SolveInside:=", False,
                "ShellElement:=", False,
                "ShellElementThickness:=", "0mm",
                "IsMaterialEditable:=", True,
                "UseMaterialAppearance:=", False,
                "IsLightweight:=", False
            ])
        
        tooth_objects.append(tooth_id)
        created_count += 1
        
        # Log adjustment if applied
        if constrained["is_adjusted"]:
            adjustment_log.append({
                "tooth_id": tooth_id,
                "original_size": constrained["adjustment_info"]["original"],
                "adjusted_size": constrained["adjustment_info"]["adjusted"],
                "position": constrained["adjustment_info"]["position"]
            })
    
    return tooth_objects

def setup_wave_port(des, edt):
    """Setup wave port"""
    large_coords = GEOMETRY_DATA["current_geometry"]["coords_large"]
    port_width = large_coords[1][0]
    port_height = large_coords[3][1]
    
    edt.CreateRectangle(
        [
            "NAME:RectangleParameters",
            "IsCovered:=", True,
            "XStart:=", "0mm",
            "YStart:=", "0mm",
            "ZStart:=", "0mm",
            "Width:=", "{}mm".format(port_width),
            "Height:=", "{}mm".format(port_height),
            "WhichAxis:=", "X"
        ],
        [
            "NAME:Attributes",
            "Name:=", "CenterPort",
            "Flags:=", "",
            "Color:=", "(255 0 0)",
            "Transparency:=", 0,
            "PartCoordinateSystem:=", "Global",
            "UDMId:=", "",
            "MaterialValue:=", "\"vacuum\"",
            "SurfaceMaterialValue:=", "\"\"",
            "SolveInside:=", True,
            "ShellElement:=", False,
            "ShellElementThickness:=", "0mm",
            "IsMaterialEditable:=", True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:=", False
        ])

    bnd = des.GetModule("BoundarySetup")
    bnd.AssignWavePort(
        [
            "NAME:1",
            "Objects:=", ["CenterPort"],
            "NumModes:=", 1,
            "UseLineModeAlignment:=", False,
            "DoDeembed:=", False,
            "RenormalizeAllTerminals:=", True,
            [
                "NAME:Modes",
                [
                    "NAME:Mode1",
                    "ModeNum:=", 1,
                    "UseIntLine:=", False,
                    "CharImp:=", "Zpi"
                ]
            ],
            "ShowReporterFilter:=", False,
            "ReporterFilter:=", [True],
            "UseAnalyticAlignment:=", False
        ])

def add_observe_planes(edt):
    """Add observe planes for field monitoring"""
    observe_data = GEOMETRY_DATA["observe_planes"]
    plane_width = observe_data["width"]
    plane_height = observe_data["height"]
    rear_x = observe_data["rear_x"]
    
    # Front observe plane at X=0
    edt.CreateRectangle(
        [
            "NAME:RectangleParameters",
            "IsCovered:=", True,
            "XStart:=", "0mm",
            "YStart:=", "0mm",
            "ZStart:=", "0mm",
            "Width:=", "{}mm".format(plane_width),
            "Height:=", "{}mm".format(plane_height),
            "WhichAxis:=", "X"
        ],
        [
            "NAME:Attributes",
            "Name:=", "FrontObservePlane",
            "Flags:=", "",
            "Color:=", "(0 255 0)",
            "Transparency:=", 0.5,
            "PartCoordinateSystem:=", "Global",
            "UDMId:=", "",
            "MaterialValue:=", "\"vacuum\"",
            "SurfaceMaterialValue:=", "\"\"",
            "SolveInside:=", True,
            "ShellElement:=", False,
            "ShellElementThickness:=", "0mm",
            "IsMaterialEditable:=", True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:=", False
        ])

    # Rear observe plane (target output plane)
    edt.CreateRectangle(
        [
            "NAME:RectangleParameters",
            "IsCovered:=", True,
            "XStart:=", "{}mm".format(rear_x),
            "YStart:=", "0mm",
            "ZStart:=", "0mm",
            "Width:=", "{}mm".format(plane_width),
            "Height:=", "{}mm".format(plane_height),
            "WhichAxis:=", "X"
        ],
        [
            "NAME:Attributes",
            "Name:=", "RearObservePlane",
            "Flags:=", "",
            "Color:=", "(0 255 0)",
            "Transparency:=", 0.5,
            "PartCoordinateSystem:=", "Global",
            "UDMId:=", "",
            "MaterialValue:=", "\"vacuum\"",
            "SurfaceMaterialValue:=", "\"\"",
            "SolveInside:=", True,
            "ShellElement:=", False,
            "ShellElementThickness:=", "0mm",
            "IsMaterialEditable:=", True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:=", False
        ])

def setup_region_and_boundaries(des, edt):
    """Setup region and observe planes"""
    edt.CreateRegion(
        [
            "NAME:RegionParameters",
            "+XPaddingType:=", "Absolute Offset",
            "+XPadding:=", "0mm",
            "-XPaddingType:=", "Absolute Offset", 
            "-XPadding:=", "0mm",
            "+YPaddingType:=", "Absolute Offset",
            "+YPadding:=", "0mm",
            "-YPaddingType:=", "Absolute Offset",
            "-YPadding:=", "0mm",
            "+ZPaddingType:=", "Absolute Offset",
            "+ZPadding:=", "0mm",
            "-ZPaddingType:=", "Absolute Offset",
            "-ZPadding:=", "0mm"
        ],
        [
            "NAME:Attributes",
            "Name:=", "Region",
            "Flags:=", "Wireframe#",
            "Color:=", "(143 175 143)",
            "Transparency:=", 0.8,
            "PartCoordinateSystem:=", "Global",
            "UDMId:=", "",
            "MaterialValue:=", "\"vacuum\"",
            "SurfaceMaterialValue:=", "\"\"",
            "SolveInside:=", True,
            "ShellElement:=", False,
            "ShellElementThickness:=", "nan ",
            "IsMaterialEditable:=", True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:=", False
        ])
    
    add_observe_planes(edt)

def setup_analysis(des, freq_ghz=2.45):
    """Setup analysis"""
    des.GetModule("AnalysisSetup").InsertSetup("HfssDriven",
        ["NAME:Setup1", "Frequency:=", "{}GHz".format(freq_ghz), "MaxDeltaS:=", 0.1,
         "PortsOnly:=", False, "MaximumPasses:=", 3])

def set_excitation(des):
    """Set excitation to 1000W"""
    oModule = des.GetModule("Solutions")
    oModule.EditSources(
        [
            [
                "IncludePortPostProcessing:=", False,
                "SpecifySystemPower:=", False
            ],
            [
                "Name:=", "1:1",
                "Magnitude:=", "1000W",
                "Phase:=", "0deg"
            ]
        ])

def run_single_simulation(trapezoidal_params, groove_params, iteration, data_dir):
    """Execute single HFSS simulation"""
    # Initialize geometry
    initialize_geometry_calculations(trapezoidal_params)
    
    # Create project with iteration tag
    tag = "TrapezoidalCavity_iter_{}_{}".format(iteration, _timestamp())
    proj, des, edt = _new_project(tag)

    try:
        # Create geometry
        shell_name = create_complete_cavity(edt)
        tooth_objects = create_teeth_array(edt, groove_params)
        
        # Setup simulation with observe planes
        setup_region_and_boundaries(des, edt)
        setup_wave_port(des, edt)
        setup_analysis(des, trapezoidal_params["f0"])
        
        # Run analysis
        des.AnalyzeAll()
        set_excitation(des)
        
        # Export field data to .fld file
        field_result = export_field_to_fld_file(des, iteration, data_dir, trapezoidal_params["f0"])
        
        return {
            "iteration": iteration,
            "teeth_count": len(tooth_objects),
            "field_export": field_result,
            "simulation_status": "completed"
        }
        
    except Exception as e:
        return {
            "iteration": iteration,
            "field_export": {"success": False, "error": str(e)},
            "simulation_status": "failed"
        }



# =====================================================================
# SHARED COUNTER MANAGEMENT (Simple functions for HFSS compatibility)
# =====================================================================

def read_counter_from_file(counter_file_path):
    """
    Simple counter file reader for HFSS script
    Returns 0 if file doesn't exist or has issues
    """
    try:
        if not os.path.exists(counter_file_path):
            return 0
        
        with open(counter_file_path, 'r') as f:
            counter_str = f.read().strip()
            if not counter_str:
                return 0
            return int(counter_str)
    except:
        return 0
# =====================================================================
# MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    # Default execution: run main simulation loop
    COUNTER_FILE_PATH = "C:\\Users\\USER\\Desktop\\wave\\project\\current_input_counter.txt"
    DATA_DIR = "C:\\Users\\USER\\Desktop\\wave\\project\\Data1"
    SIMULATION_BATCH_SIZE = 50  # Process 50 iterations at a time
    
    # Read current counter to determine starting iteration
    START_ITERATION = read_counter_from_file(COUNTER_FILE_PATH)
    END_ITERATION = START_ITERATION + SIMULATION_BATCH_SIZE
    DATA_DIR = "C:\\Users\\USER\\Desktop\\wave\\project\\Data1"
    
    main_simulation_loop(
        start_iter=START_ITERATION,
        end_iter=END_ITERATION,
        data_dir=DATA_DIR
    )
