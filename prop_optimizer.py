import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
MOTOR_CSV_PATH = 'V4006_data.csv'
APC_DATA_DIR   = './apc-data/'  # UPDATE THIS TO YOUR APC .dat FOLDER

# Competition & System Constraints
POWER_LIMIT_W = 225.0
VOLTAGE_V     = 14.8            # 4S Lipo Nominal
RHO           = 1.225           # Air density (kg/m^3)
MAX_AIRSPEED  = 20.0            # Max speed to plot (m/s)
NUM_V_POINTS  = 30              # Number of points on the thrust curve

# Search Space Filter
MIN_DIAM, MAX_DIAM = 7.0, 15.0  # inches
MIN_PITCH, MAX_PITCH = 2.0, 8.0 # inches

# ==========================================
# 2. MOTOR CHARACTERIZATION
# ==========================================
def characterize_motor(csv_path):
    """
    Fits DC motor constants (Kv, Rm) using least squares on bench data.
    Equation: RPM = Kv * Voltage - (Kv * Rm) * Current
    """
    df = pd.read_csv(csv_path)
    
    # We want to solve Y = X * Beta
    # Y = RPM, X1 = Voltage, X2 = -Current
    Y = df['rpm'].values
    X = np.column_stack((df['voltage_V'].values, -df['current_A'].values))
    
    # Least squares fit
    Beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    
    Kv = Beta[0]
    Rm = Beta[1] / Kv
    I0 = 0.5  # Approximate No-Load Current (Amps) if not explicitly measured at 0 torque
    Kt = 60.0 / (2.0 * np.pi * Kv) # Torque constant (Nm/A)
    
    print(f"--- Motor Characterization ---")
    print(f"Fitted Kv: {Kv:.1f} RPM/V")
    print(f"Fitted Rm: {Rm:.4f} Ohms")
    print(f"Calc Kt:   {Kt:.4f} Nm/A\n")
    
    return {'Kv': Kv, 'Rm': Rm, 'I0': I0, 'Kt': Kt}

# ==========================================
# 3. APC PROPELLER PARSER
# ==========================================
def parse_apc_directory(directory, d_range, p_range):
    """
    Scans directory for APC .dat files, filters by size, extracts J, Ct, Cp.
    """
    props = []
    filepaths = glob.glob(os.path.join(directory, '*.dat'))
    
    if not filepaths:
        print(f"WARNING: No .dat files found in {directory}. Please check the path.")
        return props

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        
        # Approximate size extraction from standard APC naming (e.g., 10x5E.dat)
        try:
            # Simple heuristic to extract numbers: replace non-digits with spaces
            clean_name = ''.join([c if c.isdigit() or c == '.' else ' ' for c in filename])
            parts = [float(x) for x in clean_name.split() if x]
            if len(parts) >= 2:
                D_in, P_in = parts[0], parts[1]
            else:
                continue
        except:
            continue
            
        # Filter by size
        if not (d_range[0] <= D_in <= d_range[1] and p_range[0] <= P_in <= p_range[1]):
            continue
            
        # Read the file and extract performance data
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            J_vals, Ct_vals, Cp_vals = [], [], []
            for line in lines:
                cols = line.split()
                # A standard APC data row has 7 numeric columns. We look for those.
                if len(cols) >= 7:
                    try:
                        j  = float(cols[1]) # Advance Ratio
                        ct = float(cols[3]) # Thrust Coeff
                        cp = float(cols[4]) # Power Coeff
                        J_vals.append(j)
                        Ct_vals.append(ct)
                        Cp_vals.append(cp)
                    except ValueError:
                        continue
            
            if len(J_vals) > 5:
                # Average out duplicates (APC files sometimes list multiple RPM runs sequentially)
                # For simplicity in this single script, we build a single averaged interpolator
                df = pd.DataFrame({'J': J_vals, 'Ct': Ct_vals, 'Cp': Cp_vals})
                df = df.groupby('J').mean().reset_index()
                
                props.append({
                    'name': filename.replace('.dat', ''),
                    'D_m': D_in * 0.0254,  # Convert inches to meters
                    'D_in': D_in,
                    'P_in': P_in,
                    'J': df['J'].values,
                    'Ct': df['Ct'].values,
                    'Cp': df['Cp'].values
                })
        except Exception as e:
            pass
            
    return props

# ==========================================
# 4. POWERTRAIN SOLVER
# ==========================================
def solve_dynamic_thrust(prop, motor, V_grid):
    """
    Finds operating RPM at each airspeed by equating motor torque and aero torque.
    Respects the 225W power limit.
    """
    # Create interpolation functions for the propeller coefficients
    # Fill values with 0 if outside the J range (e.g., extremely high advance ratio)
    interp_Ct = interp1d(prop['J'], prop['Ct'], bounds_error=False, fill_value=(prop['Ct'][0], 0.0))
    interp_Cp = interp1d(prop['J'], prop['Cp'], bounds_error=False, fill_value=(prop['Cp'][0], 0.0))
    
    thrust_curve = []
    power_curve = []
    
    for V in V_grid:
        # Residual function to find the equilibrium RPM
        def torque_residual(rpm_guess):
            n = rpm_guess[0] / 60.0  # revs per second
            if n <= 0: n = 1e-5
            
            J = V / (n * prop['D_m'])
            Cp = interp_Cp(J)
            
            # Aerodynamic Torque (Nm)
            Q_aero = (Cp * RHO * (n**2) * (prop['D_m']**5)) / (2 * np.pi)
            
            # Motor Current required for this torque
            I_req = (Q_aero / motor['Kt']) + motor['I0']
            
            # Software/Hardware Power Limiter Check
            max_I = POWER_LIMIT_W / VOLTAGE_V
            I = min(I_req, max_I) # Clamp current if it exceeds limit
            
            # Motor Voltage equation (V_in = V_emf + I*Rm)
            # V_emf = RPM / Kv
            V_motor_available = VOLTAGE_V - I * motor['Rm']
            RPM_motor = motor['Kv'] * max(0, V_motor_available)
            
            return RPM_motor - rpm_guess[0]
            
        # Solve for RPM (Initial guess: 6000 RPM)
        sol = fsolve(torque_residual, [6000.0])
        rpm_op = sol[0]
        
        # Recompute final state at equilibrium RPM
        n_op = rpm_op / 60.0
        J_op = V / (n_op * prop['D_m']) if n_op > 0 else 0
        Ct_op = interp_Ct(J_op)
        Cp_op = interp_Cp(J_op)
        
        # Calculate Thrust (Newtons)
        T_N = Ct_op * RHO * (n_op**2) * (prop['D_m']**4)
        
        # Calculate Power (Watts)
        Q_aero = (Cp_op * RHO * (n_op**2) * (prop['D_m']**5)) / (2 * np.pi)
        I_req = (Q_aero / motor['Kt']) + motor['I0']
        P_elec = VOLTAGE_V * min(I_req, POWER_LIMIT_W / VOLTAGE_V)
        
        thrust_curve.append(max(0, T_N))
        power_curve.append(P_elec)
        
    return np.array(thrust_curve), np.array(power_curve)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    motor = characterize_motor(MOTOR_CSV_PATH)
    
    print(f"Scanning APC directory for props ({MIN_DIAM}-{MAX_DIAM}in dia, {MIN_PITCH}-{MAX_PITCH}in pitch)...")
    props = parse_apc_directory(APC_DATA_DIR, [MIN_DIAM, MAX_DIAM], [MIN_PITCH, MAX_PITCH])
    print(f"Found {len(props)} matching propellers.\n")
    
    if not props:
        return
        
    V_grid = np.linspace(0, MAX_AIRSPEED, NUM_V_POINTS)
    results = []
    
    print("Simulating Powertrain Matching...")
    for prop in props:
        T_curve_N, P_curve_W = solve_dynamic_thrust(prop, motor, V_grid)
        
        # Convert Thrust from Newtons to Grams for easier RC intuition
        T_curve_g = T_curve_N * 101.97
        
        results.append({
            'name': prop['name'],
            'D_in': prop['D_in'],
            'P_in': prop['P_in'],
            'static_thrust_g': T_curve_g[0], # Thrust at V=0
            'thrust_curve_g': T_curve_g,
            'power_curve_w': P_curve_W
        })
        
    # Sort results by Maximum Static Thrust
    results.sort(key=lambda x: x['static_thrust_g'], reverse=True)
    
    # --- PRINT TOP 10 ---
    print("\n--- TOP 10 PROPELLERS FOR MAX STATIC THRUST ---")
    print(f"{'Propeller':<15} | {'Static Thrust (g)':<20} | {'Static Power (W)'}")
    print("-" * 55)
    for res in results[:10]:
        static_pwr = res['power_curve_w'][0]
        print(f"{res['name']:<15} | {res['static_thrust_g']:<20.1f} | {static_pwr:.1f}")
        
    # --- PLOT DYNAMIC BEHAVIOR FOR TOP 5 ---
    plt.figure(figsize=(10, 6))
    plt.title("Dynamic Thrust Behavior (Top 5 Static Performers)", fontsize=14)
    
    for res in results[:5]:
        plt.plot(V_grid, res['thrust_curve_g'], linewidth=2.5, 
                 label=f"{res['name']} ({res['static_thrust_g']:.0f}g static)")
                 
    plt.xlabel("Airspeed (m/s)", fontsize=12)
    plt.ylabel("Available Thrust (grams)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()