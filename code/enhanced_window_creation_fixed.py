import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import re
from multiprocessing import Pool, cpu_count
from functools import partial

# =============================== 
# CONFIG
# =============================== 
WINDOW_MINUTES = 5
WINDOW_SEC = WINDOW_MINUTES * 60

INPUT_CSV = "BGL_structured.csv"
OUTPUT_CSV = "windows_with_features.csv"  # No labels yet!

NUM_PROCESSES = max(1, cpu_count() - 1)

SEVERITY_MAP = {
    'INFO': 0,
    'WARNING': 1,
    'WARN': 1,
    'ERROR': 2,
    'FATAL': 3
}

# =============================== 
# PRE-COMPILED REGEX PATTERNS
# =============================== 
HEX_PATTERN = re.compile(r'0x[0-9a-fA-F]+')
NUM_PATTERN = re.compile(r'\b\d+\b')
IP_PATTERN = re.compile(r'\d+\.\d+\.\d+\.\d+')

# =============================== 
# HELPER FUNCTIONS
# =============================== 
def extract_log_template_vectorized(content_series):
    """Vectorized template extraction"""
    templates = content_series.astype(str)
    templates = templates.str.replace(HEX_PATTERN, '', regex=True)
    templates = templates.str.replace(NUM_PATTERN, '', regex=True)
    templates = templates.str.replace(IP_PATTERN, '', regex=True)
    return templates.str.strip()

def get_time_features_vectorized(timestamps):
    """Extract temporal features - vectorized"""
    dt_series = pd.to_datetime(timestamps, unit='s')
    if isinstance(dt_series, pd.DatetimeIndex):
        return pd.DataFrame({
            'hour': dt_series.hour,
            'day_of_week': dt_series.dayofweek,
            'is_weekend': (dt_series.dayofweek >= 5).astype(int),
            'is_night': ((dt_series.hour < 6) | (dt_series.hour >= 22)).astype(int)
        })
    else:
        return pd.DataFrame({
            'hour': dt_series.dt.hour,
            'day_of_week': dt_series.dt.dayofweek,
            'is_weekend': (dt_series.dt.dayofweek >= 5).astype(int),
            'is_night': ((dt_series.dt.hour < 6) | (dt_series.dt.hour >= 22)).astype(int)
        })

def extract_window_features_optimized(window_indices, g, templates, severity_numeric, time_features):
    """Extract 18 features using pre-computed values"""
    if len(window_indices) == 0:
        return {
            'total_logs': 0, 'error_count': 0, 'fatal_count': 0, 
            'warning_count': 0, 'info_count': 0, 'unique_templates': 0,
            'unique_components': 0, 'max_severity': 0, 'avg_severity': 0,
            'severity_std': 0, 'kernel_logs': 0, 'app_logs': 0,
            'has_kernel_error': 0, 'log_rate': 0, 'hour': 0,
            'day_of_week': 0, 'is_weekend': 0, 'is_night': 0
        }
    
    window_data = g.iloc[window_indices]
    total_logs = len(window_indices)
    
    # Vectorized level counts
    levels = window_data['Level'].values
    error_count = np.sum(levels == 'ERROR')
    fatal_count = np.sum(levels == 'FATAL')
    warning_count = np.sum((levels == 'WARNING') | (levels == 'WARN'))
    info_count = np.sum(levels == 'INFO')
    
    # Template diversity
    unique_templates = len(set(templates[window_indices]))
    
    # Component analysis
    if 'Component' in window_data.columns:
        components = window_data['Component'].values
        unique_components = len(set(components))
        kernel_logs = np.sum(components == 'KERNEL')
        app_logs = np.sum(components == 'APP')
        has_kernel_error = int(np.any((components == 'KERNEL') & 
                                      ((levels == 'ERROR') | (levels == 'FATAL'))))
    else:
        unique_components = kernel_logs = app_logs = has_kernel_error = 0
    
    # Severity statistics
    severity_vals = severity_numeric[window_indices]
    max_severity = np.max(severity_vals)
    avg_severity = np.mean(severity_vals)
    severity_std = np.std(severity_vals) if len(severity_vals) > 1 else 0
    
    # Time-based features
    if total_logs > 1:
        times = window_data['Time'].values
        time_diff = times[-1] - times[0]
        log_rate = total_logs / max(time_diff, 1)
    else:
        log_rate = 0
    
    # Temporal features from last log
    last_idx = window_indices[-1]
    
    return {
        'total_logs': total_logs,
        'error_count': int(error_count),
        'fatal_count': int(fatal_count),
        'warning_count': int(warning_count),
        'info_count': int(info_count),
        'unique_templates': unique_templates,
        'unique_components': unique_components,
        'max_severity': float(max_severity),
        'avg_severity': float(avg_severity),
        'severity_std': float(severity_std),
        'kernel_logs': int(kernel_logs),
        'app_logs': int(app_logs),
        'has_kernel_error': has_kernel_error,
        'log_rate': float(log_rate),
        'hour': int(time_features['hour'].iloc[last_idx]),
        'day_of_week': int(time_features['day_of_week'].iloc[last_idx]),
        'is_weekend': int(time_features['is_weekend'].iloc[last_idx]),
        'is_night': int(time_features['is_night'].iloc[last_idx])
    }

def process_node_optimized(node_data):
    """
    Process a single node - designed for parallel execution
    CREATES WINDOWS WITH 18 FEATURES ONLY (NO LABELING)
    """
    node, g = node_data
    g = g.sort_values("Time").reset_index(drop=True)
    times = g["Time"].values
    n = len(g)
    
    # Pre-compute all values
    templates = extract_log_template_vectorized(g['Content']).values
    severity_numeric = g['Level'].map(SEVERITY_MAP).fillna(0).values
    time_features = get_time_features_vectorized(times)
    node_name = g['Node'].iloc[0]
    
    # ============================================
    # Create windows with FEATURES only
    # ============================================
    node_windows = []
    
    start = 0
    for end in range(n):
        while times[end] - times[start] > WINDOW_SEC:
            start += 1
        
        window_indices = list(range(start, end + 1))
        window_time = times[end]
        
        # Extract 18 features for this window
        features = extract_window_features_optimized(
            window_indices, g, templates, severity_numeric, time_features
        )
        
        # Store window: Time, Node, + 18 features (NO LABEL YET)
        window_record = {
            "Time": window_time,
            "Node": node_name,
            **features
        }
        
        node_windows.append(window_record)
    
    return node_windows

def main():
    """Main function"""
    print("="*70)
    print("WINDOW CREATION - FEATURES ONLY (NO LABELING)")
    print("="*70)
    print("Purpose: Create 5-minute windows with 18 features each")
    print("Labeling: Will be done separately by label_windows.py using episodes")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(INPUT_CSV)
    df["Time"] = pd.to_numeric(df["Time"])
    df = df.sort_values(["Node", "Time"]).reset_index(drop=True)

    print(f"Loaded {len(df):,} log entries")
    print(f"Unique nodes: {df['Node'].nunique()}")

    # Build windows
    print(f"\nCreating {WINDOW_MINUTES}-minute windows...")
    print(f"Using {NUM_PROCESSES} parallel processes...\n")

    node_groups = list(df.groupby("Node"))

    print("Processing nodes in parallel...")
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(process_node_optimized, node_groups)

    # Flatten results
    all_windows = []
    for node_windows in results:
        all_windows.extend(node_windows)

    print(f"\nCreated {len(all_windows):,} windows")

    # Create DataFrame
    windows_df = pd.DataFrame(all_windows)

    # Save
    windows_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")

    # Analysis
    print("\n" + "="*70)
    print("WINDOW STATISTICS")
    print("="*70)
    
    print(f"\nWindows created: {len(windows_df):,}")
    print(f"Nodes: {windows_df['Node'].nunique()}")
    print(f"Features per window: {len(windows_df.columns) - 2}")  # Minus Time and Node
    
    print("\nFeature ranges:")
    print(f"  fatal_count: [{windows_df['fatal_count'].min()}, {windows_df['fatal_count'].max()}]")
    print(f"  error_count: [{windows_df['error_count'].min()}, {windows_df['error_count'].max()}]")
    print(f"  total_logs: [{windows_df['total_logs'].min()}, {windows_df['total_logs'].max()}]")
    
    print("\nAverage feature values:")
    print(f"  Mean fatal_count: {windows_df['fatal_count'].mean():.2f}")
    print(f"  Mean error_count: {windows_df['error_count'].mean():.2f}")
    print(f"  Mean total_logs: {windows_df['total_logs'].mean():.2f}")
    
    print("\n" + "="*70)
    print("NEXT STEP")
    print("="*70)
    print("\nRun label_windows.py to:")
    print("  1. Group windows into episodes (8 windows = 40 min)")
    print("  2. Analyze episodes to understand patterns")
    print("  3. Assign labels to windows based on episode analysis")
    print("  4. Reduce to 3 classes: Normal, Pre-Failure, Failure")
    
    print("\nCommand:")
    print("  python label_windows.py")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()