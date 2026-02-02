import csv
import re

INPUT_LOG = "BGL.log"         # path to raw log
OUTPUT_CSV = "BGL_structured.csv"

# [cite_start]Severity levels considered anomalous [cite: 130]
ANOMALY_LEVELS = {"ERROR", "FATAL"}

def parse_bgl_line(line):
    """
    Parses a single BGL log line with standardized formatting for Step 2 features.
    """
    line = line.strip()
    if not line:
        return None

    # Split by whitespace
    parts = line.split()

    # BGL standard format usually has 10+ parts
    if len(parts) < 10:
        return None

    try:
        # 1. Standardize Unix Time (ensure it's just the numeric portion)
        # Some BGL variants have '-' or other chars in the timestamp field
        unix_time = parts[1]
        
        # 2. Node Identification (e.g., R02-M1-N0-C:J12-U11)
        node = parts[3]
        
        # [cite_start]3. Component Extraction (KERNEL, APP, MMCS) [cite: 196]
        component = parts[7].upper()
        
        # 4. Severity Level Standardizing (Critical for Step 2 SEVERITY_MAP)
        # Converts all to uppercase to prevent mapping misses
        level = parts[8].upper()
        
        # 5. Content Extraction
        content = " ".join(parts[9:])

        # [cite_start]6. Initial Labeling (Direct anomaly detection) [cite: 130]
        label = 1 if level in ANOMALY_LEVELS else 0

        return {
            "Time": unix_time,
            "Node": node,
            "Component": component,
            "Level": level,
            "Content": content,
            "Label": label
        }

    except Exception:
        return None

def parse_bgl_log(input_path, output_path):
    # Fieldnames must match the expectations of 'enhanced_window_creation.py'
    fieldnames = ["Time", "Node", "Component", "Level", "Content", "Label"]
    
    with open(input_path, "r", errors="ignore") as infile, \
         open(output_path, "w", newline="", encoding="utf-8") as outfile:

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        total = 0
        parsed = 0

        for line in infile:
            total += 1
            parsed_line = parse_bgl_line(line)
            if parsed_line:
                writer.writerow(parsed_line)
                parsed += 1

            if total % 1_000_000 == 0:
                print(f"Processed {total:,} lines...")

        print(f"\nParsing Complete.")
        print(f"Total lines read: {total:,}")
        print(f"Successfully structured: {parsed:,}")

if __name__ == "__main__":
    parse_bgl_log(INPUT_LOG, OUTPUT_CSV)