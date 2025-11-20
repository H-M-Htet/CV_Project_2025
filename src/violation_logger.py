"""
CSV Logger for violation records
"""
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import log

class ViolationLogger:
    """
    Log violations to CSV file
    """
    
    def __init__(self, output_path: str):
        """
        Initialize logger
        
        Args:
            output_path: Path to CSV file
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create CSV with headers
        with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Violation_ID',
                'Frame_Number',
                'Timestamp_Sec',
                'Violation_Type',
                'Rider_Confidence',
                'Plate_Detected',
                'Plate_Text',
                'Plate_Confidence',
                'Plate_Crop_Path',
                'Rider_BBox',
                'Plate_BBox'
            ])
        
        self.violation_count = 0
        log.info(f"Violation logger initialized: {self.output_path}")
    
    def log_violation(
        self,
        frame_number: int,
        timestamp: float,
        violation: Dict
    ):
        """
        Log single violation to CSV
        
        Args:
            frame_number: Frame number in video
            timestamp: Timestamp in seconds
            violation: Violation dict from associator
        """
        self.violation_count += 1
        
        # Determine violation type
        class_names = ['No_helmet', 'Wearing_helmet', 'Person_on_Bike']
        violation_type = class_names[violation.get('rider_class', 0)]
        
        # Plate info
        has_plate = 'plate_bbox' in violation
        plate_text = violation.get('plate_text', 'N/A')
        plate_conf = violation.get('plate_conf', 0.0)
        plate_crop = violation.get('plate_crop_path', '')
        
        # Write to CSV
        with open(self.output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.violation_count,
                frame_number,
                f"{timestamp:.2f}",
                violation_type,
                f"{violation['rider_conf']:.2f}",
                'Yes' if has_plate else 'No',
                plate_text,
                f"{plate_conf:.2f}" if has_plate else '',
                plate_crop,
                str(violation['rider_bbox']),
                str(violation.get('plate_bbox', ''))
            ])
        
        log.debug(f"Logged violation #{self.violation_count}")
    
    def log_summary(
        self,
        total_frames: int,
        total_motorcycles: int,
        total_violations: int,
        safe_count: int
    ):
        """
        Log summary statistics
        """
        summary_path = self.output_path.parent / 'summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("="*50 + "\n")
            f.write("VIOLATION DETECTION SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total Frames Processed: {total_frames}\n")
            f.write(f"Total Motorcycles Detected: {total_motorcycles}\n")
            f.write(f"Violations Detected: {total_violations}\n")
            f.write(f"Safe Riders: {safe_count}\n")
            f.write(f"Violation Rate: {total_violations/total_motorcycles*100:.1f}%\n" if total_motorcycles > 0 else "Violation Rate: 0%\n")
            f.write(f"\nDetailed records: {self.output_path.name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        log.info(f"Summary saved: {summary_path}")
