#!/usr/bin/env python3
"""
Compare qalqalah detection results to help identify the best approach
"""

import os
import json
from pathlib import Path

def load_metadata(metadata_file):
    """Load metadata from JSON file"""
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return {}

def compare_results():
    """Compare different qalqalah detection results"""
    
    print("🔍 QALQALAH DETECTION COMPARISON")
    print("=" * 60)
    
    # Directories to compare
    old_dir = "real_qalqalah_results"  # Old inaccurate method
    new_dir = "accurate_qalqalah_results"  # New refined method
    
    # Find all samples
    samples = set()
    
    if os.path.exists(old_dir):
        for f in os.listdir(old_dir):
            if f.endswith('.wav'):
                sample_name = f.replace('_qalqalah_detected.wav', '').replace('_REFINED_qalqalah.wav', '')
                samples.add(sample_name)
    
    if os.path.exists(new_dir):
        for f in os.listdir(new_dir):
            if f.endswith('.wav'):
                sample_name = f.replace('_qalqalah_detected.wav', '').replace('_REFINED_qalqalah.wav', '')
                samples.add(sample_name)
    
    print(f"📊 Found {len(samples)} samples to compare\n")
    
    for sample in sorted(samples):
        print(f"🎵 {sample}:")
        
        # Old method results
        old_wav = os.path.join(old_dir, f"{sample}_qalqalah_detected.wav")
        old_meta = os.path.join(old_dir, f"{sample}_qalqalah_detected_metadata.json")
        
        # New method results  
        new_wav = os.path.join(new_dir, f"{sample}_REFINED_qalqalah.wav")
        new_meta = os.path.join(new_dir, f"{sample}_REFINED_qalqalah_metadata.json")
        
        if os.path.exists(old_wav):
            old_metadata = load_metadata(old_meta)
            print(f"  ❌ OLD METHOD:")
            duration = old_metadata.get('duration', 'N/A')
            if isinstance(duration, (int, float)):
                print(f"    Duration: {duration:.3f}s")
            else:
                print(f"    Duration: {duration}")
            
            detected = old_metadata.get('detected_qalqalah', ['N/A', 'N/A'])
            if len(detected) >= 2 and isinstance(detected[0], (int, float)) and isinstance(detected[1], (int, float)):
                print(f"    Detected: {detected[0]:.3f}s - {detected[1]:.3f}s")
            else:
                print(f"    Detected: {detected}")
            print(f"    Method: {old_metadata.get('method', 'unknown')}")
        
        if os.path.exists(new_wav):
            new_metadata = load_metadata(new_meta)
            print(f"  ✅ NEW REFINED METHOD:")
            duration = new_metadata.get('duration', 'N/A')
            if isinstance(duration, (int, float)):
                print(f"    Duration: {duration:.3f}s")
            else:
                print(f"    Duration: {duration}")
                
            detected = new_metadata.get('detected_qalqalah', ['N/A', 'N/A'])
            if len(detected) >= 2 and isinstance(detected[0], (int, float)) and isinstance(detected[1], (int, float)):
                print(f"    Detected: {detected[0]:.3f}s - {detected[1]:.3f}s")
            else:
                print(f"    Detected: {detected}")
            print(f"    Method: {new_metadata.get('method', 'unknown')}")
            
            focus = new_metadata.get('focus_region', ['N/A', 'N/A'])
            if len(focus) >= 2 and isinstance(focus[0], (int, float)) and isinstance(focus[1], (int, float)):
                print(f"    Focus: {focus[0]:.3f}s - {focus[1]:.3f}s")
            else:
                print(f"    Focus: {focus}")
        
        print()
    
    print("🔊 TO TEST THE RESULTS:")
    print("=" * 40)
    print("# Test old method:")
    if os.path.exists(old_dir):
        print(f"cd {old_dir}")
        old_files = [f for f in os.listdir(old_dir) if f.endswith('.wav')]
        if old_files:
            print(f"afplay {old_files[0]}")
    
    print("\n# Test new refined method:")
    if os.path.exists(new_dir):
        print(f"cd {new_dir}")
        new_files = [f for f in os.listdir(new_dir) if f.endswith('.wav')]
        if new_files:
            print(f"afplay {new_files[0]}")
    
    print("\n🎯 COMPARISON SUMMARY:")
    print("=" * 40)
    print("The NEW REFINED method uses:")
    print("✅ Same energy analysis as original perfect method (frame_length=1024, hop_length=256)")
    print("✅ Same boundary detection algorithm with both RMS and RMS_smooth")
    print("✅ Smarter focus regions (last 1.2-2.0s instead of generic last 40%)")
    print("✅ Better debugging to see exactly what's happening")
    print("✅ Same 0.1s context extension as original perfect method")

if __name__ == "__main__":
    compare_results()
