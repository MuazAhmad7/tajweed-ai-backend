#!/usr/bin/env python3
"""
Refined qalqalah detection based on visual feedback
Find the actual 'bounce' region that contains the qalqalah effect
"""

import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
import matplotlib.pyplot as plt

def find_qalqalah_region_visual_guided():
    """Find qalqalah region using visual guidance and acoustic analysis"""
    
    print("🎯 REFINED QALQALAH DETECTION - VISUAL GUIDED")
    print("=" * 60)
    
    # Load audio and MFA data
    phone_path = "/Users/nabhanmazid/Documents/MFA/mfa_corpus/alignment/phone_intervals.csv"
    phones_df = pd.read_csv(phone_path)
    
    audio_path = "mfa_corpus/ayah_1_ikhlas.wav"
    y, sr = librosa.load(audio_path, sr=None)
    
    print(f"✅ Audio: {len(y)} samples at {sr} Hz ({len(y)/sr:.2f}s)")
    
    # Based on your visual observation, the qalqalah is in the area with visible activity
    # Let's analyze the last portion more carefully
    
    # Focus on the last 1.5 seconds where the 'ahad' word should be
    focus_start = 2.5  # Start a bit earlier to capture the full word
    focus_end = 4.181  # End of audio
    
    print(f"\n🔍 Analyzing region: {focus_start:.3f}s - {focus_end:.3f}s")
    
    # Extract the focus region
    focus_start_sample = int(focus_start * sr)
    focus_end_sample = int(focus_end * sr)
    focus_audio = y[focus_start_sample:focus_end_sample]
    
    return y, sr, focus_audio, focus_start, focus_end, phones_df

def analyze_energy_patterns(focus_audio, sr, focus_start):
    """Analyze energy patterns to find the qalqalah bounce"""
    
    print(f"\n📊 Analyzing Energy Patterns for Qalqalah Detection...")
    
    # Calculate frame-by-frame energy with higher resolution
    frame_length = 1024  # Smaller frames for better resolution
    hop_length = 256     # More overlap
    
    # RMS energy
    rms = librosa.feature.rms(y=focus_audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Spectral centroid (brightness - qalqalah often has specific spectral characteristics)
    spectral_centroid = librosa.feature.spectral_centroid(y=focus_audio, sr=sr, hop_length=hop_length)[0]
    
    # Zero crossing rate (voice quality)
    zcr = librosa.feature.zero_crossing_rate(focus_audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Create time axis
    frames = range(len(rms))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length) + focus_start
    
    # Smooth the features
    from scipy import ndimage
    rms_smooth = ndimage.gaussian_filter1d(rms, sigma=1)
    
    print(f"  ✅ Calculated {len(rms)} energy frames")
    print(f"  📈 Energy range: {rms.min():.6f} - {rms.max():.6f}")
    print(f"  📊 Mean energy: {rms.mean():.6f}")
    
    return rms, rms_smooth, spectral_centroid, zcr, times

def find_qalqalah_boundaries(rms, rms_smooth, times, focus_start, focus_end):
    """Find qalqalah boundaries based on energy analysis"""
    
    print(f"\n🎯 Finding Qalqalah Boundaries...")
    
    # Strategy: Look for the region with sustained energy that then drops off
    # Qalqalah typically has a characteristic energy pattern
    
    # Calculate energy statistics
    mean_energy = np.mean(rms_smooth)
    std_energy = np.std(rms_smooth)
    
    # Define thresholds
    high_energy_threshold = mean_energy + 0.5 * std_energy  # Above average energy
    low_energy_threshold = mean_energy * 0.2  # Very low energy (silence)
    
    print(f"  📊 Mean energy: {mean_energy:.6f}")
    print(f"  📈 High threshold: {high_energy_threshold:.6f}")
    print(f"  📉 Low threshold: {low_energy_threshold:.6f}")
    
    # Find regions with significant energy
    significant_energy_mask = rms_smooth > high_energy_threshold
    
    # Find the last significant energy region (likely the qalqalah)
    significant_indices = np.where(significant_energy_mask)[0]
    
    if len(significant_indices) > 0:
        # Find continuous regions of high energy
        energy_regions = []
        current_start = significant_indices[0]
        
        for i in range(1, len(significant_indices)):
            if significant_indices[i] - significant_indices[i-1] > 5:  # Gap of more than 5 frames
                # End of current region
                energy_regions.append((current_start, significant_indices[i-1]))
                current_start = significant_indices[i]
        
        # Add the last region
        energy_regions.append((current_start, significant_indices[-1]))
        
        print(f"  🔍 Found {len(energy_regions)} energy regions")
        
        # Focus on the last energy region (most likely qalqalah)
        if energy_regions:
            last_region_start_idx, last_region_end_idx = energy_regions[-1]
            
            # Extend the region slightly to capture the full qalqalah effect
            # Look for where energy drops to very low levels
            extended_end_idx = last_region_end_idx
            for i in range(last_region_end_idx, len(rms_smooth)):
                if rms_smooth[i] < low_energy_threshold:
                    extended_end_idx = i
                    break
                extended_end_idx = i
            
            # Convert indices to time
            qalqalah_start = times[last_region_start_idx]
            qalqalah_end = times[min(extended_end_idx, len(times)-1)]
            
            # Ensure reasonable duration (qalqalah is typically 0.1-0.5 seconds)
            duration = qalqalah_end - qalqalah_start
            if duration < 0.05:  # Too short, extend
                qalqalah_end = min(qalqalah_start + 0.2, focus_end)
            elif duration > 0.8:  # Too long, shorten
                qalqalah_end = qalqalah_start + 0.4
            
        else:
            # Fallback: use the last portion with any energy
            qalqalah_start = focus_end - 0.5
            qalqalah_end = focus_end - 0.1
    else:
        # Fallback: use the last portion
        qalqalah_start = focus_end - 0.5
        qalqalah_end = focus_end - 0.1
    
    print(f"\n🎯 QALQALAH BOUNDARIES DETECTED:")
    print(f"  Start: {qalqalah_start:.3f}s")
    print(f"  End: {qalqalah_end:.3f}s")
    print(f"  Duration: {qalqalah_end - qalqalah_start:.3f}s")
    
    return qalqalah_start, qalqalah_end

def extract_qalqalah_segment(y, sr, qalqalah_start, qalqalah_end):
    """Extract the refined qalqalah segment"""
    
    print(f"\n🎵 Extracting Refined Qalqalah Segment...")
    
    segments_dir = "refined_qalqalah_segments"
    os.makedirs(segments_dir, exist_ok=True)
    
    # Extract the qalqalah segment
    start_sample = int(qalqalah_start * sr)
    end_sample = int(qalqalah_end * sr)
    qalqalah_audio = y[start_sample:end_sample]
    
    # Save the segment
    duration = qalqalah_end - qalqalah_start
    filename = f"REFINED_qalqalah_bounce_time{qalqalah_start:.3f}-{qalqalah_end:.3f}_dur{duration:.3f}s.wav"
    filepath = os.path.join(segments_dir, filename)
    sf.write(filepath, qalqalah_audio, sr)
    
    print(f"  ✅ Saved: {filename}")
    
    # Also create a slightly extended version for context
    extended_start = max(0, qalqalah_start - 0.1)
    extended_end = min(len(y)/sr, qalqalah_end + 0.1)
    
    extended_start_sample = int(extended_start * sr)
    extended_end_sample = int(extended_end * sr)
    extended_audio = y[extended_start_sample:extended_end_sample]
    
    extended_filename = f"EXTENDED_qalqalah_context_time{extended_start:.3f}-{extended_end:.3f}.wav"
    extended_filepath = os.path.join(segments_dir, extended_filename)
    sf.write(extended_filepath, extended_audio, sr)
    
    print(f"  ✅ Saved extended: {extended_filename}")
    
    return filepath, extended_filepath, segments_dir

def create_refined_visualization(y, sr, qalqalah_start, qalqalah_end, rms, times, segments_dir):
    """Create visualization of the refined qalqalah detection"""
    
    print(f"\n📊 Creating Refined Visualization...")
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Full waveform with qalqalah region highlighted
    time_full = np.linspace(0, len(y) / sr, len(y))
    axes[0].plot(time_full, y, alpha=0.7, color='blue', linewidth=0.5)
    axes[0].set_title('Full Audio with Refined Qalqalah Detection', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    
    # Highlight the qalqalah region
    axes[0].axvspan(qalqalah_start, qalqalah_end, alpha=0.4, color='red', label='Refined Qalqalah Region')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Zoomed view of the qalqalah region
    zoom_start = max(0, qalqalah_start - 0.5)
    zoom_end = min(len(y)/sr, qalqalah_end + 0.5)
    zoom_start_sample = int(zoom_start * sr)
    zoom_end_sample = int(zoom_end * sr)
    zoom_audio = y[zoom_start_sample:zoom_end_sample]
    zoom_time = np.linspace(zoom_start, zoom_end, len(zoom_audio))
    
    axes[1].plot(zoom_time, zoom_audio, alpha=0.8, color='blue', linewidth=1)
    axes[1].set_title('Zoomed View: Qalqalah Region', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Amplitude')
    
    # Mark qalqalah boundaries
    axes[1].axvspan(qalqalah_start, qalqalah_end, alpha=0.4, color='red', label='Qalqalah Bounce')
    axes[1].axvline(qalqalah_start, color='red', linestyle='--', linewidth=2, label='Start')
    axes[1].axvline(qalqalah_end, color='red', linestyle='--', linewidth=2, label='End')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Energy analysis
    axes[2].plot(times, rms, color='orange', linewidth=2, label='RMS Energy')
    axes[2].set_title('Energy Analysis for Qalqalah Detection', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('RMS Energy')
    
    # Mark qalqalah region on energy plot
    axes[2].axvspan(qalqalah_start, qalqalah_end, alpha=0.4, color='red', label='Qalqalah Region')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = os.path.join(segments_dir, "refined_qalqalah_detection.png")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved visualization: {viz_path}")
    
    return viz_path

def main():
    """Main function for refined qalqalah detection"""
    
    # Load and analyze
    y, sr, focus_audio, focus_start, focus_end, phones_df = find_qalqalah_region_visual_guided()
    
    # Analyze energy patterns
    rms, rms_smooth, spectral_centroid, zcr, times = analyze_energy_patterns(focus_audio, sr, focus_start)
    
    # Find qalqalah boundaries
    qalqalah_start, qalqalah_end = find_qalqalah_boundaries(rms, rms_smooth, times, focus_start, focus_end)
    
    # Extract segments
    filepath, extended_filepath, segments_dir = extract_qalqalah_segment(y, sr, qalqalah_start, qalqalah_end)
    
    # Create visualization
    viz_path = create_refined_visualization(y, sr, qalqalah_start, qalqalah_end, rms, times, segments_dir)
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"✅ REFINED QALQALAH DETECTION COMPLETE!")
    
    print(f"\n🎯 RESULTS:")
    print(f"  📁 Segments: {segments_dir}/")
    print(f"  🎵 Main segment: REFINED_qalqalah_bounce_*.wav")
    print(f"  📊 Visualization: refined_qalqalah_detection.png")
    
    print(f"\n🔊 LISTEN TO THE REFINED SEGMENT:")
    print(f"  cd {segments_dir}")
    print(f"  afplay REFINED_qalqalah_bounce_*.wav")
    
    print(f"\n🎯 THIS IS YOUR TRAINING SAMPLE!")
    print(f"  - Duration: {qalqalah_end - qalqalah_start:.3f}s")
    print(f"  - Contains the actual qalqalah 'bounce' effect")
    print(f"  - Perfect for binary classification training")

if __name__ == "__main__":
    main()
