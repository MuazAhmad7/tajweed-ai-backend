#!/usr/bin/env python3
"""
Visual debugger for qalqalah detection - shows exactly what the algorithm is doing
"""

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import soundfile as sf
from pathlib import Path

class QalqalahVisualDebugger:
    def __init__(self, audio_dir="real_audio_extracted"):
        self.audio_dir = audio_dir
        
    def debug_sample(self, sample_name):
        """Debug a specific sample with visual analysis"""
        
        audio_file = os.path.join(self.audio_dir, f"{sample_name}.wav")
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return
            
        print(f"🔍 VISUAL DEBUG: {sample_name}")
        print("=" * 60)
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=None)
        duration = len(y) / sr
        
        print(f"📊 Audio: {duration:.2f}s at {sr}Hz")
        
        # Determine focus region (same logic as pipeline)
        if duration > 3.0:
            focus_start = max(0, duration - 2.0)
            focus_end = duration
        elif duration > 2.0:
            focus_start = max(0, duration - 1.2)
            focus_end = duration
        else:
            focus_start = max(0, duration * 0.4)
            focus_end = duration
            
        print(f"🔍 Focus region: {focus_start:.3f}s - {focus_end:.3f}s")
        
        # Extract focus region
        focus_start_sample = int(focus_start * sr)
        focus_end_sample = int(focus_end * sr)
        focus_audio = y[focus_start_sample:focus_end_sample]
        
        # Energy analysis (same as pipeline)
        frame_length = 1024
        hop_length = 256
        
        rms = librosa.feature.rms(y=focus_audio, frame_length=frame_length, hop_length=hop_length)[0]
        rms_smooth = ndimage.gaussian_filter1d(rms, sigma=1)
        
        frames = range(len(rms))
        times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length) + focus_start
        
        # Boundary detection (same as pipeline)
        mean_energy = np.mean(rms_smooth)
        std_energy = np.std(rms_smooth)
        
        high_energy_threshold = mean_energy + 0.5 * std_energy
        low_energy_threshold = mean_energy * 0.2
        
        print(f"📊 Mean energy: {mean_energy:.6f}")
        print(f"📈 High threshold: {high_energy_threshold:.6f}")
        print(f"📉 Low threshold: {low_energy_threshold:.6f}")
        
        # Find energy regions
        significant_energy_mask = rms_smooth > high_energy_threshold
        significant_indices = np.where(significant_energy_mask)[0]
        
        energy_regions = []
        if len(significant_indices) > 0:
            current_start = significant_indices[0]
            
            for i in range(1, len(significant_indices)):
                if significant_indices[i] - significant_indices[i-1] > 5:
                    energy_regions.append((current_start, significant_indices[i-1]))
                    current_start = significant_indices[i]
            
            energy_regions.append((current_start, significant_indices[-1]))
        
        print(f"🔍 Found {len(energy_regions)} energy regions:")
        for i, (start_idx, end_idx) in enumerate(energy_regions):
            start_time = times[start_idx]
            end_time = times[end_idx]
            print(f"  Region {i+1}: {start_time:.3f}s - {end_time:.3f}s ({end_time-start_time:.3f}s)")
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Full audio waveform
        plt.subplot(3, 1, 1)
        time_full = np.linspace(0, duration, len(y))
        plt.plot(time_full, y, alpha=0.7, color='blue')
        plt.axvspan(focus_start, focus_end, alpha=0.3, color='yellow', label='Focus Region')
        plt.title(f'{sample_name} - Full Audio Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Focus region waveform
        plt.subplot(3, 1, 2)
        time_focus = np.linspace(focus_start, focus_end, len(focus_audio))
        plt.plot(time_focus, focus_audio, alpha=0.7, color='blue')
        plt.title(f'{sample_name} - Focus Region Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Energy analysis
        plt.subplot(3, 1, 3)
        plt.plot(times, rms, alpha=0.5, color='gray', label='RMS Energy')
        plt.plot(times, rms_smooth, color='red', linewidth=2, label='Smoothed RMS')
        plt.axhline(y=high_energy_threshold, color='green', linestyle='--', label='High Threshold')
        plt.axhline(y=low_energy_threshold, color='orange', linestyle='--', label='Low Threshold')
        plt.axhline(y=mean_energy, color='purple', linestyle=':', label='Mean Energy')
        
        # Highlight energy regions
        for i, (start_idx, end_idx) in enumerate(energy_regions):
            start_time = times[start_idx]
            end_time = times[end_idx]
            plt.axvspan(start_time, end_time, alpha=0.3, color='red', 
                       label=f'Energy Region {i+1}' if i == 0 else '')
        
        # Show detected qalqalah (last region)
        if energy_regions:
            last_region_start_idx, last_region_end_idx = energy_regions[-1]
            
            # Extend to capture full qalqalah
            extended_end_idx = last_region_end_idx
            for i in range(last_region_end_idx, len(rms_smooth)):
                if rms_smooth[i] < low_energy_threshold:
                    extended_end_idx = i
                    break
                extended_end_idx = i
            
            qalqalah_start = times[last_region_start_idx]
            qalqalah_end = times[min(extended_end_idx, len(times)-1)]
            
            # Validate duration
            duration_detected = qalqalah_end - qalqalah_start
            if duration_detected < 0.05:
                qalqalah_end = min(qalqalah_start + 0.2, focus_end)
            elif duration_detected > 0.8:
                qalqalah_end = qalqalah_start + 0.4
            
            plt.axvspan(qalqalah_start, qalqalah_end, alpha=0.5, color='blue', 
                       label=f'Detected Qalqalah ({qalqalah_end-qalqalah_start:.3f}s)')
            
            print(f"\n🎯 DETECTED QALQALAH:")
            print(f"  Start: {qalqalah_start:.3f}s")
            print(f"  End: {qalqalah_end:.3f}s") 
            print(f"  Duration: {qalqalah_end-qalqalah_start:.3f}s")
            
            # Extract and save the detected segment for listening
            qalqalah_start_sample = int(qalqalah_start * sr)
            qalqalah_end_sample = int(qalqalah_end * sr)
            qalqalah_audio = y[qalqalah_start_sample:qalqalah_end_sample]
            
            debug_dir = "visual_debug_results"
            os.makedirs(debug_dir, exist_ok=True)
            
            debug_file = os.path.join(debug_dir, f"{sample_name}_DEBUG_detected.wav")
            sf.write(debug_file, qalqalah_audio, sr)
            print(f"💾 Saved detected segment: {debug_file}")
        
        plt.title(f'{sample_name} - Energy Analysis & Detection')
        plt.xlabel('Time (s)')
        plt.ylabel('RMS Energy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = f"visual_debug_results/{sample_name}_DEBUG_analysis.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"📊 Saved analysis plot: {plot_file}")
        
        plt.show()
        
        return qalqalah_start, qalqalah_end if energy_regions else (None, None)

def main():
    """Debug all samples visually"""
    
    debugger = QalqalahVisualDebugger()
    
    # Find all real samples
    audio_dir = "real_audio_extracted"
    if not os.path.exists(audio_dir):
        print(f"❌ Audio directory not found: {audio_dir}")
        return
    
    samples = []
    for f in os.listdir(audio_dir):
        if f.endswith('.wav'):
            sample_name = f.replace('.wav', '')
            samples.append(sample_name)
    
    print(f"🔍 Found {len(samples)} samples to debug")
    print("Samples:", samples)
    
    # Debug first sample as example
    if samples:
        print(f"\n🎯 Debugging first sample: {samples[0]}")
        debugger.debug_sample(samples[0])
        
        print(f"\n🔊 To test the detected segment:")
        print(f"cd visual_debug_results")
        print(f"afplay {samples[0]}_DEBUG_detected.wav")
        
        print(f"\n📊 To see all samples, run:")
        for sample in samples:
            print(f"debugger.debug_sample('{sample}')")

if __name__ == "__main__":
    main()
