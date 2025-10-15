#!/usr/bin/env python3
"""
Precise qalqalah detector - focuses on finding just the 'ad' bounce, not the whole 'lad' syllable
"""

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
import soundfile as sf
from pathlib import Path

class PreciseQalqalahDetector:
    def __init__(self, audio_dir="real_audio_extracted"):
        self.audio_dir = audio_dir
        
    def find_qalqalah_bounce(self, focus_audio, sr, focus_start):
        """Find the precise qalqalah bounce - the sharp energy drop characteristic"""
        
        print(f"  🎯 Finding precise qalqalah bounce...")
        
        # Energy analysis with higher resolution for precision
        frame_length = 512  # Smaller frames for better time resolution
        hop_length = 128    # Smaller hop for more precision
        
        rms = librosa.feature.rms(y=focus_audio, frame_length=frame_length, hop_length=hop_length)[0]
        rms_smooth = ndimage.gaussian_filter1d(rms, sigma=0.5)  # Less smoothing for sharper features
        
        frames = range(len(rms))
        times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length) + focus_start
        
        # Calculate energy derivative to find sharp drops
        energy_derivative = np.gradient(rms_smooth)
        
        # Find energy statistics
        mean_energy = np.mean(rms_smooth)
        std_energy = np.std(rms_smooth)
        
        print(f"  📊 Energy stats: mean={mean_energy:.6f}, std={std_energy:.6f}")
        
        # Strategy: Look for the pattern of qalqalah
        # 1. Find regions with decent energy (speech)
        # 2. Look for sharp energy drops (the bounce)
        # 3. Focus on the last part of the focus region
        
        speech_threshold = mean_energy + 0.3 * std_energy  # Lower threshold for speech detection
        
        # Find speech regions
        speech_mask = rms_smooth > speech_threshold
        speech_indices = np.where(speech_mask)[0]
        
        if len(speech_indices) == 0:
            print(f"  ⚠️ No speech detected, using fallback")
            return focus_start + len(focus_audio)/sr - 0.3, focus_start + len(focus_audio)/sr - 0.1
        
        # Focus on the last 60% of the focus region (where qalqalah typically occurs)
        last_60_percent_start_idx = int(len(rms_smooth) * 0.4)
        
        # Look for sharp energy drops in the last part
        energy_drops = []
        for i in range(last_60_percent_start_idx, len(energy_derivative) - 1):
            if energy_derivative[i] < -0.001:  # Sharp negative derivative (energy drop)
                # Check if there's decent energy before the drop
                if i > 5 and rms_smooth[i-3:i].mean() > speech_threshold * 0.7:
                    energy_drops.append((i, energy_derivative[i]))
        
        print(f"  🔍 Found {len(energy_drops)} potential energy drops")
        
        if energy_drops:
            # Sort by how sharp the drop is (most negative derivative)
            energy_drops.sort(key=lambda x: x[1])  # Most negative first
            
            # Take the sharpest drop in the last part
            drop_idx, drop_strength = energy_drops[0]
            
            print(f"  📉 Sharpest drop at index {drop_idx}, strength: {drop_strength:.6f}")
            
            # The qalqalah bounce is around this drop
            # Start a bit before the drop, end shortly after
            bounce_start_idx = max(0, drop_idx - 3)  # Start 3 frames before drop
            bounce_end_idx = min(len(times) - 1, drop_idx + 8)  # End 8 frames after drop
            
            # Convert to time
            bounce_start = times[bounce_start_idx]
            bounce_end = times[bounce_end_idx]
            
            # Ensure reasonable duration (qalqalah bounce is very short: 0.05-0.2s)
            duration = bounce_end - bounce_start
            if duration < 0.05:
                bounce_end = bounce_start + 0.1
            elif duration > 0.25:
                bounce_end = bounce_start + 0.15
                
        else:
            # Fallback: use the very end of speech
            if len(speech_indices) > 0:
                last_speech_idx = speech_indices[-1]
                bounce_start = times[max(0, last_speech_idx - 5)]
                bounce_end = times[min(len(times) - 1, last_speech_idx + 3)]
            else:
                # Ultimate fallback
                bounce_start = focus_start + len(focus_audio)/sr - 0.2
                bounce_end = focus_start + len(focus_audio)/sr - 0.05
        
        print(f"  🎯 PRECISE BOUNCE DETECTED:")
        print(f"    Start: {bounce_start:.3f}s")
        print(f"    End: {bounce_end:.3f}s")
        print(f"    Duration: {bounce_end - bounce_start:.3f}s")
        
        return bounce_start, bounce_end, rms, rms_smooth, times, energy_derivative
    
    def process_sample_precise(self, sample_name):
        """Process a sample with precise qalqalah detection"""
        
        audio_file = os.path.join(self.audio_dir, f"{sample_name}.wav")
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return None
            
        print(f"🎯 PRECISE DETECTION: {sample_name}")
        print("=" * 50)
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=None)
        duration = len(y) / sr
        
        print(f"📊 Audio: {duration:.2f}s at {sr}Hz")
        
        # Focus on the very end where qalqalah occurs
        # For verse 3, qalqalah is typically in the last 0.8-1.2 seconds
        focus_duration = min(1.0, duration * 0.4)  # Focus on last 1s or 40% of audio
        focus_start = duration - focus_duration
        focus_end = duration
        
        print(f"🔍 Precise focus: {focus_start:.3f}s - {focus_end:.3f}s ({focus_duration:.3f}s)")
        
        # Extract focus region
        focus_start_sample = int(focus_start * sr)
        focus_end_sample = int(focus_end * sr)
        focus_audio = y[focus_start_sample:focus_end_sample]
        
        # Find precise qalqalah bounce
        bounce_start, bounce_end, rms, rms_smooth, times, energy_derivative = self.find_qalqalah_bounce(
            focus_audio, sr, focus_start)
        
        # Extract the precise bounce
        bounce_start_sample = int(bounce_start * sr)
        bounce_end_sample = int(bounce_end * sr)
        bounce_audio = y[bounce_start_sample:bounce_end_sample]
        
        # Save the precise bounce
        output_dir = "precise_qalqalah_results"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{sample_name}_PRECISE_bounce.wav")
        sf.write(output_file, bounce_audio, sr)
        
        print(f"💾 Saved precise bounce: {output_file}")
        
        # Create visualization
        self.visualize_precise_detection(
            sample_name, y, sr, duration, focus_start, focus_end, 
            focus_audio, bounce_start, bounce_end,
            rms, rms_smooth, times, energy_derivative
        )
        
        return {
            'sample_name': sample_name,
            'bounce_start': bounce_start,
            'bounce_end': bounce_end,
            'duration': bounce_end - bounce_start,
            'audio_file': output_file
        }
    
    def visualize_precise_detection(self, sample_name, y, sr, duration, focus_start, focus_end, 
                                  focus_audio, bounce_start, bounce_end,
                                  rms, rms_smooth, times, energy_derivative):
        """Create detailed visualization of precise detection"""
        
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Full waveform
        plt.subplot(4, 1, 1)
        time_full = np.linspace(0, duration, len(y))
        plt.plot(time_full, y, alpha=0.7, color='blue')
        plt.axvspan(focus_start, focus_end, alpha=0.3, color='yellow', label='Focus Region')
        plt.axvspan(bounce_start, bounce_end, alpha=0.7, color='red', label='Detected Bounce')
        plt.title(f'{sample_name} - Full Audio with Precise Detection')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Focus region waveform
        plt.subplot(4, 1, 2)
        time_focus = np.linspace(focus_start, focus_end, len(focus_audio))
        plt.plot(time_focus, focus_audio, alpha=0.7, color='blue')
        plt.axvspan(bounce_start, bounce_end, alpha=0.7, color='red', label='Detected Bounce')
        plt.title(f'{sample_name} - Focus Region Detail')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Energy analysis
        plt.subplot(4, 1, 3)
        plt.plot(times, rms, alpha=0.5, color='gray', label='RMS Energy')
        plt.plot(times, rms_smooth, color='red', linewidth=2, label='Smoothed RMS')
        plt.axvspan(bounce_start, bounce_end, alpha=0.7, color='red', label='Detected Bounce')
        plt.title(f'{sample_name} - Energy Analysis')
        plt.xlabel('Time (s)')
        plt.ylabel('RMS Energy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Energy derivative (shows drops)
        plt.subplot(4, 1, 4)
        plt.plot(times, energy_derivative, color='green', linewidth=1, label='Energy Derivative')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=-0.001, color='orange', linestyle='--', label='Drop Threshold')
        plt.axvspan(bounce_start, bounce_end, alpha=0.7, color='red', label='Detected Bounce')
        plt.title(f'{sample_name} - Energy Derivative (Sharp Drops)')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy Change Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = f"precise_qalqalah_results/{sample_name}_PRECISE_analysis.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"📊 Saved precise analysis: {plot_file}")
        
        plt.show()

def main():
    """Test precise detection on all samples"""
    
    detector = PreciseQalqalahDetector()
    
    # Find all samples
    audio_dir = "real_audio_extracted"
    if not os.path.exists(audio_dir):
        print(f"❌ Audio directory not found: {audio_dir}")
        return
    
    samples = []
    for f in os.listdir(audio_dir):
        if f.endswith('.wav'):
            sample_name = f.replace('.wav', '')
            samples.append(sample_name)
    
    print(f"🎯 PRECISE QALQALAH DETECTION")
    print(f"Found {len(samples)} samples")
    print("=" * 50)
    
    results = []
    
    # Process first sample as example
    if samples:
        result = detector.process_sample_precise(samples[0])
        if result:
            results.append(result)
            
            print(f"\n🔊 Test the PRECISE result:")
            print(f"cd precise_qalqalah_results")
            print(f"afplay {samples[0]}_PRECISE_bounce.wav")
            
            print(f"\n📊 Duration: {result['duration']:.3f}s")
            print(f"This should be MUCH shorter and contain only the 'ad' bounce!")

if __name__ == "__main__":
    main()
