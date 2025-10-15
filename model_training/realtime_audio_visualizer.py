#!/usr/bin/env python3
"""
Real-time audio visualizer - shows moving cursor through waveform as audio plays
"""

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import soundfile as sf
import threading
import time
import subprocess
from pathlib import Path

class RealtimeAudioVisualizer:
    def __init__(self):
        self.is_playing = False
        self.current_time = 0
        self.audio_duration = 0
        self.play_thread = None
        
    def play_audio_file(self, audio_file):
        """Play audio file in separate thread"""
        try:
            subprocess.run(["afplay", audio_file], check=True)
        except:
            pass
        self.is_playing = False
    
    def visualize_with_playback(self, audio_file, sample_name, method_name, extraction_bounds=None):
        """Visualize audio with real-time playback cursor"""
        
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return
            
        print(f"🎵 Loading: {sample_name} ({method_name})")
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=None)
        self.audio_duration = len(y) / sr
        time_axis = np.linspace(0, self.audio_duration, len(y))
        
        print(f"⏱️ Duration: {self.audio_duration:.3f}s")
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot waveform
        ax1.plot(time_axis, y, color='blue', alpha=0.7, linewidth=0.8)
        ax1.set_title(f'{sample_name} - {method_name}\nDuration: {self.audio_duration:.3f}s')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Highlight extraction bounds if provided
        if extraction_bounds:
            start_time, end_time = extraction_bounds
            ax1.axvspan(start_time, end_time, alpha=0.3, color='red', label=f'Extracted Region ({end_time-start_time:.3f}s)')
            ax1.legend()
        
        # Create playback cursor
        cursor_line = ax1.axvline(x=0, color='red', linewidth=2, label='Playback Position')
        
        # Plot spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = ax2.imshow(D, aspect='auto', origin='lower', extent=[0, self.audio_duration, 0, sr/2])
        ax2.set_title('Spectrogram')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        plt.colorbar(img, ax=ax2, format='%+2.0f dB')
        
        # Create spectrogram cursor
        cursor_line2 = ax2.axvline(x=0, color='red', linewidth=2)
        
        plt.tight_layout()
        
        # Animation function
        def animate(frame):
            if self.is_playing:
                self.current_time = frame * 0.1  # Update every 100ms
                if self.current_time <= self.audio_duration:
                    cursor_line.set_xdata([self.current_time])
                    cursor_line2.set_xdata([self.current_time])
                else:
                    self.is_playing = False
            return cursor_line, cursor_line2
        
        # Start animation
        ani = animation.FuncAnimation(fig, animate, interval=100, blit=False, cache_frame_data=False)
        
        # Show plot
        plt.show(block=False)
        
        # Start audio playback
        print("🎵 Starting playback...")
        self.is_playing = True
        self.current_time = 0
        self.play_thread = threading.Thread(target=self.play_audio_file, args=(audio_file,))
        self.play_thread.start()
        
        # Keep plot open during playback
        try:
            while self.is_playing:
                plt.pause(0.1)
            plt.pause(2)  # Keep open for 2 more seconds after playback
        except KeyboardInterrupt:
            self.is_playing = False
            
        plt.close()
    
    def compare_extractions_with_original(self, sample_name):
        """Compare all extractions with the original, showing extraction bounds"""
        
        print(f"🔍 REALTIME COMPARISON: {sample_name}")
        print("=" * 60)
        
        # Original audio
        original_file = f"real_audio_extracted/{sample_name}.wav"
        if not os.path.exists(original_file):
            print(f"❌ Original file not found: {original_file}")
            return
            
        # Load original to get timing info
        y_orig, sr_orig = librosa.load(original_file, sr=None)
        orig_duration = len(y_orig) / sr_orig
        
        # Find all extraction methods
        extractions = []
        
        # Refined method
        refined_file = f"accurate_qalqalah_results/{sample_name}_REFINED_qalqalah.wav"
        if os.path.exists(refined_file):
            # Load metadata to get extraction bounds
            metadata_file = refined_file.replace('.wav', '_metadata.json')
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    detected = metadata.get('detected_qalqalah', [0, 0])
                    extractions.append(("Refined Method", refined_file, detected))
        
        # Precise method
        precise_file = f"precise_qalqalah_results/{sample_name}_PRECISE_bounce.wav"
        if os.path.exists(precise_file):
            # Estimate bounds (precise method focuses on very end)
            y_precise, sr_precise = librosa.load(precise_file, sr=None)
            precise_duration = len(y_precise) / sr_precise
            # Assume it's from the very end
            precise_start = orig_duration - precise_duration - 0.05
            precise_end = orig_duration - 0.05
            extractions.append(("Precise Method", precise_file, [precise_start, precise_end]))
        
        # Debug method
        debug_file = f"visual_debug_results/{sample_name}_DEBUG_detected.wav"
        if os.path.exists(debug_file):
            # Estimate bounds from debug method
            y_debug, sr_debug = librosa.load(debug_file, sr=None)
            debug_duration = len(y_debug) / sr_debug
            # This was around 1.147s - 1.547s for sample_003
            debug_start = 1.147  # Approximate from previous debug output
            debug_end = debug_start + debug_duration
            extractions.append(("Debug Method", debug_file, [debug_start, debug_end]))
        
        print(f"📊 Found {len(extractions)} extraction methods")
        
        # Show original first
        print(f"\n🎵 1. Original Audio ({orig_duration:.3f}s)")
        input("Press Enter to play original with all extraction regions highlighted...")
        
        # Show original with all extraction bounds
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        time_axis = np.linspace(0, orig_duration, len(y_orig))
        ax1.plot(time_axis, y_orig, color='blue', alpha=0.7, linewidth=0.8)
        ax1.set_title(f'{sample_name} - Original Audio with All Extractions')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Highlight all extraction regions
        colors = ['red', 'green', 'orange', 'purple']
        for i, (method, file, bounds) in enumerate(extractions):
            start_time, end_time = bounds
            color = colors[i % len(colors)]
            ax1.axvspan(start_time, end_time, alpha=0.3, color=color, 
                       label=f'{method} ({end_time-start_time:.3f}s)')
        
        ax1.legend()
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig)), ref=np.max)
        img = ax2.imshow(D, aspect='auto', origin='lower', extent=[0, orig_duration, 0, sr_orig/2])
        ax2.set_title('Original Spectrogram with Extraction Regions')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        
        # Highlight extraction regions on spectrogram too
        for i, (method, file, bounds) in enumerate(extractions):
            start_time, end_time = bounds
            color = colors[i % len(colors)]
            ax2.axvspan(start_time, end_time, alpha=0.3, color=color)
        
        plt.colorbar(img, ax=ax2, format='%+2.0f dB')
        plt.tight_layout()
        plt.show()
        
        # Now play each extraction with realtime visualization
        for i, (method, file, bounds) in enumerate(extractions):
            print(f"\n🎵 {i+2}. {method}")
            print(f"   Extracted from: {bounds[0]:.3f}s - {bounds[1]:.3f}s")
            input("Press Enter to play with realtime visualization...")
            self.visualize_with_playback(file, sample_name, method, bounds)

def show_original_perfect_method_logic():
    """Show the logic from the original perfect method"""
    
    print("🎯 ORIGINAL PERFECT METHOD LOGIC")
    print("=" * 50)
    print("""
The original perfect method that generated the 0.424s segment used:

1. 📍 PRECISE FOCUS REGION:
   - focus_start = 2.5s  (not generic %, but specific timing)
   - focus_end = 4.181s  (end of audio)
   - This targeted the exact region where 'ahad' occurs

2. 🔬 HIGH-RESOLUTION ENERGY ANALYSIS:
   - frame_length = 1024
   - hop_length = 256  
   - Gaussian smoothing (sigma=1)

3. 🎯 SMART BOUNDARY DETECTION:
   - Found energy regions with gaps > 5 frames
   - Took the LAST energy region (where qalqalah occurs)
   - Extended until energy drops below low_energy_threshold
   - Added duration validation (0.05s - 0.8s)

4. 📏 CONTEXT EXTENSION:
   - Added 0.1s before and 0.1s after for natural sound
   - Final result: 3.353s - 3.577s (0.224s core) + 0.1s padding each side

5. 🔑 KEY DIFFERENCE:
   - Used BOTH rms and rms_smooth parameters
   - Looked for the characteristic energy DROP pattern
   - Focused on the LAST significant energy region, not the first

The current methods are missing the precise focus region timing!
""")

def main():
    """Interactive realtime audio visualizer"""
    
    visualizer = RealtimeAudioVisualizer()
    
    # Show original method logic first
    show_original_perfect_method_logic()
    
    # Find samples
    samples = []
    audio_dir = "real_audio_extracted"
    
    if os.path.exists(audio_dir):
        for f in os.listdir(audio_dir):
            if f.endswith('.wav'):
                sample_name = f.replace('.wav', '')
                samples.append(sample_name)
    
    if not samples:
        print("❌ No samples found")
        return
    
    print(f"\n🎵 REALTIME AUDIO VISUALIZER")
    print("=" * 50)
    print(f"Found {len(samples)} samples:")
    
    for i, sample in enumerate(samples):
        print(f"  {i+1}. {sample}")
    
    print("\nChoose a sample to visualize:")
    try:
        choice = int(input("Enter number (1-{}): ".format(len(samples)))) - 1
        if 0 <= choice < len(samples):
            selected_sample = samples[choice]
            visualizer.compare_extractions_with_original(selected_sample)
        else:
            print("❌ Invalid choice")
    except ValueError:
        print("❌ Please enter a valid number")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
