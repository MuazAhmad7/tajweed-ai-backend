#!/usr/bin/env python3
"""
Audio player with visual spectrograms - listen and see what got extracted
"""

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
import subprocess
import time

class AudioSpectrogramPlayer:
    def __init__(self):
        pass
        
    def create_spectrogram_comparison(self, original_file, extracted_file, sample_name, method_name):
        """Create side-by-side spectrogram comparison"""
        
        print(f"🎵 Creating spectrogram for {sample_name} ({method_name})")
        
        # Load both audio files
        try:
            y_orig, sr_orig = librosa.load(original_file, sr=None)
            y_extracted, sr_extracted = librosa.load(extracted_file, sr=None)
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return None
            
        # Create spectrograms
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Original audio waveform
        axes[0, 0].plot(np.linspace(0, len(y_orig)/sr_orig, len(y_orig)), y_orig, color='blue', alpha=0.7)
        axes[0, 0].set_title(f'Original Audio - {sample_name}\nDuration: {len(y_orig)/sr_orig:.3f}s')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Extracted audio waveform
        axes[0, 1].plot(np.linspace(0, len(y_extracted)/sr_extracted, len(y_extracted)), y_extracted, color='red', alpha=0.7)
        axes[0, 1].set_title(f'Extracted Qalqalah - {method_name}\nDuration: {len(y_extracted)/sr_extracted:.3f}s')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Original spectrogram
        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig)), ref=np.max)
        img1 = axes[1, 0].imshow(D_orig, aspect='auto', origin='lower', 
                                extent=[0, len(y_orig)/sr_orig, 0, sr_orig/2])
        axes[1, 0].set_title('Original Spectrogram')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Frequency (Hz)')
        plt.colorbar(img1, ax=axes[1, 0], format='%+2.0f dB')
        
        # Extracted spectrogram
        D_extracted = librosa.amplitude_to_db(np.abs(librosa.stft(y_extracted)), ref=np.max)
        img2 = axes[1, 1].imshow(D_extracted, aspect='auto', origin='lower',
                                extent=[0, len(y_extracted)/sr_extracted, 0, sr_extracted/2])
        axes[1, 1].set_title('Extracted Spectrogram')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Frequency (Hz)')
        plt.colorbar(img2, ax=axes[1, 1], format='%+2.0f dB')
        
        plt.tight_layout()
        
        # Save the comparison
        output_dir = "audio_spectrograms"
        os.makedirs(output_dir, exist_ok=True)
        
        plot_file = os.path.join(output_dir, f"{sample_name}_{method_name}_comparison.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"📊 Saved spectrogram: {plot_file}")
        
        plt.show()
        
        return plot_file
    
    def play_audio_with_spectrogram(self, audio_file, sample_name, method_name):
        """Play audio and show spectrogram"""
        
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return
            
        print(f"🔊 Playing: {sample_name} ({method_name})")
        print(f"📁 File: {audio_file}")
        
        # Load audio for analysis
        y, sr = librosa.load(audio_file, sr=None)
        duration = len(y) / sr
        
        print(f"⏱️ Duration: {duration:.3f}s")
        
        # Create spectrogram
        plt.figure(figsize=(12, 8))
        
        # Waveform
        plt.subplot(2, 1, 1)
        time_axis = np.linspace(0, duration, len(y))
        plt.plot(time_axis, y, color='blue', alpha=0.7)
        plt.title(f'{sample_name} - {method_name}\nDuration: {duration:.3f}s')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Spectrogram
        plt.subplot(2, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = plt.imshow(D, aspect='auto', origin='lower', extent=[0, duration, 0, sr/2])
        plt.title('Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(img, format='%+2.0f dB')
        
        plt.tight_layout()
        plt.show()
        
        # Play the audio
        print("🎵 Playing audio...")
        try:
            subprocess.run(["afplay", audio_file], check=True)
        except subprocess.CalledProcessError:
            print("❌ Could not play audio (afplay not available)")
        except FileNotFoundError:
            print("❌ afplay not found - audio playback not available")
    
    def compare_all_methods(self, sample_name):
        """Compare all detection methods for a sample"""
        
        print(f"🔍 COMPARING ALL METHODS FOR: {sample_name}")
        print("=" * 60)
        
        # Find all versions of this sample
        methods = []
        
        # Original audio
        original_file = f"real_audio_extracted/{sample_name}.wav"
        if os.path.exists(original_file):
            methods.append(("Original", original_file))
        
        # Old method
        old_file = f"real_qalqalah_results/{sample_name}_qalqalah_detected.wav"
        if os.path.exists(old_file):
            methods.append(("Old Method", old_file))
            
        # Refined method
        refined_file = f"accurate_qalqalah_results/{sample_name}_REFINED_qalqalah.wav"
        if os.path.exists(refined_file):
            methods.append(("Refined Method", refined_file))
            
        # Precise method
        precise_file = f"precise_qalqalah_results/{sample_name}_PRECISE_bounce.wav"
        if os.path.exists(precise_file):
            methods.append(("Precise Method", precise_file))
            
        # Debug method
        debug_file = f"visual_debug_results/{sample_name}_DEBUG_detected.wav"
        if os.path.exists(debug_file):
            methods.append(("Debug Method", debug_file))
        
        print(f"📊 Found {len(methods)} versions:")
        for i, (method, file) in enumerate(methods):
            y, sr = librosa.load(file, sr=None)
            duration = len(y) / sr
            print(f"  {i+1}. {method}: {duration:.3f}s")
        
        # Play each one with spectrogram
        for i, (method, file) in enumerate(methods):
            print(f"\n🎵 {i+1}/{len(methods)}: {method}")
            input("Press Enter to play and show spectrogram...")
            self.play_audio_with_spectrogram(file, sample_name, method)
            
            if i < len(methods) - 1:
                print("\n" + "="*40)
        
        # Create comparison spectrograms
        if len(methods) >= 2:
            print(f"\n📊 Creating comparison spectrograms...")
            original_file = methods[0][1]  # First is usually original
            
            for method, file in methods[1:]:
                if file != original_file:
                    self.create_spectrogram_comparison(original_file, file, sample_name, method)

def main():
    """Interactive audio player with spectrograms"""
    
    player = AudioSpectrogramPlayer()
    
    # Find all samples
    samples = []
    audio_dir = "real_audio_extracted"
    
    if os.path.exists(audio_dir):
        for f in os.listdir(audio_dir):
            if f.endswith('.wav'):
                sample_name = f.replace('.wav', '')
                samples.append(sample_name)
    
    if not samples:
        print("❌ No samples found in real_audio_extracted/")
        return
    
    print("🎵 AUDIO SPECTROGRAM PLAYER")
    print("=" * 50)
    print(f"Found {len(samples)} samples:")
    
    for i, sample in enumerate(samples):
        print(f"  {i+1}. {sample}")
    
    print("\nChoose a sample to analyze:")
    try:
        choice = int(input("Enter number (1-{}): ".format(len(samples)))) - 1
        if 0 <= choice < len(samples):
            selected_sample = samples[choice]
            player.compare_all_methods(selected_sample)
        else:
            print("❌ Invalid choice")
    except ValueError:
        print("❌ Please enter a valid number")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
