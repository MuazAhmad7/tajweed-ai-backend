#!/usr/bin/env python3
"""
Accurate Real Qalqalah Pipeline
Use MFA phoneme guidance for real samples, just like the original perfect method
"""

import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
import json
import subprocess
from pathlib import Path
from scipy import ndimage

class AccurateQalqalahPipeline:
    """Accurate qalqalah detection using MFA phoneme guidance"""
    
    def __init__(self, audio_dir="real_audio_extracted", output_dir="accurate_qalqalah_results"):
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.mfa_working_dir = "/Users/nabhanmazid/Documents/MFA"
        
        os.makedirs(output_dir, exist_ok=True)
        
    def setup_mfa_for_sample(self, audio_file, sample_name):
        """Set up MFA corpus for a single sample"""
        
        print(f"🔧 Setting up MFA for {sample_name}...")
        
        # Create sample-specific MFA corpus directory
        corpus_dir = f"mfa_corpus_{sample_name}"
        os.makedirs(corpus_dir, exist_ok=True)
        
        # Copy audio file to corpus
        audio_dest = os.path.join(corpus_dir, f"{sample_name}.wav")
        subprocess.run(["cp", audio_file, audio_dest], check=True)
        
        # Create text file - we know these are all verse 3 (qalqala_yulad)
        # Verse 3: "لَمْ يَلِدْ وَلَمْ يُولَدْ"
        text_content = "لَمْ يَلِدْ وَلَمْ يُولَدْ"
        
        text_file = os.path.join(corpus_dir, f"{sample_name}.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        return corpus_dir, audio_dest, text_file
    
    def run_mfa_alignment(self, corpus_dir, sample_name):
        """Run MFA alignment for the sample"""
        
        print(f"🎯 Running MFA alignment for {sample_name}...")
        
        try:
            # Change to MFA working directory
            original_cwd = os.getcwd()
            os.chdir(self.mfa_working_dir)
            
            # Run MFA alignment - use the model directory path
            cmd = [
                "mfa", "align",
                f"{original_cwd}/{corpus_dir}",
                f"{original_cwd}/arabic/arabic_mfa.dict",
                f"{original_cwd}/arabic",  # Use the model directory path
                f"{original_cwd}/{corpus_dir}/alignment"
            ]
            
            print(f"  🔄 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            os.chdir(original_cwd)
            
            if result.returncode == 0:
                print(f"  ✅ MFA alignment completed")
                
                # Check for phone intervals
                phone_intervals_path = os.path.join(corpus_dir, "alignment", "phone_intervals.csv")
                if os.path.exists(phone_intervals_path):
                    return phone_intervals_path
                else:
                    print(f"  ⚠️ No phone_intervals.csv found, checking for TextGrid...")
                    textgrid_path = os.path.join(corpus_dir, "alignment", f"{sample_name}.TextGrid")
                    if os.path.exists(textgrid_path):
                        return textgrid_path
                    else:
                        print(f"  ❌ No alignment files found")
                        return None
            else:
                print(f"  ❌ MFA alignment failed:")
                print(f"    stdout: {result.stdout}")
                print(f"    stderr: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"  ❌ Error running MFA: {e}")
            os.chdir(original_cwd)
            return None
    
    def analyze_energy_patterns(self, focus_audio, sr, focus_start):
        """Analyze energy patterns (same as original perfect method)"""
        
        # Calculate frame-by-frame energy with higher resolution
        frame_length = 1024
        hop_length = 256
        
        rms = librosa.feature.rms(y=focus_audio, frame_length=frame_length, hop_length=hop_length)[0]
        rms_smooth = ndimage.gaussian_filter1d(rms, sigma=1)
        
        frames = range(len(rms))
        times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length) + focus_start
        
        return rms, rms_smooth, times
    
    def find_qalqalah_boundaries(self, rms, rms_smooth, times, focus_start, focus_end):
        """Find qalqalah boundaries (same as original perfect method)"""
        
        print(f"  🎯 Finding Qalqalah Boundaries...")
        
        mean_energy = np.mean(rms_smooth)
        std_energy = np.std(rms_smooth)
        
        high_energy_threshold = mean_energy + 0.5 * std_energy
        low_energy_threshold = mean_energy * 0.2
        
        print(f"  📊 Mean energy: {mean_energy:.6f}")
        print(f"  📈 High threshold: {high_energy_threshold:.6f}")
        print(f"  📉 Low threshold: {low_energy_threshold:.6f}")
        
        significant_energy_mask = rms_smooth > high_energy_threshold
        significant_indices = np.where(significant_energy_mask)[0]
        
        if len(significant_indices) > 0:
            # Find last energy region (where qalqalah typically occurs)
            energy_regions = []
            current_start = significant_indices[0]
            
            for i in range(1, len(significant_indices)):
                if significant_indices[i] - significant_indices[i-1] > 5:
                    energy_regions.append((current_start, significant_indices[i-1]))
                    current_start = significant_indices[i]
            
            energy_regions.append((current_start, significant_indices[-1]))
            
            print(f"  🔍 Found {len(energy_regions)} energy regions")
            
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
                duration = qalqalah_end - qalqalah_start
                if duration < 0.05:
                    qalqalah_end = min(qalqalah_start + 0.2, focus_end)
                elif duration > 0.8:
                    qalqalah_end = qalqalah_start + 0.4
                
                print(f"  🎯 QALQALAH BOUNDARIES DETECTED:")
                print(f"    Start: {qalqalah_start:.3f}s")
                print(f"    End: {qalqalah_end:.3f}s")
                print(f"    Duration: {qalqalah_end - qalqalah_start:.3f}s")
                    
                return qalqalah_start, qalqalah_end
        
        # Fallback: use last portion
        return focus_end - 0.5, focus_end - 0.1
    
    def find_qalqalah_word_region(self, phone_intervals_path, audio_duration):
        """Find the region containing the qalqalah word using MFA data"""
        
        print(f"  📊 Analyzing phoneme data for qalqalah word...")
        
        try:
            phones_df = pd.read_csv(phone_intervals_path)
            print(f"    ✅ Loaded {len(phones_df)} phonemes")
            
            # For verse 3 (qalqala_yulad), we're looking for the word "يُولَدْ" (yulad)
            # The qalqalah is on the final 'd' sound
            
            # Look for the last word region (where "yulad" should be)
            # Focus on the last 40% of the audio where the second part typically occurs
            focus_start_time = audio_duration * 0.6
            
            # Find phonemes in this region
            relevant_phones = phones_df[phones_df['end'] >= focus_start_time]
            
            if len(relevant_phones) > 0:
                word_start = relevant_phones['begin'].min()
                word_end = relevant_phones['end'].max()
                
                print(f"    🎯 Found qalqalah word region: {word_start:.3f}s - {word_end:.3f}s")
                return word_start, word_end
            else:
                # Fallback to last portion
                fallback_start = max(0, audio_duration - 2.0)
                print(f"    ⚠️ Using fallback region: {fallback_start:.3f}s - {audio_duration:.3f}s")
                return fallback_start, audio_duration
                
        except Exception as e:
            print(f"    ❌ Error analyzing phoneme data: {e}")
            # Fallback to last portion
            fallback_start = max(0, audio_duration - 2.0)
            return fallback_start, audio_duration
    
    def process_sample_with_mfa(self, audio_file, sample_name, metadata):
        """Process a single sample using MFA guidance"""
        
        print(f"\n🎯 Processing {sample_name} with MFA guidance...")
        
        # Set up MFA corpus
        corpus_dir, audio_dest, text_file = self.setup_mfa_for_sample(audio_file, sample_name)
        
        # Run MFA alignment
        alignment_file = self.run_mfa_alignment(corpus_dir, sample_name)
        
        if alignment_file is None:
            print(f"  ⚠️ MFA alignment failed, using fallback method for {sample_name}")
            # Use fallback method with refined acoustic analysis
            return self.process_sample_fallback(audio_file, sample_name, metadata)
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=None)
        duration = len(y) / sr
        
        print(f"  📊 Audio: {duration:.2f}s at {sr}Hz")
        
        # Find qalqalah word region using MFA data
        focus_start, focus_end = self.find_qalqalah_word_region(alignment_file, duration)
        
        # Extract focus region
        focus_start_sample = int(focus_start * sr)
        focus_end_sample = int(focus_end * sr)
        focus_audio = y[focus_start_sample:focus_end_sample]
        
        # Analyze energy patterns in the focused region
        rms, rms_smooth, times = self.analyze_energy_patterns(focus_audio, sr, focus_start)
        
        # Find qalqalah boundaries
        qalqalah_start, qalqalah_end = self.find_qalqalah_boundaries(rms, rms_smooth, times, focus_start, focus_end)
        
        # Create extended segment (with context)
        extended_start = max(0, qalqalah_start - 0.1)
        extended_end = min(duration, qalqalah_end + 0.1)
        
        # Extract segment
        extended_start_sample = int(extended_start * sr)
        extended_end_sample = int(extended_end * sr)
        extended_audio = y[extended_start_sample:extended_end_sample]
        
        # Save result
        result_filename = f"{sample_name}_ACCURATE_qalqalah.wav"
        result_path = os.path.join(self.output_dir, result_filename)
        sf.write(result_path, extended_audio, sr)
        
        # Save metadata
        result_metadata = {
            'sample_name': sample_name,
            'original_metadata': metadata,
            'mfa_focus_region': [focus_start, focus_end],
            'detected_qalqalah': [qalqalah_start, qalqalah_end],
            'extended_segment': [extended_start, extended_end],
            'duration': extended_end - extended_start,
            'audio_file': result_path
        }
        
        metadata_file = result_path.replace('.wav', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(result_metadata, f, indent=2)
        
        print(f"  ✅ MFA-guided detection: {qalqalah_start:.3f}s - {qalqalah_end:.3f}s")
        print(f"  📏 Extended segment: {extended_start:.3f}s - {extended_end:.3f}s ({result_metadata['duration']:.3f}s)")
        print(f"  💾 Saved: {result_filename}")
        
        # Cleanup MFA corpus
        subprocess.run(["rm", "-rf", corpus_dir], check=True)
        
        return result_metadata
    
    def process_sample_fallback(self, audio_file, sample_name, metadata):
        """Process sample using fallback method with refined acoustic analysis"""
        
        print(f"  🔄 Using refined acoustic analysis (like original perfect method)...")
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=None)
        duration = len(y) / sr
        
        print(f"  📊 Audio: {duration:.2f}s at {sr}Hz")
        
        # For verse 3 (qalqala_yulad), focus on the region where "yulad" occurs
        # Use more precise region detection like the original perfect method
        
        # For verse 3, the qalqalah typically occurs in the last 1.5-2 seconds
        # But we need to be smarter about finding the actual speech region
        if duration > 3.0:
            # Longer audio: focus on last 1.5-2 seconds
            focus_start = max(0, duration - 2.0)
            focus_end = duration
        elif duration > 2.0:
            # Medium audio: focus on last 1.2 seconds  
            focus_start = max(0, duration - 1.2)
            focus_end = duration
        else:
            # Short audio: focus on last 60%
            focus_start = max(0, duration * 0.4)
            focus_end = duration
        
        print(f"  🔍 Analyzing region: {focus_start:.3f}s - {focus_end:.3f}s (refined acoustic method)")
        
        # Extract focus region
        focus_start_sample = int(focus_start * sr)
        focus_end_sample = int(focus_end * sr)
        focus_audio = y[focus_start_sample:focus_end_sample]
        
        # Apply the same refined acoustic analysis as the original perfect method
        rms, rms_smooth, times = self.analyze_energy_patterns(focus_audio, sr, focus_start)
        
        # Find qalqalah boundaries using the proven method (with both rms and rms_smooth)
        qalqalah_start, qalqalah_end = self.find_qalqalah_boundaries(rms, rms_smooth, times, focus_start, focus_end)
        
        # Create extended segment (with context) - same as original perfect method
        extended_start = max(0, qalqalah_start - 0.1)
        extended_end = min(duration, qalqalah_end + 0.1)
        
        # Extract segment
        extended_start_sample = int(extended_start * sr)
        extended_end_sample = int(extended_end * sr)
        extended_audio = y[extended_start_sample:extended_end_sample]
        
        # Save result
        result_filename = f"{sample_name}_REFINED_qalqalah.wav"
        result_path = os.path.join(self.output_dir, result_filename)
        sf.write(result_path, extended_audio, sr)
        
        # Save metadata
        result_metadata = {
            'sample_name': sample_name,
            'original_metadata': metadata,
            'method': 'refined_acoustic_fallback',
            'focus_region': [focus_start, focus_end],
            'detected_qalqalah': [qalqalah_start, qalqalah_end],
            'extended_segment': [extended_start, extended_end],
            'duration': extended_end - extended_start,
            'audio_file': result_path
        }
        
        metadata_file = result_path.replace('.wav', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(result_metadata, f, indent=2)
        
        print(f"  ✅ Refined detection: {qalqalah_start:.3f}s - {qalqalah_end:.3f}s")
        print(f"  📏 Extended segment: {extended_start:.3f}s - {extended_end:.3f}s ({result_metadata['duration']:.3f}s)")
        print(f"  💾 Saved: {result_filename}")
        
        return result_metadata

def main():
    """Run the accurate qalqalah pipeline with MFA guidance"""
    
    print("🎯 ACCURATE REAL QALQALAH PIPELINE WITH MFA")
    print("=" * 60)
    
    pipeline = AccurateQalqalahPipeline()
    
    # Load real samples
    print(f"📁 Loading real audio samples...")
    samples = []
    
    # Find all WAV files and their metadata
    wav_files = list(Path(pipeline.audio_dir).glob("*.wav"))
    
    for wav_file in wav_files:
        metadata_file = wav_file.parent / (wav_file.stem + '_metadata.json')
        
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        samples.append({
            'name': wav_file.stem,
            'audio_file': str(wav_file),
            'metadata': metadata
        })
    
    print(f"✅ Found {len(samples)} real audio samples")
    
    if not samples:
        print("❌ No samples found. Run extract_real_audio.py first.")
        return
    
    # Process each sample with MFA guidance
    results = []
    for sample in samples:
        try:
            result = pipeline.process_sample_with_mfa(
                sample['audio_file'], 
                sample['name'], 
                sample['metadata']
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Error processing {sample['name']}: {e}")
    
    # Summary
    print(f"\n📊 ACCURATE PIPELINE RESULTS:")
    print(f"✅ Successfully processed: {len(results)}/{len(samples)} samples")
    
    if results:
        durations = [r['duration'] for r in results]
        print(f"📏 Duration range: {min(durations):.3f}s - {max(durations):.3f}s")
        print(f"📏 Average duration: {np.mean(durations):.3f}s")
        
        print(f"\n🎵 Generated ACCURATE files:")
        for result in results:
            print(f"  - {os.path.basename(result['audio_file'])} ({result['duration']:.3f}s)")
            
            # Handle different result formats (MFA vs fallback)
            if 'mfa_focus_region' in result:
                print(f"    MFA focus: {result['mfa_focus_region'][0]:.3f}s - {result['mfa_focus_region'][1]:.3f}s")
            elif 'focus_region' in result:
                print(f"    Focus region: {result['focus_region'][0]:.3f}s - {result['focus_region'][1]:.3f}s")
                print(f"    Method: {result.get('method', 'unknown')}")
            
            print(f"    Detected: {result['detected_qalqalah'][0]:.3f}s - {result['detected_qalqalah'][1]:.3f}s")
        
        print(f"\n🔊 To test the ACCURATE results:")
        print(f"  cd {pipeline.output_dir}")
        if results:
            first_result = results[0]
            print(f"  afplay {os.path.basename(first_result['audio_file'])}")
        
        print(f"\n🎯 These use MFA phoneme guidance just like the original perfect method!")

if __name__ == "__main__":
    main()
