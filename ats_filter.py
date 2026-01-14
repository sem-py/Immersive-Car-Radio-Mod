import os
import numpy as np
from pydub import AudioSegment
from scipy.signal import fftconvolve, butter, lfilter

def process_all_music(ir_file, pan=0.35, volume_boost=0.0):
    
    output_folder = "ATS_Radio_Library"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(f"Loading interior acoustics: {ir_file}")
    ir = AudioSegment.from_file(ir_file).set_frame_rate(44100).set_channels(1)
    ir_data = np.array(ir.get_array_of_samples(), dtype=np.float32)

    files = [f for f in os.listdir('.') if f.endswith('.mp3') and f != "Modern_ATS_Radio.mp3"]
    
    print(f"Found {len(files)} songs to process. Starting...")

    for filename in files:
        try:
            print(f"\nProcessing: {filename}")
            
            song = AudioSegment.from_file(filename).set_frame_rate(44100).set_channels(1)
            song_data = np.array(song.get_array_of_samples(), dtype=np.float32)

            nyq = 0.5 * 44100

            b, a = butter(2, [60 / nyq, 15000 / nyq], btype='band')
            filtered_song = lfilter(b, a, song_data)

            b_bass, a_bass = butter(1, 180 / nyq, btype='low')
            bass_boost = lfilter(b_bass, a_bass, filtered_song) * 2.0 
            final_filtered = filtered_song + bass_boost

            processed_audio = fftconvolve(final_filtered, ir_data, mode='full')
            processed_audio = processed_audio / np.max(np.abs(processed_audio))

            left_channel = processed_audio * (1.0 - pan)
            right_channel = processed_audio * 1.0 
            stereo_audio = np.vstack((left_channel, right_channel)).T
            
            stereo_audio = np.int16(stereo_audio * 32767)
            final_output = AudioSegment(stereo_audio.tobytes(), frame_rate=44100, sample_width=2, channels=2)
            final_output = final_output + volume_boost
            
            output_path = os.path.join(output_folder, f"ATS_{filename}")
            final_output.export(output_path, format="mp3")
            print(f"Saved to: {output_path}")

        except Exception as e:
            print(f"Could not process {filename}: {e}")

    print("\n--- All songs finished! Check the 'ATS_Radio_Library' folder. ---")

process_all_music("truck_cab.wav")