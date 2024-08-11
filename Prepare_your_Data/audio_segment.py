import pandas as pd
from pydub import AudioSegment
import os

def process_audio_files(metadata_file, audio_dir, output_dir, summary_file):
    # Load the metadata
    metadata = pd.read_csv(metadata_file, delimiter='|')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare a list to store summary data
    summary_data = []
    
    for index, row in metadata.iterrows():
        file_name = row['audio_file']
        start_time = row['SegmentStart'] * 1000  # Convert to milliseconds
        end_time = row['SegmentEnd'] * 1000  # Convert to milliseconds
        text = row['text']
        speaker = row['speaker_name']
        
        # Construct file paths
        audio_file_path = os.path.join(audio_dir, file_name)
        
        # Load the audio file
        try:
            audio = AudioSegment.from_wav(audio_file_path)
            
            # Check if the audio file is longer than 10 seconds
            if len(audio[start_time:end_time]) / 1000 < 15:
                # Segment the audio
                segment = audio[start_time:end_time]
                
                # Generate a unique name for the segment
                segment_num = index + 1
                segment_file_name = f"audio_{segment_num}.wav"
                segment_file_path = os.path.join(output_dir, segment_file_name)
                
                # Export the segmented audio
                segment.export(segment_file_path, format="wav")
                print(f"Exported {segment_file_path}")
                
                # Add to summary data
                summary_data.append([segment_file_name, text, speaker])
            else:
                print(f"Audio file {file_name} is more than 15 seconds long. Skipping.")
            
        except FileNotFoundError:
            print(f"File not found: {audio_file_path}")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    # Save the summary data to a CSV file
    summary_df = pd.DataFrame(summary_data, columns=['audio_file', 'text', 'speaker_name'])
    summary_df.to_csv(summary_file, index=False, sep='|')
    print(f"Summary file saved as {summary_file}")

# Parameters
metadata_file = "D:/Sada_Dataset/Prepare_your_Data/cleaning_audio.csv"  # Replace with your metadata file path
audio_dir = "D:/Sada_Dataset/audio"  # Replace with your audio directory path
output_dir = "D:/python/TTSv24/TTS/xtts-trainer-no-ui-auto/model/wavs"  # Replace with your output directory path
summary_file = "D:/python/TTSv24/TTS/xtts-trainer-no-ui-auto/model/wavs/summary_file.csv"  # Replace with your summary file path

process_audio_files(metadata_file, audio_dir, output_dir, summary_file)
