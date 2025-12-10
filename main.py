import sys
import argparse
from pathlib import Path

# --- DIRECT IMPORTS (No try/except masking) ---
print("Initializing Pipeline...")
from audio_converter import AudioProcessor
from muq_captioner import StandardizedCaptioner

class AutomatedCaptioner(StandardizedCaptioner):
    """Headless captioner that uses pre-collected tags."""
    def __init__(self, artist_name, song_title):
        super().__init__()
        self.preset_artist = artist_name
        self.preset_title = song_title

    def get_user_input(self):
        prefixes = []
        if self.preset_artist: prefixes.append(f"style of {self.preset_artist}")
        if self.preset_title: prefixes.append(f"style of {self.preset_title}")
        return ", ".join(prefixes)

def main():
    print("\n" + "="*50)
    print("   AUDIO DATASET GENERATOR")
    print("="*50 + "\n")

    # 1. SETUP
    input_path = input("Input Audio (File/Folder): ").strip().strip("'").strip('"')
    output_dir = input("Output Folder: ").strip().strip("'").strip('"')
    
    print("\n--- Metadata Tags ---")
    artist = input("Artist: ").strip()
    title = input("Song Title: ").strip()
    
    # 2. VALIDATION
    p_in = Path(input_path)
    if not p_in.exists():
        print(f"\nError: '{p_in}' does not exist.")
        return

    # 3. PHASE 1: IMAGES
    print(f"\n[PHASE 1] Generating Spectrograms...")
    processor = AudioProcessor(sample_rate=44100, n_fft=2048, hop_length=512, image_size=1024)
    
    files = []
    if p_in.is_file(): files.append(str(p_in))
    else:
        for ext in ['*.mp3', '*.wav', '*.flac']: files.extend([str(x) for x in p_in.glob(ext)])
            
    for f in files:
        processor.encode_audio_to_images(f, output_dir)

    # 4. PHASE 2: CAPTIONS
    print(f"\n[PHASE 2] Generating Captions...")
    captioner = AutomatedCaptioner(artist, title)
    
    # Process the same input files
    captioner.process_directory(input_path, output_dir, n_fft=2048, hop_length=512)

    print("\n" + "="*50)
    print(f"DONE. Data ready in: {output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
