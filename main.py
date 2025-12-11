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


def run_pipeline(input_path: str, output_dir: str, artist: str = "", title: str = ""):
    """Run the spectrogram generation and captioning pipeline."""
    p_in = Path(input_path)
    if not p_in.exists():
        print(f"\nError: '{p_in}' does not exist.")
        return False

    # PHASE 1: IMAGES
    print(f"\n[PHASE 1] Generating Spectrograms...")
    processor = AudioProcessor(sample_rate=44100, n_fft=2048, hop_length=512, image_size=1024)

    files = []
    if p_in.is_file():
        files.append(str(p_in))
    else:
        for ext in ['*.mp3', '*.wav', '*.flac']:
            files.extend([str(x) for x in p_in.glob(ext)])

    if not files:
        print("No audio files found.")
        return False

    for f in files:
        processor.encode_audio_to_images(f, output_dir)

    # PHASE 2: CAPTIONS
    print(f"\n[PHASE 2] Generating Captions...")
    captioner = AutomatedCaptioner(artist, title)
    captioner.process_directory(input_path, output_dir, n_fft=2048, hop_length=512)

    print("\n" + "="*50)
    print(f"DONE. Data ready in: {output_dir}")
    print("="*50)
    return True


def interactive_mode():
    """Run in interactive mode with prompts."""
    print("\n" + "="*50)
    print("   AUDIO DATASET GENERATOR")
    print("="*50 + "\n")

    input_path = input("Input Audio (File/Folder): ").strip().strip("'").strip('"')
    output_dir = input("Output Folder: ").strip().strip("'").strip('"')

    print("\n--- Metadata Tags ---")
    artist = input("Artist: ").strip()
    title = input("Song Title: ").strip()

    run_pipeline(input_path, output_dir, artist, title)


def main():
    parser = argparse.ArgumentParser(
        description="Generate spectrogram images and captions from audio files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python main.py

  Non-interactive mode:
    python main.py -i ./audio -o ./dataset
    python main.py -i song.mp3 -o ./output --artist "Artist Name" --title "Song Title"
        """
    )
    parser.add_argument("-i", "--input", help="Input audio file or folder")
    parser.add_argument("-o", "--output", help="Output folder for spectrograms and captions")
    parser.add_argument("--artist", default="", help="Artist name for caption prefix")
    parser.add_argument("--title", default="", help="Song title for caption prefix")

    args = parser.parse_args()

    # If no input/output provided, run interactive mode
    if args.input is None or args.output is None:
        interactive_mode()
    else:
        run_pipeline(args.input, args.output, args.artist, args.title)


if __name__ == "__main__":
    main()
