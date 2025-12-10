import torch
import librosa
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict
from tinytag import TinyTag

# Try to import MuQ
try:
    from muq import MuQMuLan
except ImportError:
    print("Error: 'muq' library not found. Install: pip install muq")
    exit(1)

class DictionaryAttack:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading MuQ-MuLan (The Tagger) on {self.device}...")
        
        self.model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
        self.model = self.model.to(self.device).eval()
        
        # --- THE MEGA VOCABULARY ---
        # Curated from AudioSet, MagnaTagATune, and Discogs top tags.
        self.vocab = {
            "Genre": [
                "Abstract", "Acid", "Acid Jazz", "Acoustic", "Alternative Rock", "Ambient", "Atmospheric", 
                "Avant-garde", "Ballad", "Bass Music", "Big Beat", "Black Metal", "Blues", "Breakbeat", 
                "Breakcore", "Chillout", "Chiptune", "Cinematic", "Classical", "Country", "Cyberpunk", 
                "Dance", "Dark Ambient", "Darkwave", "Death Metal", "Deep House", "Disco", "Doom Metal", 
                "Downbeat", "Downtempo", "Dream Pop", "Drone", "Drum & Bass", "Dub", "Dubstep", "EBM", 
                "Electro", "Electronic", "Experimental", "Folk", "Funk", "Future Bass", "Garage", "Glitch", 
                "Glitch Hop", "Goth", "Grime", "Grindcore", "Grunge", "Hard Rock", "Hardcore", "Heavy Metal", 
                "Hip Hop", "House", "Hyperpop", "IDM", "Indie", "Industrial", "Jazz", "Jungle", "Krautrock", 
                "Leftfield", "Lo-Fi", "Math Rock", "Metal", "Metalcore", "Minimal", "Modern Classical", 
                "Neo-Classical", "New Age", "New Wave", "Noise", "Nu Jazz", "Orchestral", "Pop", "Post-Punk", 
                "Post-Rock", "Progressive House", "Progressive Rock", "Psychedelic", "Psytrance", "Punk", 
                "R&B", "Reggae", "Retro", "Rock", "Shoegaze", "Ska", "Sludge", "Soul", "Soundtrack", 
                "Space Rock", "Speed Metal", "Stoner Rock", "Synth-pop", "Synthwave", "Techno", "Tech House", 
                "Thrash", "Trance", "Trap", "Trip Hop", "Vaporwave", "World"
            ],
            "Instrument": [
                "Accordion", "Acoustic Guitar", "Analog Synthesizer", "Banjo", "Bass Guitar", "Bassoon", 
                "Bongo", "Brass Section", "Cello", "Choir", "Clarinet", "Conga", "Distorted Guitar", 
                "Double Bass", "Drum Machine", "Drums", "Electric Guitar", "Electric Piano", "Female Vocals", 
                "Flute", "Fretless Bass", "Glockenspiel", "Grand Piano", "Groovebox", "Guitar", "Hammond Organ", 
                "Harp", "Harpsichord", "Horn", "Male Vocals", "Mandolin", "Marimba", "Modular Synth", 
                "Oboe", "Orchestra", "Organ", "Percussion", "Piano", "Rhodes", "Sampler", "Saxophone", 
                "Shaker", "Sitar", "Slide Guitar", "Strings", "Synthesizer", "Tabla", "Tambourine", 
                "Trombone", "Trumpet", "Tuba", "Ukulele", "Vibraphone", "Viola", "Violin", "Vocals", 
                "Vocoder", "Wurlitzer", "Xylophone", "808", "909", "303"
            ],
            "Mood": [
                "Aggressive", "Angry", "Anxious", "Atmospheric", "Bittersweet", "Bouncy", "Bright", 
                "Calm", "Chaotic", "Cheerful", "Chill", "Cold", "Complex", "Confident", "Dark", 
                "Depressive", "Determined", "Dramatic", "Dreamy", "Driving", "Eclectic", "Eerie", 
                "Emotional", "Energetic", "Epic", "Ethereal", "Euphoric", "Exciting", "Funky", 
                "Futuristic", "Gentle", "Gloomy", "Groovy", "Happy", "Harsh", "Haunting", "Heavy", 
                "Hopeful", "Hypnotic", "Intense", "Introspective", "Joyful", "Laid-back", "Light", 
                "Lonely", "Lush", "Majestic", "Melancholic", "Mellow", "Melodic", "Minimalist", 
                "Mysterious", "Nostalgic", "Optimistic", "Peaceful", "Playful", "Powerful", "Quirky", 
                "Raw", "Relaxing", "Romantic", "Sad", "Scary", "Sentimental", "Serene", "Sexy", 
                "Soothing", "Sophisticated", "Spacey", "Spiritual", "Spooky", "Strange", "Suspenseful", 
                "Sweet", "Technical", "Tense", "Trippy", "Uplifting", "Warm", "Weird", "Whimsical"
            ],
            "Texture": [
                "Analog", "Arpeggiated", "Bass-Heavy", "Bitcrushed", "Clean", "Compressed", "Detuned", 
                "Digital", "Dissonant", "Distorted", "Dry", "Dynamic", "Echoing", "Electronic", 
                "Filtered", "Glitchy", "Granular", "Gritty", "Harmonic", "High-Fidelity", "Hollow", 
                "Layered", "Lo-Fi", "Loop", "Loud", "Low-Fidelity", "Mechanical", "Metallic", 
                "Monotone", "Muddy", "Muffled", "Multi-layered", "Noisy", "Percussive", "Phased", 
                "Polished", "Polyphonic", "Processed", "Pulsing", "Punchy", "Resonant", "Reverberant", 
                "Rhythmic", "Rough", "Sampled", "Saturated", "Sharp", "Shimmering", "Simple", 
                "Smooth", "Soft", "Sparse", "Stereo", "Stuttering", "Sub-bass", "Swirling", 
                "Syncopated", "Synthetic", "Textured", "Thick", "Thin", "Warm", "Washed-out", "Wet"
            ]
        }
        
        # Pre-encode text embeddings to speed up loop
        print("Pre-encoding vocabulary...")
        self.encoded_vocab = {}
        with torch.no_grad():
            for category, terms in self.vocab.items():
                self.encoded_vocab[category] = {
                    "terms": terms,
                    "embeds": self.model(texts=terms)
                }

    def get_top_tags(self, audio_embeds, category: str, top_k: int = 3) -> List[str]:
        """Returns the top K tags from a specific category."""
        vocab_data = self.encoded_vocab[category]
        text_embeds = vocab_data["embeds"]
        terms = vocab_data["terms"]
        
        # Sim: [1, N_terms]
        sim = self.model.calc_similarity(audio_embeds, text_embeds)
        
        # Get Top K indices
        # We use topk to get scores and indices
        scores, indices = torch.topk(sim, k=top_k)
        
        results = []
        for idx in indices[0]:
            results.append(terms[idx.item()])
            
        return results

    def analyze_chunk(self, y: np.ndarray, sr: int, manual_prefix: str = "") -> str:
        # Physics / Metrics
        duration = len(y) / sr
        if duration < 0.5: return "silence"
        
        rms = librosa.feature.rms(y=y)[0]
        if np.mean(rms) < 0.002: return "silence"
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        bpm = int(tempo[0])

        # MuQ Embeddings
        wav_tensor = torch.tensor(y).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            audio_embeds = self.model(wavs=wav_tensor)
        
        # --- THE ATTACK ---
        # We grab the top tags from each category
        genres = self.get_top_tags(audio_embeds, "Genre", top_k=3)
        moods = self.get_top_tags(audio_embeds, "Mood", top_k=3)
        insts = self.get_top_tags(audio_embeds, "Instrument", top_k=3)
        textures = self.get_top_tags(audio_embeds, "Texture", top_k=3)

        # Flatten and deduplicate
        all_tags = []
        if manual_prefix: all_tags.append(manual_prefix)
        all_tags.extend(genres)
        all_tags.extend(insts)
        all_tags.extend(textures)
        all_tags.extend(moods)
        all_tags.append(f"{bpm} BPM")
        
        # Final Format: "style of Artist, Rock, Hard Rock, Guitar, Distorted, Energetic, 140 BPM"
        return ", ".join(all_tags)

    def get_user_input(self):
        print("\n--- Manual Tag Entry ---")
        print("Leave blank and press Enter to skip.")
        try:
            artist = input("Artist: ").strip()
            title = input("Song Title: ").strip()
        except:
            artist, title = "", ""
        
        prefixes = []
        if artist: prefixes.append(f"style of {artist}")
        if title: prefixes.append(f"style of {title}")
        return ", ".join(prefixes)

    def process_directory(self, input_path, output_path, n_fft=2048, hop=512):
        in_p = Path(input_path)
        out_p = Path(output_path)
        out_p.mkdir(parents=True, exist_ok=True)
        
        files = []
        if in_p.is_file(): files.append(in_p)
        elif in_p.is_dir():
            for ext in ['*.mp3', '*.wav', '*.flac']:
                files.extend(list(in_p.glob(ext)))
        
        if not files: return

        manual_prefix = self.get_user_input()
        print(f"Generating Terse Tags for {len(files)} files...")
        
        for f in files:
            try:
                y, sr = librosa.load(f, sr=24000) 
                y, _ = librosa.effects.trim(y, top_db=20)
                if len(y) == 0: continue

                target_duration = ((n_fft // 2 + 1) * hop) / 44100
                chunk_samples = int(target_duration * 24000)
                num_chunks = len(y) // chunk_samples
                
                print(f"Processing {f.name}...")

                for i in range(num_chunks):
                    start = i * chunk_samples
                    end = start + chunk_samples
                    chunk = y[start:end]
                    
                    caption = self.analyze_chunk(chunk, sr, manual_prefix=manual_prefix)
                    
                    txt_name = out_p / f"{f.stem}_slice_{i:04d}.txt"
                    with open(txt_name, "w") as tf:
                        tf.write(caption)
                    
                    if i == 0: print(f"  Example: {caption}")

            except Exception as e:
                print(f"Error {f.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default="./training_captions")
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    
    args = parser.parse_args()
    
    attacker = DictionaryAttack()
    attacker.process_directory(args.input, args.output, args.n_fft, args.hop)
