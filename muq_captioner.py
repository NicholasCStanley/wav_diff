import torch
import librosa
import numpy as np
import argparse
import requests
import json
import os
from pathlib import Path
from typing import List, Dict, Union
from tinytag import TinyTag

# Try to import MuQ
try:
    from muq import MuQMuLan
except ImportError:
    print("Error: 'muq' library not found. Install: pip install muq")
    exit(1)

class OntologyLoader:
    AUDIOSET_URL = "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"
    
    @staticmethod
    def get_audioset_vocabulary() -> Dict[str, List[str]]:
        vocab = {"instrument": [], "genre": [], "mood": []}
        try:
            # Add headers to prevent 403 Forbidden
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(OntologyLoader.AUDIOSET_URL, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                roots = {"instrument": "/m/04szw", "genre": "/m/064t9"}
                for item in data:
                    name = item.get('name', '')
                    desc = item.get('description', '').lower()
                    id = item.get('id', '')
                    if "instrument" in desc or id == roots['instrument']: vocab["instrument"].append(name)
                    elif "genre" in desc or id == roots['genre']: vocab["genre"].append(name)
                    elif "mood" in desc: vocab["mood"].append(name)
                # Cleanup
                if "Musical instrument" in vocab["instrument"]: vocab["instrument"].remove("Musical instrument")
                if "Music genre" in vocab["genre"]: vocab["genre"].remove("Music genre")
                return vocab
        except:
            pass
        
        # Fallback
        return {
            "instrument": ["Electric Guitar", "Synthesizer", "Bass", "Drums", "Piano", "Strings", "Vocals", "Distortion"],
            "genre": ["Rock", "Electronic", "Pop", "Jazz", "Ambient", "Techno", "Metal", "Hip Hop", "Industrial"],
            "mood": ["Dark", "Happy", "Energetic", "Calm", "Aggressive", "Ethereal"]
        }

    @staticmethod
    def get_magnatagatune_vocabulary() -> List[str]:
        return ["guitar", "techno", "strings", "drums", "rock", "piano", "ambient", "synth", "vocals", "metal", "pop", "bass", "dark", "industrial", "heavy"]

class StandardizedCaptioner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading MuQ-MuLan on {self.device}...")
        self.model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(self.device).eval()
        self.audioset = OntologyLoader.get_audioset_vocabulary()
        self.mtt_tags = OntologyLoader.get_magnatagatune_vocabulary()

    def get_best_tag(self, audio_embeds, text_list: List[str]) -> str:
        if not text_list: return ""
        with torch.no_grad():
            text_embeds = self.model(texts=text_list)
            sim = self.model.calc_similarity(audio_embeds, text_embeds)
            return text_list[torch.argmax(sim).item()]

    def analyze_chunk(self, y: np.ndarray, sr: int, manual_prefix: str = "") -> str:
        # Physics
        if len(y)/sr < 0.5 or np.mean(librosa.feature.rms(y=y)[0]) < 0.002: return "silence"
        
        # BPM
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        bpm = int(tempo[0])

        # MuQ
        wav_tensor = torch.tensor(y).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            audio_embeds = self.model(wavs=wav_tensor)
        
        inst = self.get_best_tag(audio_embeds, self.audioset["instrument"])
        genre = self.get_best_tag(audio_embeds, self.audioset["genre"])
        mood = self.get_best_tag(audio_embeds, self.audioset["mood"])
        desc = self.get_best_tag(audio_embeds, self.mtt_tags)

        caption = f"A {mood} {genre} track featuring {inst}. Characterized as {desc}. {bpm} BPM."
        return f"{manual_prefix}, {caption}" if manual_prefix else caption

    def get_user_input(self):
        print("\n--- Metadata ---")
        artist = input("Artist (Enter to skip): ").strip()
        title = input("Song Title (Enter to skip): ").strip()
        tags = []
        if artist: tags.append(f"style of {artist}")
        if title: tags.append(f"style of {title}")
        return ", ".join(tags)

    def process_directory(self, input_path: Union[str, Path], output_path: Union[str, Path], n_fft: int = 2048, hop_length: int = 512):
        in_p = Path(input_path)
        out_p = Path(output_path)
        out_p.mkdir(parents=True, exist_ok=True)
        
        files = []
        if in_p.is_file(): files.append(in_p)
        elif in_p.is_dir():
            for ext in ['*.mp3', '*.wav', '*.flac']: files.extend(list(in_p.glob(ext)))
        
        if not files:
            print("No audio files found.")
            return

        manual_prefix = self.get_user_input()
        print(f"Captioning {len(files)} files...")
        
        for f in files:
            try:
                y, sr = librosa.load(f, sr=24000)
                y, _ = librosa.effects.trim(y, top_db=20)
                if len(y) == 0: continue
                
                target_duration = ((n_fft // 2 + 1) * hop_length) / 44100
                chunk_samples = int(target_duration * 24000)
                num_chunks = len(y) // chunk_samples
                
                print(f"Processing {f.name}...")
                for i in range(num_chunks):
                    start = i * chunk_samples
                    end = start + chunk_samples
                    chunk = y[start:end]
                    caption = self.analyze_chunk(chunk, sr, manual_prefix)
                    
                    with open(out_p / f"{f.stem}_slice_{i:04d}.txt", "w") as tf:
                        tf.write(caption)
            except Exception as e:
                print(f"Error {f.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default="./training_captions")
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    args = parser.parse_args()
    captioner = StandardizedCaptioner()
    captioner.process_directory(args.input, args.output, args.n_fft, args.hop)
