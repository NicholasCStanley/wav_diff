import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa
import soundfile as sf
import argparse
from pathlib import Path
from typing import Union
import io

class Qwen2AudioCaptioner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Qwen2-Audio-7B-Instruct on {self.device}...")
        
        # Qwen2-Audio is natively supported in Transformers 4.45+
        # Your version (4.57.1) is perfect for this.
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            device_map="auto",
            torch_dtype=torch.bfloat16
        ).eval()

    def generate_caption(self, audio_array, sr) -> str:
        """
        Qwen2-Audio pipeline.
        """
        # Qwen2 expects a specific conversation format
        conversation = [
            {'role': 'user', 'content': [
                {'type': 'audio', 'audio_url': 'placeholder_will_be_replaced'},
                {'type': 'text', 'text': 'Describe this audio in detail, focusing on instruments, texture, and mood.'},
            ]},
        ]
        
        # Preprocess using the processor
        # Note: 'audios' expects a list of (array, sr) tuples or file paths
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        inputs = self.processor(
            text=text,
            audios=[audio_array], 
            sampling_rate=sr, 
            return_tensors="pt", 
            padding=True
        )
        inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            
        # Decode (Trim the input prompt from the output)
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response.strip()

    def process_directory(self, input_path: Union[str, Path], output_path: Union[str, Path], n_fft: int = 2048, hop_length: int = 512):
        in_p = Path(input_path)
        out_p = Path(output_path)
        out_p.mkdir(parents=True, exist_ok=True)
        
        files = []
        if in_p.is_file(): files.append(in_p)
        elif in_p.is_dir():
            for ext in ['*.mp3', '*.wav', '*.flac']:
                files.extend(list(in_p.glob(ext)))
        
        if not files:
            print("No audio files found.")
            return

        print(f"Generating Qwen2 Captions for {len(files)} files...")
        
        for f in files:
            print(f"Processing: {f.name}...")
            try:
                # Load Audio (Qwen2 works best at 16k or original SR, but we'll use 16000 as standard for LLMs)
                # However, the processor handles resampling, so we can load at native 44100 
                # to match our image slices logic.
                y, sr = librosa.load(f, sr=44100) 
                
                # Trim Silence (Critical for sync)
                y, _ = librosa.effects.trim(y, top_db=20)
                
                if len(y) == 0: continue

                # Calculate Chunk Size (Sync with Image Generator)
                stft_height = n_fft // 2 + 1
                chunk_samples = stft_height * hop_length
                num_chunks = len(y) // chunk_samples
                
                if num_chunks == 0: continue

                for i in range(num_chunks):
                    start = i * chunk_samples
                    end = start + chunk_samples
                    chunk = y[start:end]
                    
                    # Generate Caption
                    caption = self.generate_caption(chunk, sr)
                    
                    # Save Text
                    txt_name = out_p / f"{f.stem}_slice_{i:04d}.txt"
                    with open(txt_name, "w") as tf:
                        tf.write(caption)
                    
                    print(f"  [{i:03d}] {caption}")

            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", default="./training_captions")
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    
    args = parser.parse_args()
    
    captioner = Qwen2AudioCaptioner()
    captioner.process_directory(args.input, args.output, args.n_fft, args.hop)