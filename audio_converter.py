import librosa
import numpy as np
import soundfile as sf
import cv2
from pathlib import Path
from typing import List
import argparse
import sys

class AudioProcessor:
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        image_size: int = 1024  # <--- CHANGED DEFAULT TO 1024
    ):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.image_size = image_size
        self.stft_height = self.n_fft // 2 + 1 # 1025 bins
        self.max_ref = 1000.0

    def encode_audio_to_images(self, audio_path: str, output_dir: str) -> List[str]:
        path = Path(audio_path)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Encoding {self.image_size}x{self.image_size} (16-bit): {path.name}...")
        
        try:
            # 1. Load & Trim Silence
            y, _ = librosa.load(path, sr=self.sr)
            y, _ = librosa.effects.trim(y, top_db=20)
            if len(y) < self.n_fft: return []
        except Exception as e:
            print(f"Error: {e}")
            return []
        
        # 2. STFT
        # center=False to avoid padding artifacts
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, center=False)
        
        # 3. Complex -> 16-bit RGB
        rgb_tensor = self._complex_to_rgb_16bit(D)
        
        # 4. Slice & Save
        saved_files = []
        height, total_width, _ = rgb_tensor.shape
        
        # Calculate how many square chunks fit
        # Since we resize the final image, we slice based on the STFT aspect ratio 
        # to ensure the final image isn't squashed.
        # Ideally, we slice a chunk of width = height (1025px) to keep 1:1 aspect.
        
        chunk_width = height # 1025 cols
        num_chunks = total_width // chunk_width 
        
        if num_chunks == 0: return []
        
        for i in range(num_chunks):
            start = i * chunk_width
            end = start + chunk_width
            chunk = rgb_tensor[:, start:end, :]
            
            # Resize to Target Resolution (e.g. 1024x1024)
            # Lanczos4 is critical here for preserving the phase lines
            chunk_resized = cv2.resize(chunk, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)
            
            fname = out_path / f"{path.stem}_slice_{i:04d}.png"
            
            # Save as 16-bit PNG (Swap RGB->BGR for OpenCV)
            chunk_bgr = cv2.cvtColor(chunk_resized, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(fname), chunk_bgr)
            
            saved_files.append(str(fname))
            
        print(f"-> Generated {len(saved_files)} images.")
        return saved_files

    def decode_images_to_audio(self, image_paths: List[str], output_filename: str) -> None:
        print(f"Decoding {len(image_paths)} images...")
        image_paths.sort()
        complex_chunks = []
        
        for p in image_paths:
            try:
                # Load 16-bit
                img_bgr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if img_bgr is None: continue
                
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Resize back to STFT height (e.g. 1024 -> 1025)
                # We resize both width and height to match the STFT bins
                img_upscaled = cv2.resize(img_rgb, (self.stft_height, self.stft_height), interpolation=cv2.INTER_LANCZOS4)
                
                complex_chunk = self._rgb_to_complex_16bit(img_upscaled)
                complex_chunks.append(complex_chunk)
            except Exception as e:
                print(f"Skipping {p}: {e}")

        if not complex_chunks: return

        full_spectrogram = np.concatenate(complex_chunks, axis=1)
        y_recon = librosa.istft(full_spectrogram, hop_length=self.hop_length)
        
        # Normalize to avoid clipping
        #if np.max(np.abs(y_recon)) > 0:
        #    y_recon = librosa.util.normalize(y_recon)
            
        sf.write(output_filename, y_recon, self.sr)
        print(f"-> Audio saved: {output_filename}")

    # --- 16-BIT MATH HELPERS (Same as before) ---
    def _complex_to_rgb_16bit(self, D: np.ndarray) -> np.ndarray:
        mag = np.abs(D)
        # Fixed Reference: 10000.0 (Global Max)
        mag_db = librosa.amplitude_to_db(mag, ref=self.max_ref)
        
        # -100dB floor
        c1 = np.clip((mag_db + 100) / 100, 0, 1) * 65535
        
        angle = np.angle(D)
        c2 = ((np.cos(angle) + 1) / 2) * 65535
        c3 = ((np.sin(angle) + 1) / 2) * 65535
        
        img = np.dstack((c1, c2, c3)).astype(np.uint16)
        return np.flipud(img)

    def _rgb_to_complex_16bit(self, rgb_img: np.ndarray) -> np.ndarray:
        norm = rgb_img.astype(np.float32) / 65535.0
        norm = np.flipud(norm)
        
        r, g, b = norm[:,:,0], norm[:,:,1], norm[:,:,2]
        
        mag_db = (r * 100) - 100
        magnitude = librosa.db_to_amplitude(mag_db, ref=self.max_ref)
        
        # Reconstruct phase unit vector
        phase_complex = ((g * 2) - 1) + 1j * ((b * 2) - 1)
        phase_complex = phase_complex / (np.abs(phase_complex) + 1e-6)
        
        return magnitude * phase_complex

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input file or folder")
    parser.add_argument("-o", "--output", default="./output", help="Output location")
    parser.add_argument("--decode", action="store_true", help="Convert Images -> Audio")
    
    # NEW ARGUMENT: Allow user to override size
    parser.add_argument("--size", type=int, default=1024, help="Image resolution (default 1024)")
    
    args = parser.parse_args()
    
    # Pass the custom size to the processor
    proc = AudioProcessor(image_size=args.size)
    
    # Logic is identical to previous script
    if args.decode:
        import glob
        input_p = Path(args.input)
        if input_p.is_dir():
            files = sorted(glob.glob(str(input_p / "*.png")))
            proc.decode_images_to_audio(files, args.output)
        else:
             print("Error: For decoding, input must be a directory of PNGs.")
    else:
        import glob
        files = []
        inp = Path(args.input)
        if inp.is_file(): files.append(str(inp))
        elif inp.is_dir():
            files = glob.glob(str(inp / "*.mp3")) + glob.glob(str(inp / "*.wav"))
            
        if not files:
            print("No audio files found.")
        else:
            for f in files: proc.encode_audio_to_images(f, args.output)