import cv2
import numpy as np
import argparse
from pathlib import Path
import math

class SpectrogramComparator:
    def __init__(self):
        pass

    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculates Peak Signal-to-Noise Ratio (PSNR).
        """
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100.0
        max_pixel = 65535.0 # 16-bit max
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return psnr

    def generate_comparison(self, dir_original: str, dir_recon: str, output_dir: str):
        path_a = Path(dir_original)
        path_b = Path(dir_recon)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Get all PNG files and SORT them so 0000 aligns with 0000
        files_a = sorted(list(path_a.glob("*.png")))
        files_b = sorted(list(path_b.glob("*.png")))

        if not files_a or not files_b:
            print("Error: One of the input directories is empty.")
            return

        # Handle mismatch in slice counts
        count_a = len(files_a)
        count_b = len(files_b)
        match_count = min(count_a, count_b)
        
        print(f"Found {count_a} originals and {count_b} reconstructed.")
        print(f"Comparing first {match_count} slices by index...")

        psnr_scores = []

        for i in range(match_count):
            f_a = files_a[i]
            f_b = files_b[i]
            
            # Load images as 16-bit
            img1 = cv2.imread(str(f_a), cv2.IMREAD_UNCHANGED)
            img2 = cv2.imread(str(f_b), cv2.IMREAD_UNCHANGED)

            if img1 is None or img2 is None:
                print(f"Error loading files at index {i}")
                continue

            # Check Dimensions (Recon might be slightly different due to resizing math)
            if img1.shape != img2.shape:
                # Resize Recon (img2) to match Original (img1)
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LANCZOS4)

            # 1. Calculate Difference
            # Convert to int32 to safely calculate negative differences, then abs
            img1_32 = img1.astype(np.int32)
            img2_32 = img2.astype(np.int32)
            diff = np.abs(img1_32 - img2_32).astype(np.uint16)

            # 2. Calculate PSNR
            psnr = self.calculate_psnr(img1.astype(np.float64), img2.astype(np.float64))
            psnr_scores.append(psnr)

            # 3. Visualization
            # Amplify difference by 50x so we can see the noise
            diff_amplified = cv2.multiply(diff, 50) 
            
            # Create a separator line
            sep = np.zeros((img1.shape[0], 10, 3), dtype=np.uint16)
            
            combined = np.hstack((img1, sep, img2, sep, diff_amplified))

            # Convert to 8-bit for "Human Viewable" Preview
            preview_canvas = (combined / 256).astype(np.uint8)
            
            # Add Labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Original Label
            cv2.putText(preview_canvas, f"Orig: {f_a.name}", (50, 50), font, 1, (255, 255, 255), 2)
            
            # Recon Label
            col2_start = img1.shape[1] + 10
            cv2.putText(preview_canvas, f"Recon: {f_b.name}", (col2_start + 50, 50), font, 1, (255, 255, 255), 2)
            cv2.putText(preview_canvas, f"PSNR: {psnr:.2f}dB", (col2_start + 50, 100), font, 1, (100, 255, 100), 2)
            
            # Diff Label
            col3_start = col2_start + img1.shape[1] + 10
            cv2.putText(preview_canvas, "Diff (x50)", (col3_start + 50, 50), font, 1, (0, 0, 255), 2)

            # Save
            # Use original index name for clarity
            save_name = out_path / f"compare_slice_{i:04d}.png"
            cv2.imwrite(str(save_name), preview_canvas)

            print(f"Slice {i:02d}: {psnr:.2f} dB")

        # Summary
        if psnr_scores:
            avg_psnr = sum(psnr_scores) / len(psnr_scores)
            print("-" * 30)
            print(f"AVERAGE PSNR: {avg_psnr:.2f} dB")
            print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("original_dir")
    parser.add_argument("reconstructed_dir")
    parser.add_argument("-o", "--output", default="./comparison_results")
    
    args = parser.parse_args()
    
    comp = SpectrogramComparator()
    comp.generate_comparison(args.original_dir, args.reconstructed_dir, args.output)
