import io
import numpy as np
from PIL import Image
import imageio
from datasets import load_dataset

# Configuration
REPO_ID = "Aasdfip/habitat_web_pose_train"  # Replace with the actual repository ID
OUTPUT_FILE = "output_episode.mp4"
FPS = 3

dataset = load_dataset(REPO_ID, split="train", streaming=True)
print("Fetching the first episode...")
first_episode = next(iter(dataset))

episode_id = first_episode.get("episode_id", "unknown")
images_data = first_episode.get("images", [])

print(f"Processing Episode ID: {episode_id}")
print(f"Found {len(images_data)} frames. Compiling video...")

writer = imageio.get_writer(OUTPUT_FILE, fps=FPS)
for img_dict in images_data:
    raw_bytes = img_dict["bytiter(dataset)es"]
    img_pil = Image.open(io.BytesIO(raw_bytes))
    frame_array = np.array(img_pil)
    
    writer.append_data(frame_array)

writer.close()
print(f"Video saved to ./{OUTPUT_FILE}")

