import os
import pickle
import random
import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to load trajectory files
def load_trajectories(directory, num_samples=3):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pkl")]
    if len(files) < num_samples:
        print("Not enough trajectory files found. Using all available files.")
        num_samples = len(files)
    
    sampled_files = random.sample(files, num_samples)
    trajectories = []
    
    for file in sampled_files:
        with open(file, "rb") as f:
            trajectory = pickle.load(f)
            trajectories.append(trajectory)
    
    return trajectories

# Function to render and save frames using Matplotlib
def render_frame_with_text(frame, qa_text):
    fig, ax = plt.subplots(figsize=(8, 6))  # Increase figure size for better spacing
    ax.imshow(frame)
    ax.axis("off")
    
    # Adjust text placement to ensure it's within the frame
    plt.figtext(0.5, 0.05, qa_text, wrap=True, horizontalalignment='center', fontsize=16, bbox={"facecolor":"white", "alpha":0.7, "pad":5})
    
    # Save frame as image
    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    
    return img_array

# Function to generate video with imageio
def create_video(trajectories, output_file="trajectory_output_gemini.mp4", frame_rate=2):
    if not trajectories:
        print("No trajectories found. Exiting.")
        return
    
    video_writer = imageio.get_writer(output_file, fps=frame_rate, codec="libx264")

    for traj in trajectories:
        frames = traj['frames']
        qa_pairs = traj.get('qa_pairs', [[]] * len(frames))  # Ensure we have a QA list per frame

        for i in range(len(frames)):
            frame = frames[i]
            qa_list = qa_pairs[i] if i < len(qa_pairs) else []
            
            # Sample two QA pairs if available
            sampled_qa = random.sample(qa_list, min(len(qa_list), 2))
            qa_text = "\n".join([f"Q: {q} | A: {a}" for q, a in sampled_qa])

            frame_with_text = render_frame_with_text(frame, qa_text)
            video_writer.append_data(frame_with_text)

    video_writer.close()
    print(f"Video saved as {output_file}")

# Main function
def main():
    directory = input("Enter the directory containing trajectory .pkl files: ").strip()
    
    if not os.path.exists(directory):
        print("Invalid directory. Exiting.")
        return
    
    trajectories = load_trajectories(directory)
    create_video(trajectories)

if __name__ == "__main__":
    main()
