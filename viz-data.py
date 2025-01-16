import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import pickle
import os
import re

class TrajectoryVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Trajectory Visualizer")

        # UI Elements
        self.create_widgets()

        # Variables
        self.trajectory_files = []
        self.current_trajectory = None
        self.current_frames = []
        self.current_frame_index = 0
        self.playing = False
        self.search_regex = ""

    def create_widgets(self):
        # Frame for file selection
        self.file_frame = ttk.Frame(self.root)
        self.file_frame.pack(fill=tk.X, padx=10, pady=5)

        self.load_button = ttk.Button(self.file_frame, text="Load Trajectories", command=self.load_trajectories)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.trajectory_list = ttk.Combobox(self.file_frame, state="readonly")
        self.trajectory_list.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.trajectory_list.bind("<<ComboboxSelected>>", self.select_trajectory)

        # Frame for search
        self.search_frame = ttk.Frame(self.root)
        self.search_frame.pack(fill=tk.X, padx=10, pady=5)

        self.search_label = ttk.Label(self.search_frame, text="Search QA (Regex):")
        self.search_label.pack(side=tk.LEFT, padx=5)

        self.search_entry = ttk.Entry(self.search_frame)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.search_entry.bind("<Return>", self.update_search)

        # Frame for video and QA
        self.display_frame = ttk.Frame(self.root)
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Canvas for displaying images
        self.canvas = tk.Canvas(self.display_frame, width=640, height=360, bg="black")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)

        # QA display
        self.qa_frame = ttk.Frame(self.display_frame)
        self.qa_frame.pack(fill=tk.BOTH, expand=True)

        self.qa_label = ttk.Label(self.qa_frame, text="QA Pairs:")
        self.qa_label.pack(anchor="w")

        self.qa_listbox = tk.Listbox(self.qa_frame, height=5)
        self.qa_listbox.pack(fill=tk.BOTH, expand=True)

        # Frame for controls
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)

        self.prev_button = ttk.Button(self.control_frame, text="Previous", command=self.prev_frame)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.play_button = ttk.Button(self.control_frame, text="Play", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=5)

        self.next_button = ttk.Button(self.control_frame, text="Next", command=self.next_frame)
        self.next_button.pack(side=tk.LEFT, padx=5)

    def load_trajectories(self):
        directory = filedialog.askdirectory(title="Select Trajectory Directory")
        if not directory:
            return

        self.trajectory_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pkl")]
        self.trajectory_list["values"] = [os.path.basename(f) for f in self.trajectory_files]
        self.trajectory_list.set("")
        self.current_trajectory = None
        self.current_frames = []
        self.current_frame_index = 0
        self.update_canvas()
        self.qa_listbox.delete(0, tk.END)

    def select_trajectory(self, event):
        selected_index = self.trajectory_list.current()
        if selected_index < 0:
            return

        file_path = self.trajectory_files[selected_index]
        with open(file_path, "rb") as f:
            trajectory = pickle.load(f)

        self.current_trajectory = trajectory
        self.current_frames = [ImageTk.PhotoImage(Image.fromarray(frame)) for frame in trajectory['frames']]
        self.current_frame_index = 0

        self.update_canvas()

    def update_canvas(self):
        self.canvas.delete("all")
        if self.current_frames:
            self.canvas.create_image(320, 180, image=self.current_frames[self.current_frame_index])
        self.update_qa_list()

    def update_qa_list(self):
        self.qa_listbox.delete(0, tk.END)
        if self.current_trajectory and 'qa_pairs' in self.current_trajectory:
            qa_pairs = self.current_trajectory['qa_pairs']
            if 0 <= self.current_frame_index < len(qa_pairs):
                frame_qa_pairs = qa_pairs[self.current_frame_index]
                for question, answer in frame_qa_pairs:
                    if not self.search_regex or re.search(self.search_regex, question):
                        self.qa_listbox.insert(tk.END, f"Q: {question} | A: {answer}")

    def update_search(self, event):
        self.search_regex = self.search_entry.get()
        self.update_qa_list()

    def prev_frame(self):
        if self.current_frames:
            self.current_frame_index = (self.current_frame_index - 1) % len(self.current_frames)
            self.update_canvas()

    def next_frame(self):
        if self.current_frames:
            self.current_frame_index = (self.current_frame_index + 1) % len(self.current_frames)
            self.update_canvas()

    def toggle_play(self):
        self.playing = not self.playing
        self.play_button.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self.play_frames()

    def play_frames(self):
        if self.playing and self.current_frames:
            self.next_frame()
            self.root.after(100, self.play_frames)

if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryVisualizer(root)
    root.mainloop()
