#!/usr/bin/env python3
"""
Lane Detection Project - Main Runner Script
------------------------------------------
This script provides a simple way to run the lane detection pipeline
on a video file.
"""

import os
import cv2
import numpy as np
from keras.models import load_model
from draw_detected_lanes import road_lines, Lanes

def main():
    """
    Main function to run the lane detection pipeline
    """
    print("Lane Detection Using Fully Convolutional Neural Networks")
    print("-------------------------------------------------------")
    
    # Check if model exists
    model_file = 'converted_CNN_model.h5'
    if not os.path.exists(model_file):
        print(f"Error: Model file '{model_file}' not found.")
        print("Please ensure you have the model file in the current directory.")
        return
    
    # Load the model
    print(f"Loading model from {model_file}...")
    model = load_model(model_file, compile=False)
    print("Model loaded successfully!")
    
    # Input video options
    default_video = 'footage1.mp4'
    available_videos = [f for f in os.listdir('.') if f.endswith('.mp4')]
    
    print("\nAvailable video files:")
    for i, video in enumerate(available_videos):
        print(f"  {i+1}. {video}")
    
    # Select video
    video_path = default_video
    if default_video in available_videos:
        print(f"\nUsing default video: {default_video}")
    else:
        print(f"\nDefault video '{default_video}' not found.")
        if available_videos:
            video_path = available_videos[0]
            print(f"Using {video_path} instead.")
        else:
            print("No video files found. Please add a video file to the project directory.")
            return
    
    # Set up output path
    output_video = 'output_lane_detection.mp4'
    print(f"Output will be saved as: {output_video}")
    
    # Create lanes object for tracking
    lanes = Lanes()
    
    # Process the video
    try:
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        # Process frames
        print(f"\nProcessing video ({width}x{height} at {fps:.2f} fps, {total_frames} frames)...")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with lane detection
            processed_frame = road_lines(frame)
            
            # Write to output video
            writer.write(processed_frame)
            
            # Update progress
            frame_count += 1
            if frame_count % 20 == 0:
                progress = frame_count / total_frames * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Clean up
        cap.release()
        writer.release()
        print(f"\nProcessing complete! Output saved to {output_video}")
    
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main() 