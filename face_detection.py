import face_recognition
import cv2
import os
from datetime import timedelta, datetime
from collections import defaultdict
import numpy as np
from jinja2 import Template
from tqdm import tqdm
import logging
import multiprocessing
from multiprocessing import Pool, Manager, Lock
from functools import partial

# Thresholds for face matching and image quality
FACE_DISTANCE_THRESHOLD = 0.5
MIN_FACE_QUALITY_THRESHOLD = 100  # Minimum quality score for face images

class PersonDetection:
    def __init__(self, face_encoding, face_image, video_file, timestamp, face_quality=None):
        self.face_encoding = face_encoding
        quality = face_quality or self._calculate_face_quality(face_image)
        if quality < MIN_FACE_QUALITY_THRESHOLD:
            raise ValueError("Face image quality below threshold")
        self.face_images = [(face_image, quality)]
        self.appearances = [(video_file, timestamp)]
        
    def _calculate_face_quality(self, face_image):
        # Calculate face quality based on image size and clarity
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()  # Image clarity
        size = face_image.shape[0] * face_image.shape[1]  # Image size
        return variance * size
        
    def add_appearance(self, video_file, timestamp, face_image):
        # Calculate quality for new face image
        quality = self._calculate_face_quality(face_image)
        
        # Only add appearance if quality meets threshold
        if quality >= MIN_FACE_QUALITY_THRESHOLD:
            self.appearances.append((video_file, timestamp))
            
            # Keep up to 5 best quality face images
            self.face_images.append((face_image, quality))
            self.face_images.sort(key=lambda x: x[1], reverse=True)  # Sort by quality
            if len(self.face_images) > 5:
                self.face_images = self.face_images[:5]
            
    @property
    def best_face_image(self):
        return self.face_images[0][0] if self.face_images else None

def create_output_folder():
    # Create timestamp string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"face_detection_results_{timestamp}"
    
    # Create main output folder and faces subfolder
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "detected_faces"), exist_ok=True)
    
    # Set up logging for this run
    log_file = os.path.join(output_folder, "face_detection.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return output_folder

def save_face_image(image, filename, output_folder):
    path = os.path.join(output_folder, "detected_faces", filename)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def process_single_video(args):
    video_file, folder_path, debug, log_lock = args
    local_people = []
    
    logging.info(f"Processing video: {video_file}")
    
    video_path = os.path.join(folder_path, video_file)
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if debug:
        # Limit to first minute (fps * 60 frames)
        total_frames = min(total_frames, int(fps * 60))
        with log_lock:
            logging.info(f"DEBUG MODE: Processing {total_frames} frames")
    
    frame_count = 0
    with tqdm(total=total_frames, desc=f"Frames in {video_file}", leave=False) as pbar:
        while video.isOpened():
            if debug and frame_count >= total_frames:
                break
                
            ret, frame = video.read()
            if not ret:
                break
            
            # Process every 30th frame to improve performance
            if frame_count % 30 == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Find faces in the frame
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if face_locations:
                    with log_lock:
                        logging.debug(f"Found {len(face_locations)} faces in frame {frame_count}")
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    timestamp = timedelta(seconds=frame_count/fps)
                    
                    for face_encoding, face_location in zip(face_encodings, face_locations):
                        # Extract face image with extended bounds
                        top, right, bottom, left = face_location
                        height = bottom - top
                        width = right - left
                        
                        # Calculate new bounds (3x larger)
                        new_top = max(top - height, 0)
                        new_bottom = min(bottom + height, rgb_frame.shape[0])
                        new_left = max(left - width, 0)
                        new_right = min(right + width, rgb_frame.shape[1])
                        
                        face_image = rgb_frame[new_top:new_bottom, new_left:new_right]
                        
                        # Check if this person has been seen before
                        found_match = False
                        best_match_distance = float('inf')
                        best_match_person = None
                        
                        for person in local_people:
                            # Calculate face distance
                            face_distance = face_recognition.face_distance([person.face_encoding], face_encoding)[0]
                            
                            if face_distance < FACE_DISTANCE_THRESHOLD and face_distance < best_match_distance:
                                best_match_distance = face_distance
                                best_match_person = person
                                found_match = True
                        
                        if found_match and best_match_person:
                            best_match_person.add_appearance(video_file, timestamp, face_image)
                            with log_lock:
                                logging.debug(f"Matched existing person at {timestamp} (distance: {best_match_distance:.3f})")
                        else:
                            try:
                                # New person detected
                                person = PersonDetection(face_encoding, face_image, video_file, timestamp)
                                local_people.append(person)
                                with log_lock:
                                    logging.info(f"New person detected at {timestamp} in {video_file}")
                            except ValueError as e:
                                # Skip low quality face detections
                                with log_lock:
                                    logging.debug(f"Skipping low quality face at {timestamp} in {video_file}")
                                continue
            
            frame_count += 1
            pbar.update(1)
    
    video.release()
    with log_lock:
        logging.info(f"Completed processing {video_file}")
    
    return local_people

def merge_people_lists(people_lists):
    merged_people = []
    
    # For each list of people from a video
    for people_list in people_lists:
        # For each person in that list
        for new_person in people_list:
            found_match = False
            best_match_distance = float('inf')
            best_match_person = None
            
            # Check against all merged people
            for existing_person in merged_people:
                # Calculate face distance
                face_distance = face_recognition.face_distance([existing_person.face_encoding], new_person.face_encoding)[0]
                
                if face_distance < FACE_DISTANCE_THRESHOLD and face_distance < best_match_distance:
                    best_match_distance = face_distance
                    best_match_person = existing_person
                    found_match = True
            
            if found_match and best_match_person:
                # Merge appearances and face images
                best_match_person.appearances.extend(new_person.appearances)
                for face_image, quality in new_person.face_images:
                    best_match_person.add_appearance(new_person.appearances[-1][0], 
                                                   new_person.appearances[-1][1], 
                                                   face_image)
            else:
                merged_people.append(new_person)
    
    return merged_people

def process_videos(folder_path, output_folder, debug=False):
    # Get list of video files
    video_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    if debug:
        logging.info("DEBUG MODE: Processing only first minute of first video")
        video_files = video_files[:3]  # Take only first three videos
    
    logging.info(f"Found {len(video_files)} video files to process")
    
    # Create a manager for sharing the logging lock
    manager = Manager()
    log_lock = manager.Lock()
    
    try:
        # Create a pool of workers
        num_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
        logging.info(f"Using {num_workers} parallel workers")
        
        # Process videos in parallel
        with Pool(num_workers) as pool:
            # Prepare arguments for each video
            video_args = [(video_file, folder_path, debug, log_lock) for video_file in video_files]
            
            # Process videos and get results
            people_lists = list(tqdm(
                pool.imap(process_single_video, video_args),
                total=len(video_files),
                desc="Processing videos"
            ))
        
        # Merge results from all videos
        people = merge_people_lists(people_lists)
        logging.info(f"Total unique people detected: {len(people)}")
        
        # Save face images and generate HTML
        generate_report(people, output_folder)
        
    finally:
        # Clean up manager resources
        manager.shutdown()

def generate_report(people, output_folder):
    logging.info("Generating report...")
    
    # Save face images with unique identifiers for each person and image
    for person_idx, person in enumerate(tqdm(people, desc="Saving face images")):
        for image_idx, (face_image, quality) in enumerate(person.face_images):
            filename = f"person_{person_idx}_image_{image_idx}_quality_{quality:.2f}.jpg"
            save_face_image(face_image, filename, output_folder)
    
    # Generate HTML with improved styling and collapsible sections
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Detection Report</title>
        <style>
            body { 
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .person { 
                margin-bottom: 30px;
                border: 1px solid #ddd;
                padding: 20px;
                border-radius: 8px;
            }
            .face-images {
                display: flex;
                gap: 10px;
                margin-bottom: 15px;
                flex-wrap: wrap;
            }
            .face-image { 
                max-width: 200px;
                border-radius: 4px;
            }
            .collapsible {
                background-color: #f1f1f1;
                cursor: pointer;
                padding: 10px;
                width: 100%;
                border: none;
                text-align: left;
                outline: none;
                border-radius: 4px;
            }
            .active, .collapsible:hover {
                background-color: #ddd;
            }
            .content {
                display: none;
                padding: 10px;
                overflow: hidden;
            }
            .appearances-count {
                float: right;
                background-color: #666;
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 0.9em;
            }
        </style>
        <script>
            function toggleContent(element) {
                element.classList.toggle("active");
                var content = element.nextElementSibling;
                if (content.style.display === "block") {
                    content.style.display = "none";
                } else {
                    content.style.display = "block";
                }
            }
        </script>
    </head>
    <body>
        <h1>Face Detection Report</h1>
        {% for person in people %}
        <div class="person">
            <h2>Person {{ loop.index }}</h2>
            <div class="face-images">
                {% set outer_loop = loop %}
                {% for i in range(person.face_images | length) %}
                <img class="face-image" src="detected_faces/person_{{ outer_loop.index0 }}_image_{{ i }}_quality_{{ '%0.2f' | format(person.face_images[i][1]) }}.jpg">
                {% endfor %}
            </div>
            <button class="collapsible" onclick="toggleContent(this)">
                Appearances <span class="appearances-count">{{ person.appearances | length }}</span>
            </button>
            <div class="content">
                <ul>
                {% for video, timestamp in person.appearances %}
                    <li>{{ video }} at {{ timestamp }}</li>
                {% endfor %}
                </ul>
            </div>
        </div>
        {% endfor %}
    </body>
    </html>
    """
    
    template = Template(html_template)
    html_content = template.render(people=people)
    
    report_path = os.path.join(output_folder, "face_detection_report.html")
    with open(report_path, "w") as f:
        f.write(html_content)
    
    logging.info(f"Report generated with {len(people)} people")

if __name__ == "__main__":
    try:
        video_folder = "/Users/kbals/Library/CloudStorage/Dropbox/1 Projects/773520-Bals/8mm films van Bompa Vos"
        output_folder = create_output_folder()
        debug_mode = False  # Set to False for full processing
        
        logging.info(f"Starting face detection on folder: {video_folder}")
        logging.info(f"Results will be saved in: {output_folder}")
        
        process_videos(video_folder, output_folder, debug=debug_mode)
        logging.info("Processing complete")
    finally:
        # Multiprocessing cleanup is handled automatically
        pass 