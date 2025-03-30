import cv2
import asyncio
import base64
import time
import datetime
import numpy as np
from pathlib import Path
import collections
from typing import Set, Optional, Deque, List, Tuple, Dict, Any
import threading
import queue
import logging
import json
import os
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CameraManager")

EVENT_DIR = Path('./events')

# Create events directory if it doesn't exist
EVENT_DIR.mkdir(exist_ok=True, parents=True)

# Define the time window for recording (seconds)
RECORDING_WINDOW_BEFORE = 15.0
RECORDING_WINDOW_AFTER = 15.0
RECORDING_DELAY = 15.0  # Delay before starting the recording job
BUFFER_DURATION = RECORDING_WINDOW_BEFORE + RECORDING_WINDOW_AFTER + 5.0 # Keep a bit extra buffer
RECORDING_COOLDOWN = 2.0 # Minimum seconds between triggering recordings
LONG_PRESENCE_THRESHOLD = 15.0 # Seconds of continuous presence to override cooldown

# --- Add Type Alias ---
# Define a type hint for detection data stored in the buffer
# Assuming format: (confidence, class_id, bbox_tuple)
DetectionInfo = Tuple[float, int, Tuple[int, int, int, int]]

class CameraManager:
    """
    Manages local camera operations: setup, frame processing, event recording trigger.
    Puts processed frames onto a queue for consumption elsewhere.
    """
    def __init__(self):
        self.is_running: bool = False
        self.frame_queue = queue.Queue(maxsize=5) # Queue for base64 encoded frames
        self.device: Optional[AiCamera] = None
        self.model = None
        self.annotator = None
        self.processing_thread = None
        self.loop = None

        # Frame history stores raw frames+detections for recording events
        self.frame_history: Deque[Tuple[np.ndarray, float, List[DetectionInfo]]] = collections.deque(maxlen=1000)

        # Video settings
        self.target_fps = 25.0 # Target FPS for recorded video

        # Keep track of scheduled recording tasks to avoid duplicates if needed, though the requirement implies separate clips
        self.active_recording_tasks: Set[asyncio.Task] = set()
        
        # Cooldown tracking
        self.last_recording_trigger_time: float = 0.0 # Time the last recording job was initiated

        # Continuous presence tracking
        self.person_continuously_detected_since: float = 0.0 # Timestamp when current continuous detection started, 0 if none

        self.local_camera_id = os.environ.get("CAMERA_ID", "local_cam") # Get local cam ID

        logger.info(f"CameraManager initialized for local camera ID: {self.local_camera_id}")

    def _setup_camera(self):
        """Initialize the camera and ML model"""
        try:
            logger.info("Setting up camera and model...")
            self.device = AiCamera()
            # Infer framerate for buffer calculation if possible (optional)
            # self.target_fps = self.device.get_property(cv2.CAP_PROP_FPS) or 25.0
            # self.frame_history = collections.deque(maxlen=int(self.target_fps * BUFFER_DURATION))
            
            self.model = SSDMobileNetV2FPNLite320x320()
            self.device.deploy(self.model)
            self.annotator = Annotator(thickness=1, text_thickness=1, text_scale=0.4)
            logger.info(f"Camera and model setup complete. Frame buffer size: {self.frame_history.maxlen}")
            return True
        except Exception as e:
            logger.error(f"Error setting up camera: {str(e)}")
            return False

    def start(self):
        """Start the camera processing and related tasks"""
        if self.is_running:
            logger.info("Camera is already running")
            return
        if not self.device and not self._setup_camera():
            logger.error("Failed to start camera: setup failed")
            return
            
        logger.info("Starting camera manager...")
        self.is_running = True
        self.loop = asyncio.get_running_loop()

        self.processing_thread = threading.Thread(target=self._process_frames_thread, daemon=True)
        self.processing_thread.start()

        logger.info("Camera manager started successfully (frame processing only)")

    def _get_frame_buffer_duration(self):
        """Calculate the current duration covered by the frame buffer in seconds"""
        if len(self.frame_history) < 2:
            return 0
        oldest_time = self.frame_history[0][1]
        newest_time = self.frame_history[-1][1]
        return newest_time - oldest_time

    def _process_frames_thread(self):
        """
        Processes frames, detects, annotates, stores history for recording,
        and puts encoded+annotated frames onto the frame_queue for viewers.
        """
        logger.info("Frame processing thread started")
        last_log_time = 0

        try:
            with self.device as stream:
                for frame_obj in stream:
                    if not self.is_running:
                        logger.info("Stopping frame processing as is_running=False")
                        break

                    try:
                        current_time = time.time()
                        frame_image = frame_obj.image

                        # Log buffer stats occasionally (can be done before detection)
                        if current_time - last_log_time > 60: # Log every minute
                            buffer_duration = self._get_frame_buffer_duration()
                            logger.info(f"Frame buffer stats: {len(self.frame_history)} frames covering {buffer_duration:.2f} seconds")
                            last_log_time = current_time

                        # --- Detection ---
                        # Get the filtered Detections object
                        filtered_detections_obj = frame_obj.detections[frame_obj.detections.confidence > 0.55]

                        # --- Edit 2: Convert detections for storage ---
                        current_detections_list: List[DetectionInfo] = [
                            (float(score), int(class_id), tuple(map(int, bbox)))
                            for bbox, score, class_id, _ in filtered_detections_obj
                        ]

                        # --- Annotation ---
                        # Prepare labels based on the filtered detections
                        labels = [f"{self.model.labels[class_id]}: {score:.2f}"
                                  for _, score, class_id, _ in filtered_detections_obj]
                        # Annotate the frame IN PLACE (modifies frame_obj.image, which is frame_image)
                        self.annotator.annotate_boxes(frame_obj, filtered_detections_obj, labels=labels)

                        # --- Edit 3: Store Annotated Frame and Detections ---
                        # Now frame_image (which is frame_obj.image) is annotated.
                        # Make a copy before storing.
                        annotated_frame_copy = frame_image.copy()
                        # Store the annotated frame, timestamp, and the detection list
                        self.frame_history.append((annotated_frame_copy, current_time, current_detections_list))

                        # --- Person Detection & Recording Trigger ---
                        person_currently_detected = any(
                            self.model.labels[class_id] == 'person' and score > 0.6
                            for score, class_id, _ in current_detections_list # Use the list we created
                        )

                        # --- Update Continuous Detection State & Trigger Logic ---
                        # ... (This logic remains the same as the previous version) ...
                        if person_currently_detected:
                            if self.person_continuously_detected_since == 0.0:
                                self.person_continuously_detected_since = current_time
                                logger.debug(f"Person continuous detection started at {current_time:.2f}")
                        else:
                            if self.person_continuously_detected_since != 0.0:
                                logger.debug(f"Person continuous detection ended.")
                            self.person_continuously_detected_since = 0.0

                        should_trigger = False
                        trigger_reason = ""
                        if person_currently_detected:
                            cooldown_passed = (current_time - self.last_recording_trigger_time >= RECORDING_COOLDOWN)
                            long_presence_override = False
                            if self.person_continuously_detected_since > 0.0:
                                continuous_duration = current_time - self.person_continuously_detected_since
                                if continuous_duration >= LONG_PRESENCE_THRESHOLD:
                                    if current_time - self.last_recording_trigger_time >= LONG_PRESENCE_THRESHOLD:
                                        long_presence_override = True
                                        trigger_reason = "long presence override"

                            if cooldown_passed and not long_presence_override:
                                should_trigger = True
                                trigger_reason = "cooldown passed"
                            elif long_presence_override:
                                 should_trigger = True

                        if should_trigger:
                            logger.info(f"Person detected at {current_time:.2f}. Triggering recording ({trigger_reason}).")
                            self.last_recording_trigger_time = current_time
                            self.person_continuously_detected_since = current_time # Reset continuous start on trigger

                            # Pass the shape of the annotated frame
                            if self.loop:
                                asyncio.run_coroutine_threadsafe(
                                    self._schedule_recording(current_time, annotated_frame_copy.shape), # Pass shape from the copy
                                    self.loop
                                )
                            else:
                                logger.warning("Asyncio loop not available, cannot schedule recording.")
                        elif person_currently_detected:
                             logger.debug(f"Person detected at {current_time:.2f}, but within cooldown and below long presence threshold. Ignoring trigger.")

                        # --- Edit 5: Frame Encoding for Viewer Queue ---
                        # Encode the *annotated* frame for the viewer queue
                        _, buffer = cv2.imencode('.jpg', frame_image) # frame_image is already annotated
                        encoded_image_str = base64.b64encode(buffer).decode('utf-8')

                        # Put base64 encoded frame in queue for distribution by main app
                        try:
                            # Clear queue if full to always have latest frames
                            while self.frame_queue.qsize() >= self.frame_queue.maxsize:
                                 self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(encoded_image_str)
                        except queue.Full:
                             logger.warning("Frame queue was unexpectedly full after clearing attempt.")
                        except Exception as e:
                             logger.error(f"Error putting frame onto queue: {e}")

                    except Exception as e:
                        logger.exception(f"Error processing frame: {str(e)}")
                        time.sleep(0.1)

        except Exception as e:
            logger.exception(f"Fatal error in frame processing thread: {str(e)}")
        finally:
            logger.info("Frame processing thread stopped")
            self.is_running = False

    async def _schedule_recording(self, detection_time: float, frame_shape: Tuple[int, int, int]):
        """Schedules the _record_event coroutine to run after a delay."""
        task = asyncio.create_task(self._delayed_record_event(detection_time, frame_shape))
        self.active_recording_tasks.add(task)
        task.add_done_callback(self.active_recording_tasks.discard)
        logger.info(f"Recording task scheduled for detection at {detection_time:.2f}, will run in {RECORDING_DELAY}s")

    async def _delayed_record_event(self, detection_time: float, frame_shape: Tuple[int, int, int]):
        """Waits for the delay, then prepares data and triggers the actual recording via shared function."""
        await asyncio.sleep(RECORDING_DELAY) # Use constant from this file
        logger.info(f"[{self.local_camera_id}] Preparing recording job for detection at {detection_time:.2f}")

        # --- Data Preparation Logic (Adapted from old _record_event) ---
        start_time = detection_time - RECORDING_WINDOW_BEFORE
        end_time = detection_time + RECORDING_WINDOW_AFTER
        # Calculate target count based on constants in this file
        target_frame_count = int((RECORDING_WINDOW_BEFORE + RECORDING_WINDOW_AFTER) * self.target_fps)

        candidate_data = [] # Tuples of (frame, timestamp, detections_list<DetectionInfo>)
        try:
            history_copy = list(self.frame_history) # Use self.frame_history
            for frame, timestamp, detections_list in history_copy:
                if start_time <= timestamp <= end_time:
                    candidate_data.append((frame, timestamp, detections_list))
        except Exception as e:
            logger.error(f"[{self.local_camera_id}] Error accessing frame history: {e}")
            return

        if not candidate_data:
            logger.warning(f"[{self.local_camera_id}] No frames found in buffer for event window")
            return

        final_data_to_write = candidate_data[:target_frame_count]

        # --- Convert Local Detections to ExternalDetectionInfo format ---
        # Need access to ExternalDetectionInfo model (either import or redefine)
        # Assuming it's imported or available globally (e.g., from main)
        # Import the shared function and model definition
        try:
             from main import record_event_from_data, ExternalDetectionInfo # Import here or at top
        except ImportError:
             logger.error("Could not import shared recorder function/model from main.py")
             return

        processed_frames_data: List[Tuple[np.ndarray, float, List[ExternalDetectionInfo]]] = []
        try:
            for frame, timestamp, local_detections in final_data_to_write:
                external_detections = [
                    ExternalDetectionInfo(
                        class_id=det[1],
                        label=self.model.labels[det[1]], # Get label using local model
                        confidence=det[0],
                        bbox=list(det[2]) # Convert bbox tuple to list
                    ) for det in local_detections
                ]
                processed_frames_data.append((frame, timestamp, external_detections))
        except Exception as e:
             logger.exception(f"[{self.local_camera_id}] Error converting local detections for recording: {e}")
             return

        # --- Call Shared Recorder Function ---
        try:
            await asyncio.to_thread(
                record_event_from_data,
                self.local_camera_id,
                detection_time,
                processed_frames_data, # Pass the prepared data
                self.target_fps,      # Pass target FPS from CameraManager instance
                EVENT_DIR              # Pass EVENT_DIR (ensure it's accessible)
            )
            logger.info(f"[{self.local_camera_id}] Recording job for detection at {detection_time:.2f} handed off.")
        except Exception as e:
            logger.exception(f"[{self.local_camera_id}] Error during recording execution for {detection_time:.2f}: {e}")

    def stop(self):
        """Stop the camera processing and clean up resources"""
        if not self.is_running:
            logger.info("Camera is already stopped")
            return
            
        logger.info("Stopping camera manager...")
        self.is_running = False # Signal threads/tasks to stop

        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            logger.info("Waiting for frame processing thread to stop...")
            self.processing_thread.join(timeout=5.0) # Wait max 5 seconds
            if self.processing_thread.is_alive():
                 logger.warning("Frame processing thread did not stop gracefully.")

        # Cancel any pending recording tasks
        if self.active_recording_tasks:
            logger.info(f"Cancelling {len(self.active_recording_tasks)} active recording tasks...")
            for task in list(self.active_recording_tasks):
                 if not task.done():
                    task.cancel()
            self.active_recording_tasks.clear()

        # Clean up device
        if self.device:
            try:
                self.device.close()
                logger.info("Camera device closed")
            except Exception as e:
                logger.error(f"Error closing camera device: {str(e)}")

        logger.info("Camera manager stopped")