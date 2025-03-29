import cv2
import asyncio
import base64
import time
import datetime
import numpy as np
from pathlib import Path
import collections
from typing import Set, Optional, Deque, List, Tuple
import threading
import queue
import logging
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320
from fastapi import WebSocket

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

class CameraManager:
    """
    Manages camera operations: setup, frame processing, streaming, and event recording.
    Events are recorded 15 seconds after a person is detected, capturing footage
    from 15 seconds before to 15 seconds after the detection.
    Includes a cooldown period to prevent rapid successive recordings.
    """
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.is_running: bool = False
        self.frame_queue = queue.Queue(maxsize=10) # Increased buffer slightly for smoother streaming
        self.latest_frame: Optional[str] = None
        self.device: Optional[AiCamera] = None
        self.model = None
        self.annotator = None
        self.processing_thread = None
        self.distribution_task = None
        self.loop = None # To store the asyncio event loop

        # Frame history settings
        # Estimate required buffer size based on expected FPS and buffer duration
        # Assuming an average FPS of 20-30. A buffer size of ~1000 should cover > 30 seconds.
        self.frame_history: Deque[Tuple[np.ndarray, float]] = collections.deque(maxlen=1000) 

        # Video settings
        self.target_fps = 25.0 # Target FPS for recorded video

        # Keep track of scheduled recording tasks to avoid duplicates if needed, though the requirement implies separate clips
        self.active_recording_tasks: Set[asyncio.Task] = set()
        
        # Cooldown tracking
        self.last_recording_trigger_time: float = 0.0 # Time the last recording job was initiated

        # Continuous presence tracking
        self.person_continuously_detected_since: float = 0.0 # Timestamp when current continuous detection started, 0 if none

        logger.info("CameraManager initialized")

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
        self.loop = asyncio.get_running_loop() # Get the loop where start() is called

        # Start camera processing in a separate thread
        self.processing_thread = threading.Thread(target=self._process_frames_thread, daemon=True)
        self.processing_thread.start()

        # Start the frame distribution in an async task
        self.distribution_task = asyncio.create_task(self._distribute_frames())
        logger.info("Camera manager started successfully")

    def _get_frame_buffer_duration(self):
        """Calculate the current duration covered by the frame buffer in seconds"""
        if len(self.frame_history) < 2:
            return 0
        oldest_time = self.frame_history[0][1]
        newest_time = self.frame_history[-1][1]
        return newest_time - oldest_time

    def _process_frames_thread(self):
        """
        Processes frames from the camera in a non-blocking thread.
        Stores *annotated* frames, performs detection, schedules recording jobs,
        and prepares frames for streaming.
        """
        logger.info("Frame processing thread started")
        last_log_time = 0

        try:
            with self.device as stream:
                for frame_obj in stream: # Assuming stream yields a frame object
                    if not self.is_running:
                        logger.info("Stopping frame processing as is_running=False")
                        break

                    try:
                        current_time = time.time()
                        # Get the frame image *before* annotation for potential clean storage if needed later
                        # but we will annotate before storing for this request.
                        frame_image = frame_obj.image

                        # Log buffer stats occasionally (can be done before detection)
                        if current_time - last_log_time > 60: # Log every minute
                            buffer_duration = self._get_frame_buffer_duration()
                            logger.info(f"Frame buffer stats: {len(self.frame_history)} frames covering {buffer_duration:.2f} seconds")
                            last_log_time = current_time

                        # --- Detection ---
                        detections = frame_obj.detections[frame_obj.detections.confidence > 0.55]

                        # --- Annotation ---
                        # Prepare labels for annotation
                        labels = [f"{self.model.labels[class_id]}: {score:0.2f}"
                                  for _, score, class_id, _ in detections]
                        # Annotate the frame IN PLACE (modifies frame_obj.image)
                        self.annotator.annotate_boxes(frame_obj, detections, labels=labels)

                        # --- Store Annotated Frame ---
                        # Now frame_image (which is frame_obj.image) is annotated
                        # Make a copy before storing to avoid issues if frame_obj is reused/modified further
                        annotated_frame_copy = frame_image.copy()
                        self.frame_history.append((annotated_frame_copy, current_time))

                        # --- Person Detection & Recording Trigger ---
                        person_currently_detected = False
                        # Reuse detections from earlier
                        for _, score, class_id, _ in detections:
                            if self.model.labels[class_id] == 'person' and score > 0.6:
                                person_currently_detected = True
                                break

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
                                    self._schedule_recording(current_time, annotated_frame_copy.shape),
                                    self.loop
                                )
                            else:
                                logger.warning("Asyncio loop not available, cannot schedule recording.")
                        elif person_currently_detected:
                             logger.debug(f"Person detected at {current_time:.2f}, but within cooldown and below long presence threshold. Ignoring trigger.")

                        # --- Frame Encoding for Streaming ---
                        # Encode the *already annotated* frame for streaming
                        _, buffer = cv2.imencode('.jpg', frame_image) # frame_image is already annotated
                        encoded_image = base64.b64encode(buffer).decode('utf-8')

                        # Put encoded frame in queue for distribution
                        try:
                            if self.frame_queue.full():
                                self.frame_queue.get_nowait() # Remove oldest if full
                            self.frame_queue.put_nowait(encoded_image)
                        except queue.Full:
                            pass # Skip if queue is somehow full after check

                    except Exception as e:
                        logger.exception(f"Error processing frame: {str(e)}") # Log traceback
                        time.sleep(0.1) # Small delay if errors occur

        except Exception as e:
            logger.exception(f"Fatal error in frame processing thread: {str(e)}") # Log traceback
        finally:
            logger.info("Frame processing thread stopped")
            self.is_running = False # Ensure flag is set if thread exits unexpectedly

    async def _schedule_recording(self, detection_time: float, frame_shape: Tuple[int, int, int]):
        """Schedules the _record_event coroutine to run after a delay."""
        task = asyncio.create_task(self._delayed_record_event(detection_time, frame_shape))
        self.active_recording_tasks.add(task)
        # Optional: Clean up completed tasks from the set
        task.add_done_callback(self.active_recording_tasks.discard)
        logger.info(f"Recording task scheduled for detection at {detection_time:.2f}, will run in {RECORDING_DELAY}s")

    async def _delayed_record_event(self, detection_time: float, frame_shape: Tuple[int, int, int]):
        """Waits for the delay, then triggers the actual recording."""
        await asyncio.sleep(RECORDING_DELAY)
        logger.info(f"Executing recording job for detection at {detection_time:.2f}")
        # Run the potentially blocking file I/O in a separate thread to avoid blocking asyncio loop
        await asyncio.to_thread(self._record_event, detection_time, frame_shape)

    def _record_event(self, detection_time: float, frame_shape: Tuple[int, int, int]):
        """
        Retrieves *annotated* frames around the detection time and writes them to a video file,
        aiming for a duration close to the defined window by limiting frame count.
        This method is designed to be run in a separate thread via asyncio.to_thread.
        """
        start_time = detection_time - RECORDING_WINDOW_BEFORE
        end_time = detection_time + RECORDING_WINDOW_AFTER
        target_duration = RECORDING_WINDOW_BEFORE + RECORDING_WINDOW_AFTER # 30.0 seconds
        target_frame_count = int(target_duration * self.target_fps) # Calculate target number of frames

        # Create unique filename based on detection time
        timestamp_str = datetime.datetime.fromtimestamp(detection_time).strftime("%Y%m%d_%H%M%S_%f")
        video_path = EVENT_DIR / f"event_{timestamp_str}.avi"
        video_path_str = str(video_path)

        # Get frame dimensions (height, width from shape)
        height, width = frame_shape[:2]
        if height == 0 or width == 0:
            logger.error(f"Invalid frame dimensions for recording: {frame_shape}")
            return

        # Filter frames from the buffer based on time window
        candidate_frames = []
        try:
            # Create a snapshot for thread safety during iteration
            history_copy = list(self.frame_history)
            for frame, timestamp in history_copy: # frame here is now the annotated one
                if start_time <= timestamp <= end_time:
                    candidate_frames.append(frame)
        except Exception as e:
            logger.error(f"Error accessing frame history: {e}")
            return

        if not candidate_frames:
            logger.warning(f"No frames found in buffer for detection time {detection_time:.2f} window [{start_time:.2f} - {end_time:.2f}]")
            return

        # Limit the number of frames to write to match the target duration
        frames_to_write = candidate_frames[:target_frame_count]
        actual_frame_count = len(frames_to_write)
        actual_duration = actual_frame_count / self.target_fps if self.target_fps > 0 else 0

        logger.info(f"Selected {actual_frame_count} frames (target: {target_frame_count}) for ~{actual_duration:.2f}s video at {video_path_str} for detection at {detection_time:.2f}")

        # Create VideoWriter object
        video_writer = None
        try:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # Use the actual frame dimensions for the writer
            video_writer = cv2.VideoWriter(video_path_str, fourcc, self.target_fps, (width, height))

            if not video_writer.isOpened():
                logger.error(f"Failed to open video writer at {video_path_str}")
                return

            # Write the selected frames (these are annotated)
            for frame in frames_to_write:
                video_writer.write(frame)

            logger.info(f"Successfully wrote video: {video_path_str} ({actual_frame_count} frames, ~{actual_duration:.2f}s)")

        except Exception as e:
            logger.exception(f"Error writing video file {video_path_str}: {str(e)}") # Log traceback
        finally:
            if video_writer:
                video_writer.release()

    # --- Methods below remain largely unchanged ---

    async def _distribute_frames(self):
        """Distribute frames to all connected clients"""
        logger.info("Frame distribution task started")
        while self.is_running:
            try:
                # Get latest frame with a timeout to avoid blocking indefinitely
                try:
                    self.latest_frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    # No new frame, wait briefly
                    await asyncio.sleep(0.01)
                    continue

                # Only process if we have connections and a frame
                if self.active_connections and self.latest_frame:
                    # Send frame to all connected clients
                    # Create a copy of the set for safe iteration
                    current_connections = list(self.active_connections)
                    tasks = []
                    for connection in current_connections:
                        # Create task for each send
                        tasks.append(self._send_frame_to_client(connection, self.latest_frame))
                    
                    # Wait for all send tasks to complete (or fail)
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Handle disconnected clients based on results
                    disconnected = set()
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            conn = current_connections[i]
                            logger.warning(f"Error sending frame to client: {result}. Marking for removal.")
                            disconnected.add(conn)
                    
                    # Remove disconnected clients from the main set
                    if disconnected:
                         self.active_connections.difference_update(disconnected)
                         logger.info(f"Removed {len(disconnected)} disconnected client(s), {len(self.active_connections)} remaining")

            except Exception as e:
                logger.exception(f"Error in distribute frames loop: {str(e)}")
                await asyncio.sleep(0.1) # Longer sleep on general error

            # Small delay even if frame was processed
            await asyncio.sleep(0.02) # Adjust as needed for desired streaming FPS

        logger.info("Frame distribution task stopped")

    async def _send_frame_to_client(self, connection: WebSocket, frame_data: str):
        """Sends frame data to a single WebSocket client."""
        await connection.send_text(frame_data)

    def add_client(self, websocket: WebSocket):
        """Add a new client connection"""
        self.active_connections.add(websocket)
        logger.info(f"Client added, now have {len(self.active_connections)} connections")

        # Send the latest frame immediately if available
        if self.latest_frame:
            asyncio.create_task(self._send_latest_frame(websocket))

    async def _send_latest_frame(self, websocket: WebSocket):
        """Send the latest frame to a specific client upon connection"""
        if self.latest_frame and websocket in self.active_connections:
            try:
                logger.info(f"Sending latest frame to new client")
                await websocket.send_text(self.latest_frame)
            except Exception as e:
                logger.warning(f"Error sending latest frame to new client: {str(e)}")
                # Remove failed connection
                self.active_connections.discard(websocket) # Use discard for sets

    def remove_client(self, websocket: WebSocket):
        """Remove a client connection"""
        removed = websocket in self.active_connections
        self.active_connections.discard(websocket)
        if removed:
            logger.info(f"Client removed, now have {len(self.active_connections)} connections")

    def stop(self):
        """Stop the camera processing and clean up resources"""
        if not self.is_running:
            logger.info("Camera is already stopped")
            return
            
        logger.info("Stopping camera manager...")
        self.is_running = False # Signal threads/tasks to stop

        # Cancel distribution task
        if self.distribution_task and not self.distribution_task.done():
            self.distribution_task.cancel()
            logger.info("Frame distribution task cancelled")
            # Optionally wait for it
            # try:
            #    await self.distribution_task
            # except asyncio.CancelledError:
            #    logger.info("Distribution task acknowledged cancellation.")


        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            logger.info("Waiting for frame processing thread to stop...")
            self.processing_thread.join(timeout=5.0) # Wait max 5 seconds
            if self.processing_thread.is_alive():
                 logger.warning("Frame processing thread did not stop gracefully.")

        # Cancel any pending recording tasks
        if self.active_recording_tasks:
            logger.info(f"Cancelling {len(self.active_recording_tasks)} active recording tasks...")
            for task in list(self.active_recording_tasks): # Iterate over a copy
                 if not task.done():
                    task.cancel()
            # Clear the set after cancellation attempts
            self.active_recording_tasks.clear()


        # Clean up device
        if self.device:
            try:
                self.device.close()
                logger.info("Camera device closed")
            except Exception as e:
                logger.error(f"Error closing camera device: {str(e)}")

        logger.info("Camera manager stopped")