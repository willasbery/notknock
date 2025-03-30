import cv2
import asyncio
import json
import time
import datetime
import numpy as np
from pathlib import Path
import collections
from typing import Deque, List, Tuple, Any, Dict, Set
import logging

logger = logging.getLogger("EventsManager")

# Constants (can be shared or passed from CameraManager if needed)
EVENT_DIR = Path('./events') # Consider making this configurable
RECORDING_WINDOW_BEFORE = 15.0
RECORDING_WINDOW_AFTER = 15.0
TARGET_FPS = 25.0

# Ensure event directory exists
EVENT_DIR.mkdir(exist_ok=True, parents=True)

# Define a type hint for detection data (adjust if your model outputs differently)
# Assuming format: (confidence, class_id, bbox_tuple)
DetectionInfo = Tuple[float, int, Tuple[int, int, int, int]]
# Frame data stored will be (frame_image, timestamp, list_of_detections)
FrameData = Tuple[np.ndarray, float, List[DetectionInfo]]

class EventsManager:
    """Handles the recording of event videos and associated metadata."""

    def __init__(self, model_labels: Dict[int, str]):
        self.model_labels = model_labels # Needed to map class_id to label name
        self.active_tasks: asyncio.Queue[asyncio.Task] = asyncio.Queue()
        logger.info("EventsManager initialized")

    async def schedule_event_recording(
        self,
        detection_time: float,
        frame_history_snapshot: List[FrameData],
        delay: float
    ):
        """Schedules the recording task to run after a specified delay."""
        if not frame_history_snapshot:
            logger.warning("Received empty frame history snapshot, cannot schedule recording.")
            return

        # Create and queue the task
        task = asyncio.create_task(
            self._delayed_record_event(detection_time, frame_history_snapshot, delay)
        )
        await self.active_tasks.put(task)
        logger.info(f"Recording task scheduled for detection at {detection_time:.2f}, will run in {delay:.1f}s. Queue size: {self.active_tasks.qsize()}")

    async def _delayed_record_event(
        self,
        detection_time: float,
        frame_history_snapshot: List[FrameData],
        delay: float
    ):
        """Waits for the delay, then triggers the actual recording in a thread."""
        try:
            await asyncio.sleep(delay)
            logger.info(f"Executing recording job for detection event at {detection_time:.2f}")
            # Run the potentially blocking file I/O in a separate thread
            await asyncio.to_thread(
                self._record_event_to_files,
                detection_time,
                frame_history_snapshot
            )
        except asyncio.CancelledError:
             logger.info(f"Recording task for detection at {detection_time:.2f} cancelled.")
        except Exception as e:
             logger.exception(f"Error during delayed recording execution for detection at {detection_time:.2f}: {e}")
        finally:
            # Dequeue the completed/cancelled task
             try:
                 # Get task without waiting, it should be the one that just finished/cancelled
                 completed_task = self.active_tasks.get_nowait()
                 self.active_tasks.task_done()
                 logger.debug(f"Dequeued task. Queue size: {self.active_tasks.qsize()}")
             except asyncio.QueueEmpty:
                 logger.warning("Attempted to dequeue task, but queue was empty.")


    def _record_event_to_files(
        self,
        detection_time: float,
        frame_history_snapshot: List[FrameData]
    ):
        """
        Retrieves frames and detections from the snapshot, writes video and JSON metadata.
        This method runs in a separate thread via asyncio.to_thread.
        """
        start_time = detection_time - RECORDING_WINDOW_BEFORE
        end_time = detection_time + RECORDING_WINDOW_AFTER
        target_duration = RECORDING_WINDOW_BEFORE + RECORDING_WINDOW_AFTER
        target_frame_count = int(target_duration * TARGET_FPS)

        # Create unique base filename based on detection time
        timestamp_str = datetime.datetime.fromtimestamp(detection_time).strftime("%Y%m%d_%H%M%S_%f")
        base_filename = f"event_{timestamp_str}"
        video_path = EVENT_DIR / f"{base_filename}.avi"
        json_path = EVENT_DIR / f"{base_filename}.json"

        # Filter frames from the snapshot based on time window
        candidate_frames_data = []
        for frame_data in frame_history_snapshot:
            frame_timestamp = frame_data[1]
            if start_time <= frame_timestamp <= end_time:
                candidate_frames_data.append(frame_data)

        if not candidate_frames_data:
            logger.warning(f"No frames found in snapshot for detection time {detection_time:.2f} window [{start_time:.2f} - {end_time:.2f}]")
            return

        # Limit the number of frames to write to match the target duration
        final_frames_data = candidate_frames_data[:target_frame_count]
        actual_frame_count = len(final_frames_data)
        if actual_frame_count == 0:
            logger.warning(f"Frame count became 0 after limiting for detection time {detection_time:.2f}.")
            return

        actual_duration = actual_frame_count / TARGET_FPS if TARGET_FPS > 0 else 0

        # Prepare data for JSON and get frames for video
        json_metadata = {
            "detection_time": detection_time,
            "video_filename": video_path.name,
            "metadata_filename": json_path.name,
            "event_start_time": final_frames_data[0][1], # Actual start time of saved clip
            "event_end_time": final_frames_data[-1][1], # Actual end time of saved clip
            "target_duration_seconds": target_duration,
            "actual_duration_seconds": round(actual_duration, 2),
            "target_fps": TARGET_FPS,
            "frame_count": actual_frame_count,
            "frames_info": []
        }
        frames_for_video: List[np.ndarray] = []
        all_detected_class_ids: Set[int] = set()

        height, width = 0, 0
        for frame_image, timestamp, detections in final_frames_data:
            if height == 0 or width == 0: # Get dimensions from first valid frame
                 h, w = frame_image.height, frame_image.width
                 if h > 0 and w > 0:
                     height, width = h, w

            frames_for_video.append(frame_image.image) # Append the original frame
            frame_info = {
                "timestamp": timestamp,
                "detections": []
            }
            for score, class_id, bbox in detections:
                label = self.model_labels[class_id]
                frame_info["detections"].append({
                    "class_id": class_id,
                    "label": label,
                    "confidence": round(float(score), 4), # Ensure serializable
                    "bbox": [int(b) for b in bbox] # Ensure serializable
                })
                all_detected_class_ids.add(class_id)
            json_metadata["frames_info"].append(frame_info)

        json_metadata["summary_detected_labels"] = sorted([self.model_labels[cid] for cid in all_detected_class_ids])

        if height == 0 or width == 0:
            logger.error(f"Could not determine valid frame dimensions for recording event at {detection_time:.2f}")
            return

        logger.info(f"Writing {actual_frame_count} frames ({actual_duration:.2f}s) to {video_path.name} and {json_path.name}")

        # --- Write Video File ---
        video_writer = None
        try:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # Video is written aiming for TARGET_FPS playback speed
            video_writer = cv2.VideoWriter(str(video_path), fourcc, TARGET_FPS, (width, height))

            if not video_writer.isOpened():
                logger.error(f"Failed to open video writer at {video_path}")
            else:
                for frame in frames_for_video:
                    video_writer.write(frame)
                logger.debug(f"Video writing complete for {video_path.name}")

        except Exception as e:
            logger.exception(f"Error writing video file {video_path}: {e}")
        finally:
            if video_writer:
                video_writer.release()

        # --- Write JSON Metadata File ---
        try:
            with open(json_path, 'w') as f:
                json.dump(json_metadata, f, indent=2)
            logger.debug(f"Metadata writing complete for {json_path.name}")
            logger.info(f"Successfully recorded event for detection at {detection_time:.2f}")
        except Exception as e:
            logger.exception(f"Error writing JSON metadata file {json_path}: {e}")

    async def stop(self):
        """Cancel any pending recording tasks."""
        logger.info("Stopping EventsManager and cancelling pending tasks...")
        cancelled_count = 0
        while not self.active_tasks.empty():
            try:
                task = self.active_tasks.get_nowait()
                if not task.done():
                    task.cancel()
                    cancelled_count += 1
                self.active_tasks.task_done()
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                 logger.error(f"Error cancelling task during stop: {e}")
        if cancelled_count > 0:
            logger.info(f"Cancelled {cancelled_count} pending recording tasks.")
        logger.info("EventsManager stopped.")
