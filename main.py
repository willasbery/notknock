import os
import uvicorn
import asyncio
import logging
import base64
import cv2
import numpy as np
import datetime
import time
import json
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request, HTTPException, Depends, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect, WebSocketState
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import collections
from queue import Empty
from fastapi.staticfiles import StaticFiles

from CameraManager import CameraManager

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DoorbellApp")

# --- Configuration (Consider moving to a config file/class) ---
EVENT_DIR = Path('./events')
TARGET_FPS = 25.0 # Keep consistent with CameraManager's target
EVENT_DIR.mkdir(exist_ok=True, parents=True) # Ensure exists

# --- Pydantic Models for External Ingestion ---

class ExternalDetectionInfo(BaseModel):
    class_id: int
    label: str # External camera should provide the label
    confidence: float
    bbox: List[int] # List of [xmin, ymin, xmax, ymax]

class ExternalFrameData(BaseModel):
    timestamp: float
    image_base64: str # Base64 encoded JPEG image string
    detections: List[ExternalDetectionInfo]

class ExternalEventPayload(BaseModel):
    camera_id: str = Field(..., description="Unique identifier for the external camera")
    detection_time: float = Field(..., description="Timestamp of the primary detection event")
    frames_data: List[ExternalFrameData] = Field(..., min_items=1)

# --- Shared Recorder Function ---
def record_event_from_data(
    camera_id: str,
    detection_time: float,
    frames_data: List[Tuple[np.ndarray, float, List[ExternalDetectionInfo]]], # Decoded frames + timestamp + detections
    target_fps: float,
    event_dir: Path # Base event directory (e.g., ./events)
):
    """
    Records an event video and metadata JSON from provided frame data into a
    camera-specific subdirectory within the event_dir.
    Intended to be run in a threadpool (e.g., via asyncio.to_thread).
    """
    if not frames_data:
        logger.warning(f"[{camera_id}] No frame data provided for recording event at {detection_time:.2f}")
        return

    # Determine video dimensions from the first frame
    first_frame = frames_data[0][0]
    height, width = first_frame.shape[:2]
    if height == 0 or width == 0:
        logger.error(f"[{camera_id}] Invalid frame dimensions for recording: H={height}, W={width}")
        return

    # --- Edit 1: Create camera-specific subdirectory ---
    camera_event_dir = event_dir / camera_id # e.g., ./events/local_cam or ./events/external-cam-01
    try:
        camera_event_dir.mkdir(exist_ok=True, parents=True) # Ensure the subdirectory exists
    except OSError as e:
         logger.error(f"[{camera_id}] Failed to create event subdirectory {camera_event_dir}: {e}")
         return # Cannot proceed without directory

    # --- Edit 2: Construct paths within the subdirectory ---
    # Create unique filename (base name remains the same)
    timestamp_dt = datetime.datetime.fromtimestamp(detection_time)
    timestamp_str = timestamp_dt.strftime("%Y%m%d_%H%M%S_%f")
    # Keep base filename simple, path includes camera ID already
    base_filename = f"event_{timestamp_str}"
    # Create full paths within the camera-specific directory
    video_path = camera_event_dir / f"{base_filename}.avi"
    json_path = camera_event_dir / f"{base_filename}.json"

    actual_start_time = frames_data[0][1]
    actual_end_time = frames_data[-1][1]
    actual_duration = actual_end_time - actual_start_time
    actual_frame_count = len(frames_data)
    effective_fps = actual_frame_count / actual_duration if actual_duration > 0 else target_fps
    logger.info(f"[{camera_id}] Recording event: {actual_frame_count} frames ({actual_duration:.2f}s) "
                f"for detection at {detection_time:.2f}. -> {video_path}") # Log full path

    # Prepare metadata (include relative paths if desired, or keep full name)
    metadata = {
        "recording_info": {
            "camera_id": camera_id,
            "detection_time": detection_time,
            "target_fps": target_fps,
            "video_file": video_path.name, # Store only the filename in metadata
            "metadata_file": json_path.name, # Store only the filename
            "creation_timestamp": time.time(),
            "actual_frame_count": actual_frame_count,
            "actual_duration_seconds": round(actual_duration, 2),
            "effective_fps": round(effective_fps, 2),
            "actual_start_time": actual_start_time,
            "actual_end_time": actual_end_time,
        },
        "frames_analyzed": [],
        "detections": []
    }
    all_detected_labels = set()
    for i, (frame, timestamp, detections_list) in enumerate(frames_data):
        if detections_list:
            frame_meta_entry = {
                 "frame_index": i, # Index within the saved video
                 "timestamp": timestamp,
                 "num_detections": len(detections_list)
            }
            metadata["frames_analyzed"].append(frame_meta_entry)

            for det in detections_list:
                 metadata["detections"].append({
                    "frame_index": i,
                    "timestamp": timestamp,
                    "class_id": det.class_id,
                    "label": det.label,
                    "confidence": det.confidence,
                    "bbox": det.bbox
                 })
                 all_detected_labels.add(det.label)
    metadata["recording_info"]["summary_detected_labels"] = sorted(list(all_detected_labels))
    metadata["recording_info"]["total_detections"] = len(metadata["detections"])

    # --- Write Video File (using the new video_path) ---
    video_writer = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, target_fps, (width, height))
        if not video_writer.isOpened():
            logger.error(f"[{camera_id}] Failed to open video writer at {video_path}")
            return
        for frame, _, _ in frames_data: video_writer.write(frame)
        logger.debug(f"[{camera_id}] Video writing complete for {video_path.name}")

    except Exception as e:
        logger.exception(f"[{camera_id}] Error writing video file {video_path}: {e}")
        if video_path.exists(): 
            try: 
                video_path.unlink() 
            except OSError: 
                pass
        return
    finally:
        if video_writer: video_writer.release()

    # --- Write JSON Metadata File (using the new json_path) ---
    try:
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"[{camera_id}] Successfully recorded event: {video_path.name}, {json_path.name}")
    except Exception as e:
        logger.exception(f"[{camera_id}] Error writing JSON metadata file {json_path}: {e}")


# --- WebSocket Connection Management for VIEWERS ---
# Maps camera_id -> list of viewer WebSockets interested in that camera
viewer_connections: Dict[str, List[WebSocket]] = collections.defaultdict(list)
# Stores the latest frame received from external cameras (bytes)
latest_external_frames: Dict[str, bytes] = {}
# Lock for safely accessing shared viewer_connections dict from multiple tasks
viewer_lock = asyncio.Lock()

async def add_viewer(camera_id: str, websocket: WebSocket):
    """Adds a viewer WebSocket to the list for a given camera_id."""
    async with viewer_lock:
        viewer_connections[camera_id].append(websocket)
    logger.info(f"Viewer connected for camera: {camera_id}. Total viewers: {len(viewer_connections[camera_id])}")

async def remove_viewer(camera_id: str, websocket: WebSocket):
    """Removes a viewer WebSocket from the list."""
    async with viewer_lock:
        if camera_id in viewer_connections:
            try:
                viewer_connections[camera_id].remove(websocket)
                if not viewer_connections[camera_id]: # Remove camera_id if list is empty
                    del viewer_connections[camera_id]
                logger.info(f"Viewer disconnected for camera: {camera_id}. Remaining: {len(viewer_connections.get(camera_id, []))}")
            except ValueError:
                pass # Ignore if websocket not found (already removed)

async def broadcast_frame_to_viewers(camera_id: str, frame_data: Any, data_type: str):
    """Sends frame data (bytes or text) to all viewers of a camera."""
    disconnected_viewers = []
    async with viewer_lock: # Ensure exclusive access while iterating/sending
        if camera_id in viewer_connections:
            viewers = viewer_connections[camera_id]
            # logger.debug(f"Broadcasting frame for {camera_id} to {len(viewers)} viewers.")
            for viewer_ws in viewers:
                try:
                    if viewer_ws.client_state == WebSocketState.CONNECTED:
                        if data_type == "text":
                            await viewer_ws.send_text(frame_data)
                        elif data_type == "bytes":
                            await viewer_ws.send_bytes(frame_data)
                    else:
                        disconnected_viewers.append(viewer_ws)
                except Exception: # Handle potential errors during send
                    disconnected_viewers.append(viewer_ws)

    # Remove viewers that failed or were disconnected outside the lock
    if disconnected_viewers:
         logger.warning(f"Found {len(disconnected_viewers)} disconnected viewers for {camera_id} during broadcast.")
         # Removing them requires the lock again, or do it separately
         # For simplicity, let the disconnect handler manage removal.


# --- FastAPI Setup ---

# --- State for External SOURCE Connections (Cameras sending frames) ---
# Maps camera_id -> WebSocket connection of the camera sending frames
# (Assuming only one source per camera_id)
source_connections: Dict[str, WebSocket] = {}

def get_camera_manager():
    # Pass shared config if needed, e.g. target_fps, event_dir
    # For now, CameraManager uses its own constants, but recorder func uses globals/args
    return CameraManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create camera manager for the *local* camera on startup
    app.state.camera_manager = get_camera_manager()
    logger.info("Starting application and local camera manager...")
    app.state.camera_manager.start() # Starts local camera processing
    logger.info("Local camera manager started")

    # --- Start Background Task for Local Camera Broadcasting ---
    app.state.local_broadcast_task = asyncio.create_task(broadcast_local_camera_frames(app.state.camera_manager))
    logger.info("Local camera frame broadcasting task started.")

    yield

    logger.info("Shutting down application...")
    # --- Stop Background Task ---
    if app.state.local_broadcast_task:
         app.state.local_broadcast_task.cancel()
         try:
              await app.state.local_broadcast_task # Allow task to finish cancelling
         except asyncio.CancelledError:
              logger.info("Local camera frame broadcasting task cancelled.")
         except Exception as e:
              logger.error(f"Error during broadcast task shutdown: {e}")

    # Clean up any remaining external connections on shutdown
    for ws in list(source_connections.values()):
         try:
             if ws.client_state != WebSocketState.DISCONNECTED:
                 await ws.close(code=1001, reason="Server shutting down")
         except Exception:
             pass # Ignore errors during shutdown cleanup
    source_connections.clear()
    logger.info("External source WebSocket connections cleared.")

    # Clean up WebSocket connections
    async with viewer_lock:
         for camera_id, viewers in list(viewer_connections.items()):
              for ws in viewers:
                  if ws.client_state != WebSocketState.DISCONNECTED:
                      try: await ws.close(code=1001)
                      except Exception: pass
         viewer_connections.clear()
    logger.info("Viewer WebSocket connections cleared.")

async def broadcast_local_camera_frames(manager: CameraManager):
    """Task to broadcast frames from the local camera queue to viewers."""
    local_cam_id = manager.local_camera_id # Get ID from manager
    logger.info(f"Broadcasting task running for local camera: {local_cam_id}")
    while True:
        try:
             # Get latest frame from queue (non-blocking)
             frame_base64 = manager.frame_queue.get_nowait()
             # Broadcast this frame (text format)
             await broadcast_frame_to_viewers(local_cam_id, frame_base64, "text")
             manager.frame_queue.task_done() # Mark task as done if using Queue for flow control

        except Empty:
             # Queue is empty, wait a short time
             await asyncio.sleep(0.02) # Approx 50 FPS check rate
        except Exception as e:
             logger.error(f"Error in local broadcast loop: {e}")
             await asyncio.sleep(1) # Longer sleep on error


app = FastAPI(lifespan=lifespan)

# --- Edit 1: Mount the static directory ---
# Serve files from the ./events directory under the path /static/events
# Check if EVENT_DIR exists before mounting
if EVENT_DIR.is_dir():
    app.mount("/static/events", StaticFiles(directory=EVENT_DIR, html=False), name="event_files")
    logger.info(f"Serving static files from {EVENT_DIR} at /static/events")
else:
    logger.warning(f"Event directory {EVENT_DIR} not found. Static file serving for events disabled.")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTML for testing the video stream
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Camera Viewer</title>
    </head>
    <body>
        <h1>Camera Stream</h1>
        Camera ID: <input type="text" id="cameraId" value="local_cam"/>
        <button onclick="connectWs()">Connect</button>
        <br/>
        <img id="stream" style="max-width: 100%;" />
        
        <script>
            let ws = null;
            const image = document.getElementById('stream');

            function connectWs() {
                if (ws) {
                    ws.close();
                }
                const cameraId = document.getElementById('cameraId').value;
                if (!cameraId) {
                    alert("Please enter a Camera ID");
                    return;
                }
                // Connect to the new viewer endpoint
                const wsUrl = `ws://${window.location.host}/ws/view/${cameraId}`;
                console.log(`Connecting to ${wsUrl}`);
                ws = new WebSocket(wsUrl);

                ws.onopen = function(event) {
                    console.log("WebSocket connection opened");
                    image.src = ""; // Clear image on new connection
                };
            
            ws.onmessage = function(event) {
                    // --- Edit: Simplify JS - Always expect base64 string ---
                    if (typeof event.data === 'string') {
                image.src = "data:image/jpeg;base64," + event.data;
                    } else {
                        console.warn("Received non-string data, expected base64 string.");
                    }
                    // --- End Edit ---
            };
            ws.onclose = function(event) {
                    console.log('Connection closed:', event.reason);
                    image.src = ""; // Clear image
                    ws = null;
                };
                 ws.onerror = function(event) {
                    console.error('WebSocket error:', event);
                };
            }
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.get("/status")
async def status(request: Request):
    """API endpoint to check application status"""
    # Include status of the local camera manager
    local_cam_manager = request.app.state.camera_manager
    return {
        "status": "running",
        "local_camera_running": local_cam_manager.is_running if local_cam_manager else False,
        "local_camera_clients": len(local_cam_manager.active_connections) if local_cam_manager else 0
    }
    
@app.get("/cameras")
async def cameras(request: Request):
    """API endpoint to list available cameras (local + active external sources)."""
    cameras_list = []

    # 1. Add the local camera (defined by environment variables)
    local_cam_id = os.environ.get("CAMERA_ID", "local_cam")
    cameras_list.append({
        "id": local_cam_id,
        "name": os.environ.get("CAMERA_NAME", "Local Camera"),
        "location": os.environ.get("CAMERA_LOCATION", "N/A"),
        "type": "local",
        "live_status": "connected" if local_cam_id in source_connections else "disconnected" # Check if local cam ALSO sends via WS (unlikely with current setup but possible)
                                       # OR base this on CameraManager status:
                                       # "live_status": "active" if request.app.state.camera_manager and request.app.state.camera_manager.is_running else "inactive"
    })

    # 2. Add currently connected external cameras (from source_connections)
    # Make a copy of keys for safe iteration if needed, though access should be quick
    connected_external_ids = list(source_connections.keys())

    for cam_id in connected_external_ids:
        logger.info(f"Adding external camera: {cam_id}")
        # Avoid duplicating the local camera if it happens to connect via WebSocket too
        if cam_id != local_cam_id:
            cameras_list.append({
                "id": cam_id,
                "name": f"External Camera ({cam_id})", # Use ID as name, or implement registration later
                "location": "Unknown", # Location info not available from WebSocket connection alone
                "type": "external",
                "live_status": "connected" # It's in source_connections, so it's connected
            })

    return {"cameras": cameras_list}

# --- New Endpoint for External Camera Event Ingestion ---
@app.post("/ingest/event")
async def ingest_external_event(payload: ExternalEventPayload):
    """
    Endpoint for external cameras to POST event data (frames + detections).
    """
    logger.info(f"Received event payload from camera: {payload.camera_id} "
                f"with {len(payload.frames_data)} frames.")

    processed_frames_data: List[Tuple[np.ndarray, float, List[ExternalDetectionInfo]]] = []

    # Decode base64 images and prepare data structure for the recorder function
    try:
        for frame_data in payload.frames_data:
            img_bytes = base64.b64decode(frame_data.image_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_np is None:
                 raise ValueError(f"Failed to decode image for timestamp {frame_data.timestamp}")
            # Store: Decoded Image, Timestamp, Original Detections List
            processed_frames_data.append((img_np, frame_data.timestamp, frame_data.detections))

    except (base64.binascii.Error, ValueError) as e:
        logger.error(f"Failed to process images from {payload.camera_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data received: {e}")
    except Exception as e:
         logger.exception(f"Unexpected error processing payload from {payload.camera_id}: {e}")
         raise HTTPException(status_code=500, detail="Internal server error processing event data.")

    # Sort data by timestamp just in case it's out of order
    processed_frames_data.sort(key=lambda x: x[1])

    # Schedule the recording task (run blocking I/O in a thread)
    # Use the shared recorder function
    try:
        await asyncio.to_thread(
            record_event_from_data,
            payload.camera_id,
            payload.detection_time,
            processed_frames_data,
            TARGET_FPS, # Use shared target FPS
            EVENT_DIR   # Use shared event dir
        )
        return {"status": "received", "message": f"Event from {payload.camera_id} scheduled for recording."}
    except Exception as e:
         # Log the error from the recording function if it bubbles up unexpectedly
         logger.exception(f"Error occurred during scheduling/execution of recording for {payload.camera_id}: {e}")
         # Return a server error - the recording likely failed
         raise HTTPException(status_code=500, detail="Failed to process and save the event.")

# --- Endpoint for Browsing Events ---
@app.get("/events/{camera_id}")
async def get_camera_events(
    request: Request, # Add Request to access url_for
    camera_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Number of events per page")
):
    """
    Retrieves a paginated list of recorded event metadata for a specific camera,
    sorted by detection time (newest first), including video URLs.
    """
    camera_event_dir = EVENT_DIR / camera_id
    if not camera_event_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"No events found for camera ID: {camera_id}")

    try:
        # Find all JSON files in the camera's event directory
        json_files = sorted(
            list(camera_event_dir.glob('event_*.json')),
            key=lambda p: p.name,
            reverse=True
        )
    except Exception as e:
        logger.error(f"Error listing event files for {camera_id}: {e}")
        raise HTTPException(status_code=500, detail="Error reading event directory")

    total_items = len(json_files)
    if total_items == 0:
        return {"total_items": 0, "total_pages": 0, "page": page, "size": size, "items": []}

    total_pages = (total_items + size - 1) // size

    start_index = (page - 1) * size
    end_index = start_index + size

    if start_index >= total_items:
        raise HTTPException(status_code=404, detail=f"Page {page} out of range for camera {camera_id}")

    page_files = json_files[start_index:end_index]

    event_items = []
    for json_path in page_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                # --- Edit 2: Construct and add video URL ---
                video_filename = data.get("recording_info", {}).get("video_file")
                if video_filename:
                    # Construct the path relative to the mounted directory
                    relative_video_path = f"{camera_id}/{video_filename}"
                    try:
                        # Use url_for with the name assigned during mounting
                        video_url = request.url_for('event_files', path=relative_video_path)
                        data['recording_info']['video_url'] = str(video_url)
                    except RuntimeError as e:
                         logger.error(f"Failed to generate URL for {relative_video_path}: {e}. Is 'event_files' route mounted?")
                         data['recording_info']['video_url'] = None # Or omit the key
                else:
                     data.setdefault('recording_info', {})['video_url'] = None
                # --- End Edit ---
                event_items.append(data)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode JSON file: {json_path}")
            event_items.append({"error": "Failed to load event metadata", "filename": json_path.name})
        except Exception as e:
            logger.error(f"Error reading event file {json_path}: {e}")
            event_items.append({"error": "Error reading event file", "filename": json_path.name})

    return {
        "total_items": total_items,
        "total_pages": total_pages,
        "page": page,
        "size": size,
        "items": event_items
    }

# --- WebSocket Endpoint for EXTERNAL Cameras to SEND Frames ---
@app.websocket("/ws/live/{camera_id}")
async def websocket_receive_external_frames(websocket: WebSocket, camera_id: str):
    """ WebSocket endpoint for receiving live frames FROM external cameras. """
    await websocket.accept()
    # Basic check: Allow only one source connection per camera_id
    if camera_id in source_connections:
        logger.warning(f"Camera {camera_id} tried to connect source feed while already connected. Closing new connection.")
        await websocket.close(code=1008, reason="Source feed already established for this camera ID.")
        return

    logger.info(f"External camera SOURCE connected: {camera_id} from {websocket.client.host}:{websocket.client.port}")
    source_connections[camera_id] = websocket
    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            logger.debug(f"Received frame ({len(frame_bytes)} bytes) from source: {camera_id}")
            # Store the latest frame (raw bytes is fine for storage)
            latest_external_frames[camera_id] = frame_bytes

            # --- Edit 1: Encode to Base64 before broadcasting ---
            # Encode bytes to base64 string for viewers
            frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
            # Broadcast as text (base64)
            await broadcast_frame_to_viewers(camera_id, frame_base64, "text")
            # --- End Edit ---

    except WebSocketDisconnect:
        logger.info(f"External camera SOURCE WebSocket disconnected: {camera_id}")
    except Exception as e:
        logger.error(f"WebSocket error (external camera source {camera_id}): {e}")
    finally:
        # Clean up source connection and latest frame data
        if camera_id in source_connections:
            del source_connections[camera_id]
        if camera_id in latest_external_frames:
            del latest_external_frames[camera_id]
        logger.info(f"Cleaned up source connection/data for {camera_id}. Sources: {len(source_connections)}")


# --- NEW WebSocket Endpoint for VIEWERS to RECEIVE Frames ---
@app.websocket("/ws/view/{camera_id}")
async def websocket_send_to_viewer(websocket: WebSocket, camera_id: str):
    """ WebSocket endpoint for VIEWERS to connect and receive frames. """
    await websocket.accept()
    await add_viewer(camera_id, websocket)
    
    local_cam_id = os.environ.get("CAMERA_ID", "local_cam") # Get local ID

    try:
        # Send the latest frame immediately if available (always as base64 text)
        latest_frame_base64 = None
        if camera_id == local_cam_id:
            manager = websocket.app.state.camera_manager
            if manager:
                 try:
                      # Get latest from local queue
                      latest_frame_base64 = manager.frame_queue.get_nowait()
                      # Put it back quickly - potential race condition remains if queue is very small/fast
                      manager.frame_queue.put_nowait(latest_frame_base64)
                 except Empty:
                      pass # No initial frame available
        elif camera_id in latest_external_frames:
            # --- Edit 2: Encode stored bytes to base64 for initial send ---
            frame_bytes = latest_external_frames.get(camera_id)
            if frame_bytes:
                latest_frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
            # --- End Edit ---

        if latest_frame_base64:
             try:
                 logger.info(f"Sending initial frame (text) to viewer for {camera_id}")
                 await websocket.send_text(latest_frame_base64) # Always send text
             except Exception as e:
                 logger.warning(f"Failed to send initial frame to viewer for {camera_id}: {e}")

        # Keep connection alive, wait for disconnect/commands
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
            
    except WebSocketDisconnect:
        logger.info(f"Viewer WebSocket disconnected for camera: {camera_id}")
    except Exception as e:
        logger.error(f"WebSocket error (viewer for {camera_id}): {e}")
    finally:
        # Remove viewer from connection list
        await remove_viewer(camera_id, websocket)


if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0") # Allow configuring host
    
    logger.info(f"Starting Uvicorn server on {host}:{port}...")
    uvicorn.run(
        "main:app",
        host=host,
        port=port, 
        reload=False, # Keep reload=False for stability unless debugging lifecycle
        log_level="info"
    )