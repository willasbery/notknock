import os
import uvicorn
import asyncio
import logging
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from contextlib import asynccontextmanager

from CameraManager import CameraManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DoorbellApp")

# Create a camera manager factory instead of a global instance
# This allows reloading without maintaining state between reloads
def get_camera_manager():
    return CameraManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create camera manager on startup
    app.state.camera_manager = get_camera_manager()
    
    # Startup: Configure and start the camera manager
    logger.info("Starting application...")
    app.state.camera_manager.start()
    logger.info("Camera manager started")
    
    yield  # This will keep the lifespan open
    
    # Shutdown: Stop the camera manager
    logger.info("Shutting down application...")
    app.state.camera_manager.stop()
    logger.info("Camera manager stopped")

app = FastAPI(lifespan=lifespan)

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
        <title>Doorbell Camera</title>
    </head>
    <body>
        <h1>Doorbell Camera Stream</h1>
        <img id="stream" style="max-width: 100%;" />
        
        <script>
            const ws = new WebSocket(`ws://${window.location.host}/stream`);
            const image = document.getElementById('stream');
            
            ws.onmessage = function(event) {
                image.src = "data:image/jpeg;base64," + event.data;
            };
            ws.onclose = function(event) {
                console.log('Connection closed');
            };
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
    return {
        "status": "running",
        "camera_running": request.app.state.camera_manager.is_running,
        "connected_clients": len(request.app.state.camera_manager.active_connections)
    }

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("New WebSocket connection established")
    
    # Add client to camera manager
    # Access camera_manager from app.state
    websocket.app.state.camera_manager.add_client(websocket)
    
    try:
        # Keep the connection open until client disconnects
        while True:
            # Wait for messages from the client (if any)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Remove client from camera manager
        websocket.app.state.camera_manager.remove_client(websocket)

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=port, 
        reload=False,  # Enable reload in dev mode
        log_level="info"
    )