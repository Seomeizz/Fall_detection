from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from app.config import settings
from app.services.ai_camera import AICameraEngine
from app.services.event_store import EventStore
from app.services.ws_hub import WebSocketHub

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("fallguard.backend")

hub = WebSocketHub()
event_store = EventStore(settings.event_log)


def on_fall_event(event: dict[str, Any]) -> None:
    saved = event_store.add(event)
    hub.publish_from_thread({"type": "fall_event", "event": saved})
    logger.warning("FALL EVENT: track=%s confidence=%s", saved.get("track_id"), saved.get("confidence"))


engine = AICameraEngine(settings, on_fall_event)


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.set_loop(asyncio.get_running_loop())
    logger.info("Starting FallGuard API; camera auto-start=%s", settings.auto_start_camera)
    if settings.auto_start_camera:
        engine.start()
    yield
    logger.info("Stopping FallGuard API")
    engine.stop()


app = FastAPI(
    title="FallGuard AI Backend",
    version="0.2.0",
    description="Local FastAPI service for laptop webcam + YOLO Pose + Transformer 8-class inference.",
    lifespan=lifespan,
)

# Prototype is designed for a trusted local network. Do not expose it to the public Internet as-is.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root_page() -> str:
    # Reconnect the browser image after backend restarts or MJPEG stream failures.
    return """<!doctype html><html lang='vi'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>FallGuard AI</title><style>body{font-family:system-ui;background:#101a2a;color:#eef4ff;margin:0;padding:24px}main{max-width:1000px;margin:auto}img{width:100%;max-height:70vh;object-fit:contain;background:#0b1220;border-radius:16px}a{color:#6ee7d0}code{background:#243247;padding:3px 6px;border-radius:6px}</style></head><body><main><h1>FallGuard AI</h1><p>Backend AI trên laptop. Trạng thái JSON: <a href='/api/status'>/api/status</a> · Sự kiện: <a href='/api/events'>/api/events</a> · Tài liệu API: <a href='/docs'>/docs</a></p><img id='camera-stream' src='/stream.mjpg' alt='AI camera stream'><p id='stream-state'>Đang kết nối camera...</p><p>Giữ backend trong mạng nội bộ tin cậy. Luồng video chưa có xác thực trong MVP.</p></main><script>
(()=>{
 const img=document.getElementById('camera-stream');
 const state=document.getElementById('stream-state');
 let retryTimer=null, wasReachable=false, lastUptime=null;
 function reconnect(message){
   state.textContent=message||'Đang kết nối lại camera...';
   if(retryTimer!==null) clearTimeout(retryTimer);
   retryTimer=setTimeout(()=>{img.src='/stream.mjpg?reconnect='+Date.now();},1200);
 }
 img.addEventListener('error',()=>reconnect('Mất luồng camera; đang tự kết nối lại...'));
 async function checkBackend(){
   try{
     const response=await fetch('/api/status',{cache:'no-store'});
     if(!response.ok) throw new Error('API offline');
     const data=await response.json();
     const uptime=Number(data.uptime_seconds);
     const restarted=!wasReachable || (lastUptime!==null && Number.isFinite(uptime) && uptime+2<lastUptime);
     wasReachable=true;
     if(Number.isFinite(uptime)) lastUptime=uptime;
     if(restarted && data.camera==='online') reconnect('Backend đã hoạt động; đang mở lại luồng camera...');
     else if(data.camera!=='online') state.textContent='Camera: '+String(data.camera||'đang chờ');
     else state.textContent='Camera đang phát trực tiếp';
   }catch(_){wasReachable=false;state.textContent='Chưa kết nối backend; đang thử lại...';}
 }
 setInterval(checkBackend,2500);
 checkBackend();
})();
</script></body></html>"""


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "service": "fallguard-backend", "version": app.version}


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    return engine.status()


@app.get("/api/events")
async def api_events(limit: int = Query(default=50, ge=1, le=300)) -> dict[str, Any]:
    items = event_store.list(limit)
    return {"items": items, "count": len(items)}


@app.post("/api/events/{event_id}/ack")
async def acknowledge_event(event_id: str) -> dict[str, Any]:
    event = event_store.acknowledge(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy sự kiện.")
    await hub.broadcast({"type": "event_acknowledged", "event": event})
    return {"ok": True, "event": event}


@app.post("/api/camera/start")
async def start_camera() -> dict[str, Any]:
    result = engine.start()
    await hub.broadcast({"type": "camera_control", "action": "start", **result})
    return result


@app.post("/api/camera/stop")
async def stop_camera() -> dict[str, Any]:
    result = engine.stop()
    await hub.broadcast({"type": "camera_control", "action": "stop", **result})
    return result


async def mjpeg_generator(request: Request):
    boundary = b"--frame\r\n"
    last_version = -1
    try:
        while True:
            if await request.is_disconnected():
                return
            # Do not block the event loop while waiting for the next JPEG frame.
            version, jpeg = await asyncio.to_thread(engine.wait_for_frame, last_version, timeout=1.0)
            status = engine.status()
            if await request.is_disconnected():
                return
            camera_state = status.get("camera")
            if jpeg is None:
                # During loading we wait for a frame; once stopped/failed, close the HTTP stream.
                if camera_state in {"stopping", "stopped", "offline", "error"}:
                    return
                continue
            if version == last_version:
                if camera_state in {"stopping", "stopped", "offline", "error"}:
                    return
                continue
            last_version = version
            yield boundary + b"Content-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n"
    except asyncio.CancelledError:
        # Expected when the server or client closes a streaming request.
        raise


@app.get("/stream.mjpg")
async def stream_mjpeg(request: Request) -> StreamingResponse:
    return StreamingResponse(
        mjpeg_generator(request),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await hub.connect(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            if message.lower() == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        await hub.disconnect(websocket)
    except Exception:
        await hub.disconnect(websocket)
        logger.debug("WebSocket client disconnected", exc_info=True)
