# FastAPI backend — API đã triển khai

Backend source nằm trong `backend/app/`. Camera và model được điều khiển bởi `AICameraEngine`; app Flutter dùng REST/MJPEG/WebSocket.

## `GET /api/health`

Health check: `{"ok": true, "service": "fallguard-backend", "version": "0.2.0"}`.

## `GET /api/status`

Các trường chính: `ok`, `camera`, `ai`, `model`, `device`, `fps`, `people_count`, `predictions`, `last_error`, `last_event`, `uptime_seconds`, `checkpoint`, `stream_url`. `camera` có thể là `starting`, `online`, `stopped`, `offline` hoặc `error`.

## `GET /api/events?limit=50`

Trả `{items: [...], count: n}`. Event được ghi vào `backend/runtime/events.json` (file runtime bị `.gitignore`). Mỗi sự kiện được tạo khi state machine phát hiện chuyển sang trạng thái `FALL_ALERT`, không phải chỉ khi model argmax là `falling`.

## `POST /api/events/{event_id}/ack`

Đánh dấu sự kiện đã xem. Trả 404 nếu event ID không tồn tại.

## `POST /api/camera/start` và `/api/camera/stop`

Khởi động/dừng worker camera + AI. Camera tự khởi động theo mặc định; có thể đổi bằng `FALLGUARD_AUTO_START_CAMERA`.

## `GET /stream.mjpg`

Trả `multipart/x-mixed-replace; boundary=frame` với JPEG đã được YOLO/Transformer annotate. Chỉ một backend process nên mở camera.

## `WS /ws`

Khi sự kiện té ngã được xác nhận, broadcast JSON dạng:

```json
{
  "type": "fall_event",
  "event": {
    "id": "...",
    "timestamp": "...",
    "camera_id": "laptop-camera-0",
    "track_id": 1,
    "label": "falling",
    "confidence": 0.9,
    "confirmed": true,
    "acknowledged": false
  }
}
```

Trường trong ví dụ chỉ mô tả schema. Dữ liệu thật được tạo bởi engine khi chạy camera.

## Bảo mật

MVP chưa có authentication/TLS, chỉ sử dụng trong mạng LAN tin cậy. Không mở port backend trực tiếp ra Internet.
