# FastAPI backend — FallGuard AI

Backend này chạy trên **laptop đang gắn webcam**. Nó tải checkpoint Transformer 8 lớp, chạy YOLO Pose + BoT-SORT, vẽ nhãn lên hình và cung cấp trạng thái, video MJPEG và sự kiện cho ứng dụng Flutter.

## 1. Cài đặt

Từ thư mục gốc repository, tạo virtual environment và cài PyTorch tương thích với máy (CPU hoặc CUDA) trước theo hướng dẫn chính thức: <https://pytorch.org/get-started/locally/>.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Nếu bạn dùng CMD thay PowerShell, kích hoạt bằng:

```cmd
.venv\Scripts\activate.bat
```

## 2. Thêm checkpoint và cấu hình

Đặt checkpoint đã huấn luyện vào:

```text
artifacts/best_transformer_8class.pth
```

Checkpoint không nằm trong Git mặc định. Nếu tên/đường dẫn khác, sao chép `backend/.env.example` thành `backend/.env` và sửa `FALLGUARD_CHECKPOINT`.

Các cấu hình chính:

- `FALLGUARD_CAMERA_INDEX=0`: camera mặc định; thử 1 nếu có nhiều camera.
- `FALLGUARD_POSE_WEIGHTS=yolo11l-pose.pt`: model pose. Dùng `yolo11n-pose.pt` nếu laptop quá chậm.
- `FALLGUARD_IMGSZ=960`: tăng chất lượng cho người xa nhưng tốn tính toán; có thể thử 640.
- `FALLGUARD_MIRROR_CAMERA=true`: lật ngang hình camera như gương, rồi mới vẽ nhãn nên chữ không bị lật.
- `FALLGUARD_AUTO_START_CAMERA=true`: tự mở camera khi backend khởi động.

Lần đầu chạy, Ultralytics có thể cần Internet để tải YOLO Pose weights.

## 3. Chạy backend

Chạy từ thư mục gốc repo:

```powershell
python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

Mở trên laptop:

- `http://127.0.0.1:8000/` — xem video AI trong trình duyệt.
- `http://127.0.0.1:8000/docs` — tài liệu API Swagger.
- `http://127.0.0.1:8000/api/status` — trạng thái camera/AI.
- `http://127.0.0.1:8000/api/events` — lịch sử cảnh báo.
- `http://127.0.0.1:8000/stream.mjpg` — video MJPEG trực tiếp.

Camera sẽ tự mở. Có thể điều khiển qua `POST /api/camera/start` và `POST /api/camera/stop`.

## 4. Kết nối Flutter trên điện thoại

1. Laptop và điện thoại kết nối cùng Wi-Fi.
2. Trên laptop chạy `ipconfig`; lấy IPv4 của adapter Wi-Fi, ví dụ `192.168.1.25`.
3. Mở FallGuard app → Cài đặt → nhập `http://192.168.1.25:8000` → Lưu địa chỉ → Kiểm tra API.
4. Mở tab Camera để xem luồng MJPEG từ laptop; tab Sự kiện lấy event từ backend.
5. Nếu điện thoại không truy cập được, kiểm tra Windows Firewall trên mạng Private và đảm bảo port 8000 được phép.

Với Android, nếu app bị chặn HTTP nội bộ, trong `mobile_app_runtime/android/app/src/main/AndroidManifest.xml` (được sinh bởi `flutter create`) thêm `android:usesCleartextTraffic="true"` vào thẻ `<application>`. Chỉ nên dùng HTTP trong mạng LAN tin cậy khi phát triển; khi triển khai thực tế, hãy thêm xác thực và HTTPS.

## 5. API

| Method | Endpoint | Mục đích |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/status` | Trạng thái engine, camera, device, FPS, dự đoán gần nhất |
| GET | `/api/events?limit=50` | Danh sách sự kiện được xác nhận |
| POST | `/api/events/{id}/ack` | Đánh dấu sự kiện đã xem |
| POST | `/api/camera/start` | Bật camera/AI worker |
| POST | `/api/camera/stop` | Dừng camera/AI worker |
| GET | `/stream.mjpg` | Luồng JPEG multipart có overlay AI |
| WS | `/ws` | Thông báo sự kiện mới (JSON) |

Sự kiện được lưu tại `backend/runtime/events.json`, file này bị loại khỏi Git.

## Hạn chế và bảo mật

- Backend MVP chưa có đăng nhập, token hay TLS. **Không mở port này trực tiếp ra Internet** và chỉ chạy trong mạng bạn tin cậy.
- Sự kiện `falling` là đầu ra của mô hình 8 lớp kết hợp một state machine hình học heuristic; cần đánh giá trên video có nhãn. Đây không phải thiết bị an toàn y tế.
- Chỉ một process backend nên sở hữu camera tại một thời điểm; dừng `scripts/run_demo.py` trước khi khởi động API nếu cả hai đang dùng cùng webcam.
- Nếu checkpoint thiếu/sai hoặc weights không tải được, `GET /api/status` sẽ trả `last_error`.
- Sự kiện lưu JSON phù hợp cho MVP; khi dùng nhiều người dùng/thiết bị, thay bằng SQLite/PostgreSQL và bổ sung authentication.
