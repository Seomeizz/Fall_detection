# Chạy giao diện FallGuard AI trên điện thoại trước

Các bước dưới đây chỉ chạy giao diện Flutter; **không cần khởi động FastAPI, camera hoặc checkpoint AI**. App sẽ hiển thị trạng thái offline và placeholder ở trang camera cho đến khi kết nối backend.

## 1. Chuẩn bị điện thoại Android

- Cài Flutter SDK trên máy tính và thêm `flutter\bin` vào `PATH`.
- Cài Android Studio / Android SDK hoặc cấu hình SDK tương đương.
- Trên điện thoại: Cài đặt → Giới thiệu điện thoại → chạm Số bản dựng 7 lần; vào Developer options và bật USB debugging.
- Kết nối USB, mở khóa điện thoại và bấm Allow nếu xuất hiện lời hỏi tin cậy máy tính.

Kiểm tra trong PowerShell:

```powershell
flutter doctor -v
flutter devices
```

Phải thấy tên điện thoại Android (ví dụ Samsung Galaxy A22) trong danh sách.

## 2. Tạo và chạy app

Từ thư mục gốc dự án:

```powershell
powershell -ExecutionPolicy Bypass -File .\mobile_app\run_on_android.ps1
```

Lần đầu, script tạo thư mục `mobile_app_runtime`, copy source Flutter từ `mobile_app/`, bật quyền Internet/HTTP nội bộ trong AndroidManifest, chạy `flutter pub get` và cuối cùng `flutter run`.

Nếu Flutter hỏi chọn thiết bị, chọn ID điện thoại. App sẽ cài đặt và mở trên máy.

## 3. Điều gì sẽ xuất hiện?

- Tab **Tổng quan**: thẻ trạng thái backend và nút mở trang camera.
- Tab **Camera**: vùng video với thông báo chờ luồng video.
- Tab **Sự kiện**: danh sách sự kiện lấy từ backend; khi backend chưa chạy, giao diện báo chưa tải được dữ liệu.
- Tab **Cài đặt**: nhập địa chỉ FastAPI sau khi đã khởi động backend.

Trạng thái offline hiện tại là đúng vì backend chưa được chạy. Mục tiêu của bước này là xác nhận giao diện, navigation và app chạy được trên điện thoại.

## 4. Khi sẵn sàng kết nối camera thật

1. Chạy FastAPI trên laptop theo `backend/README.md`.
2. Laptop và điện thoại kết nối cùng Wi-Fi.
3. Dùng `ipconfig` lấy IPv4 của laptop.
4. Trong app → Cài đặt, nhập `http://<IPv4-laptop>:8000`, bấm lưu và kiểm tra API.
5. Mở tab Camera để xem `/stream.mjpg` và tab Sự kiện để theo dõi cảnh báo.

Không dùng `localhost` trên điện thoại. HTTP cleartext trong script chỉ dành cho phát triển trong mạng nội bộ tin cậy; không mở port này trực tiếp ra Internet.
