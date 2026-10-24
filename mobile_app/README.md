# FallGuard AI — ứng dụng Flutter

Đây là source Flutter để xem trạng thái laptop AI, camera trực tiếp, danh sách sự kiện và cài đặt địa chỉ backend. App không chạy model AI trên điện thoại; laptop chạy YOLO Pose + BoT-SORT + Temporal Transformer 8 lớp.

## Chạy giao diện trên điện thoại trước (chưa cần backend)

1. Cài Flutter SDK và Android SDK trên máy tính, sau đó kiểm tra `flutter doctor -v`.
2. Trên điện thoại Android, bật Developer options và USB debugging, kết nối cáp USB và chọn Allow nếu điện thoại hỏi tin cậy máy tính.
3. Mở PowerShell tại thư mục gốc repository, chạy:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\mobile_app\run_on_android.ps1
   ```

4. Script sẽ tạo `mobile_app_runtime/` ở thư mục gốc, chép source app, lấy dependencies rồi chạy `flutter run`.
5. Nếu Flutter thấy nhiều thiết bị, chọn ID điện thoại. Nếu chưa thấy điện thoại, chạy `flutter devices` và `flutter doctor -v`.

Backend chưa chạy vẫn xem được toàn bộ giao diện. App sẽ báo “Chưa kết nối laptop” và hiển thị placeholder ở trang camera; đây là trạng thái bình thường, không phải lỗi giao diện.

## Chạy thủ công

Từ thư mục gốc repository:

```powershell
flutter create --platforms=android --project-name fall_detection_monitor mobile_app_runtime
Copy-Item mobile_app\lib\main.dart mobile_app_runtime\lib\main.dart -Force
Copy-Item mobile_app\pubspec.yaml mobile_app_runtime\pubspec.yaml -Force
cd mobile_app_runtime
flutter pub get
flutter devices
flutter run
```

Nếu `mobile_app_runtime/` đã được tạo, không cần chạy lại `flutter create`; chỉ copy source, chạy `flutter pub get`, rồi `flutter run`.

## Khi tích hợp backend

Khi backend FastAPI đã chạy, laptop và điện thoại phải cùng Wi-Fi. Lấy IPv4 laptop bằng `ipconfig`, sau đó trong app → **Cài đặt**, nhập `http://IPV4_CUA_LAPTOP:8000` và bấm **Kiểm tra API**. Không dùng `localhost` trên điện thoại.

`mobile_app_runtime/android/app/src/main/AndroidManifest.xml` cần có quyền Internet và `android:usesCleartextTraffic="true"` cho HTTP trong LAN khi phát triển. Script tự thiết lập hai mục đó. Chỉ dùng trong mạng tin cậy; không mở backend trực tiếp ra Internet. Firebase Cloud Messaging chưa được tích hợp.
