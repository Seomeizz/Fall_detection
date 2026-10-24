# Ban va luu dia chi backend FallGuard tren dien thoai

Ban nay cap nhat app Flutter de:
- Luu dia chi backend tren dien thoai khi bam **Luu dia chi**.
- Tu dong luu lai dia chi neu **Kiem tra API** ket noi thanh cong.
- Khi mo app lai, tu nap dia chi da luu va thu ket noi backend.
- Khi backend chua chay, app giu dia chi cu va tu kiem tra lai moi 5 giay.

## Cach ap dung

1. Sao luu file cu `mobile_app/lib/main.dart` va `mobile_app/pubspec.yaml`.
2. Giai nen ZIP nay vao thu muc goc repository, de ghi de dung hai file:
   - `mobile_app/lib/main.dart`
   - `mobile_app/pubspec.yaml`
3. Mo terminal tai thu muc `mobile_app`, chay:

   ```cmd
   flutter pub get
   flutter analyze
   flutter build apk
   ```

4. Cai APK moi len dien thoai. Neu chi hot-reload tu app dang chay, dong app va mo lai; thay doi SharedPreferences can rebuild/cai lai app de su dung ban moi.
5. Lan dau vao **Cai dat**, nhap URL nhu `http://192.168.1.71:8000`, sau do bam **Kiem tra API**. Neu ket noi thanh cong, URL se duoc luu. Tu lan sau app se tu dien va thu ket noi.

Luu y: dia chi IP laptop co the thay doi neu router cap IP dong. Neu IP thay doi, cap nhat lai URL trong muc Cai dat.
