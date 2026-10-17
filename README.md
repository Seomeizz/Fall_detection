# Nhận diện hành động từ bộ xương người bằng Transformer — NTU RGB+D 60

Dự án nghiên cứu thử nghiệm nhận diện **8 nhóm hành động** từ chuỗi keypoints 2D của cơ thể. Khi chạy với camera laptop hoặc video MP4, hệ thống dùng YOLO Pose để trích xuất keypoints, BoT-SORT để theo dõi từng người và Temporal Transformer để dự đoán hành động. Chương trình có thêm bộ quy tắc theo thời gian và hình học để tạo cảnh báo té ngã; các ngưỡng cảnh báo cần được kiểm chứng trên video có nhãn.

> **Phạm vi:** đây là bài toán nhận diện hành động 8 lớp có lớp `falling`, không phải một mô hình nhị phân chuyên dụng `té ngã/không té ngã`, cũng không phải thiết bị an toàn đã được chứng nhận.

## 1. Luồng xử lý

```text
Camera laptop hoặc video MP4
    ↓
YOLO Pose: trích xuất 17 keypoints COCO-17
    ↓
BoT-SORT: theo dõi từng người qua các frame
    ↓
Lấy đoạn video gần nhất của mỗi track
    ↓
Lấy mẫu đều thành 30 frame
    ↓
Nội suy keypoints thiếu và chuẩn hóa tọa độ
    ↓
51 đặc trưng/frame: 34 tọa độ XY + 17 confidence
    ↓
Temporal Transformer
    ↓
Dự đoán một trong 8 nhóm hành động
    ↓
Bộ quy tắc thời gian/hình học hỗ trợ cảnh báo té ngã
    ↓
Xuất video MP4 đã chú thích
```

## 2. Tám nhóm nhãn

Notebook ánh xạ một số action ID của NTU RGB+D 60 vào các nhóm do dự án định nghĩa:

| Nhãn dự án | Action ID NTU |
|---|---|
| `walking` | A059, A060 |
| `standing` | A001, A002, A003, A004, A010, A014, A015, A018, A019, A020, A021, A022, A023, A028, A031, A032, A033, A035, A036, A038, A039, A040, A041, A049, A056, A058 |
| `sitting` | A008, A011, A012, A029, A030 |
| `bending` | A005, A006, A013, A025, A037 |
| `falling` | A043 |
| `getting_up` | A009 |
| `stumbling` | A042 |
| `abnormal_movement` | A044–A048 |

Đây là **các nhóm hành động được gộp cho dự án**, không phải 8 lớp tiêu chuẩn có sẵn trong NTU. Một số nhóm hiện tại có thể không khớp hoàn toàn về ý nghĩa với tên gọi. Ví dụ, `standing` bao gồm cả các hành động như uống nước, ăn, đánh răng hoặc vỗ tay; `bending` cũng chứa một số hoạt động không nhất thiết là cúi người. Cần nêu rõ giới hạn này trong báo cáo và xem xét điều chỉnh mapping nếu cần ngữ nghĩa nhất quán hơn.

## 3. Kiến trúc Transformer

- Đầu vào mỗi clip: `30 × 51`.
- Mỗi frame: 34 giá trị tọa độ XY đã chuẩn hóa và 17 giá trị confidence.
- Input projection: `51 → 128`, LayerNorm, ReLU và Dropout `0.15`.
- Positional encoding: sinusoidal.
- Transformer Encoder: 4 lớp, 4 attention heads, `d_model=128`, feed-forward dimension `256`.
- Kích hoạt encoder: GELU; dùng pre-norm.
- Tổng hợp thời gian: mean pooling.
- Classification head: `128 → 64 → 8`.

Transformer là model được chọn cho demo triển khai. Notebook đi kèm có thể so sánh Transformer với BiLSTM, BiGRU và TCN; hãy căn cứ vào kết quả validation/test thực tế, không mặc định Transformer luôn tốt nhất.

## 4. Dữ liệu và huấn luyện

Notebook huấn luyện: `notebooks/train_compare_8class.ipynb`.

Các bước cơ bản:

1. Mở notebook trên Kaggle hoặc môi trường phù hợp.
2. Thêm dữ liệu keypoints NTU RGB+D 60 có quyền sử dụng, hoặc thiết lập nguồn tải được phép theo điều khoản dataset.
3. Chạy các cell từ đầu đến cuối.
4. Notebook lưu checkpoint, cấu hình thí nghiệm, metric và biểu đồ dưới thư mục `/kaggle/working/ntu60_8class_cliplevel_comparison/`.
5. Tải `best_transformer_8class.pth` về máy và đặt trong thư mục `artifacts/`.

Mỗi clip nguồn là một mẫu đánh giá; kết quả chính được báo cáo ở cấp clip, không phải cấp window. Notebook báo cáo Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1, Precision/Recall/F1 của lớp `falling`, báo cáo theo từng lớp và confusion matrix.

## 5. Cài đặt trên máy cá nhân

Khuyến nghị Python 3.10 hoặc 3.11 và môi trường ảo. Hãy cài PyTorch phù hợp với hệ điều hành và CPU/CUDA của máy trước, theo hướng dẫn chính thức tại [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

### Windows PowerShell

Chạy tại thư mục gốc của repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Nếu cần, cài dependencies bằng `python -m pip install -r requirements.txt` sau khi cài PyTorch.

Lần đầu chạy có thể tải weights YOLO Pose tự động. Ví dụ bên dưới dùng `yolo11n-pose.pt` để ưu tiên tốc độ; có thể thử một biến thể lớn hơn nếu phần cứng đáp ứng và bạn muốn so sánh chất lượng.

## 6. Chạy thử video MP4

Đặt checkpoint tại `artifacts/best_transformer_8class.pth`, sau đó chạy lệnh từ thư mục gốc repository:

```powershell
python scripts/run_demo.py --video "tests/test3.mp4" --checkpoint "artifacts/best_transformer_8class.pth" --pose-weights yolo11n-pose.pt --imgsz 640 --output "output/test3_result.mp4"
```

Thay `tests/test3.mp4` bằng đường dẫn video của bạn. Video kết quả được ghi vào `output/test3_result.mp4`. Chương trình chỉ xuất MP4, không xuất CSV.

## 7. Chạy camera laptop

```powershell
python scripts/run_demo.py --camera 0 --checkpoint "artifacts/best_transformer_8class.pth" --pose-weights yolo11n-pose.pt --imgsz 640 --output "output/camera_result.mp4"
```

- Nếu có nhiều camera, thử `--camera 1`.
- Camera được lật ngang mặc định để hiển thị như gương; thêm `--no-mirror-camera` nếu không muốn lật.
- Nhấn `q` trong cửa sổ video để dừng.
- Thời gian đầu chương trình cần thu thập đủ đoạn gần nhất (mặc định khoảng 2,5 giây) trước khi có dự đoán ổn định.

### Một số tham số

- `--checkpoint`: đường dẫn checkpoint Transformer 8 lớp.
- `--pose-weights`: weights YOLO Pose.
- `--imgsz`: kích thước ảnh đầu vào YOLO; số lớn có thể cải thiện phát hiện người nhỏ nhưng chậm hơn.
- `--clip-seconds`: thời lượng đoạn gần nhất mỗi track được lấy để phân loại.
- `--stride-frames`: số frame giữa hai lần suy luận Transformer; YOLO vẫn xử lý từng frame để tracking.
- `--no-display`: xử lý video không mở cửa sổ preview.

## 8. Kiểm tra cơ bản

Từ thư mục gốc repository, chạy:

```powershell
python -m pytest tests -q
```

Các kiểm thử cơ bản không thay thế cho việc đánh giá thực nghiệm trên video. Cần thử nhiều điều kiện ánh sáng, khoảng cách, góc camera, người bị che khuất và hành động dễ gây nhầm lẫn.

## 9. Giới hạn hiện tại

- Model được huấn luyện trên keypoints của dataset, trong khi inference trực tiếp dùng keypoints do YOLO Pose tạo ra; chất lượng hai nguồn có thể khác nhau.
- Lớp `falling` hiện ánh xạ tới action A043 của NTU; điều đó không bao phủ mọi dạng té ngã trong đời thực.
- Lấy mẫu đều 30 frame cho mỗi clip dài có thể bỏ qua chuyển động té ngã ngắn.
- Demo trực tiếp dùng một đoạn gần nhất dạng rolling clip; kết quả online có thể khác điểm đánh giá offline trên clip NTU hoàn chỉnh.
- Khung đỏ/cảnh báo có kết hợp quy tắc hình học và thời gian. Các ngưỡng này chưa được xác nhận rộng rãi trên video có nhãn, nên có thể tạo cảnh báo sai hoặc bỏ sót.
- Kết quả trên NTU không tự động chứng minh hiệu quả trên camera laptop. Để báo cáo khả năng thực tế, cần có tập video độc lập có nhãn và đo Recall sự kiện té ngã, cảnh báo nhầm, độ trễ và FPS.
- Đây là prototype nghiên cứu, không thay thế giám sát trực tiếp hoặc thiết bị an toàn được chứng nhận.

## 10. Dữ liệu, weights và quyền sử dụng

Repository không kèm dataset NTU, checkpoint Transformer hoặc weights YOLO mặc định. Người dùng cần tự chuẩn bị các file theo quyền sử dụng và hướng dẫn tương ứng.

Trước khi public, hãy đọc điều khoản chính thức của NTU RGB+D và Ultralytics YOLO. Không tải lên dataset, dữ liệu keypoints phái sinh, video có thể nhận dạng người, checkpoint hoặc weights nếu bạn chưa xác minh quyền phân phối. Không chọn giấy phép cho toàn bộ repository một cách máy móc khi các thành phần bên thứ ba có điều khoản riêng. Xem thêm `LICENSE_NOTE.md`.

## 11. Cấu trúc repository

```text
configs/                       Cấu hình model và tiền xử lý
notebooks/                     Notebook huấn luyện và so sánh
src/fall_detection/             Nhãn, Transformer và tiền xử lý dùng chung
scripts/run_demo.py             Chạy camera laptop hoặc video MP4
tests/                           Kiểm thử cơ bản
docs/                            Ghi chú dữ liệu và đánh giá
reports/                         Nơi lưu metric và biểu đồ đã tạo
artifacts/                       Checkpoint cục bộ (được .gitignore loại trừ)
assets/                          Ảnh/GIF minh họa dự án
```

## Trạng thái dự án

Đây là prototype đang phát triển. Các số liệu trong báo cáo cần được tạo từ lần chạy thực tế và không nên suy diễn hiệu quả trên camera từ kết quả dataset. Phần cảnh báo té ngã cần được kiểm thử độc lập trước khi dùng trong tình huống thực tế.
