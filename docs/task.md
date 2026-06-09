# Kế hoạch triển khai đề tài AIrPPG

## 0. Tóm tắt đề tài

**Tên đề tài:** Lightweight Multi-ROI rPPG for Heart Rate Estimation from Webcam Videos / Đo nhịp tim từ video webcam bằng rPPG đa vùng khuôn mặt nhẹ.

**Mục tiêu:** xây dựng pipeline rPPG nhẹ để ước lượng nhịp tim từ video khuôn mặt RGB/webcam, dùng nhiều ROI trên mặt gồm trán, má trái và má phải; sau đó so sánh baseline, fusion đa ROI, mô hình nhẹ và confidence score.

**Dataset chính:** UBFC-rPPG. Dataset không commit vào repo.

**Nguyên tắc chung:**

- Dùng Python 3.11 và `uv` để quản lý môi trường.
- Không hardcode đường dẫn dataset cá nhân trong file commit.
- Giữ nguyên timeline video: không drop frame lỗi một cách âm thầm.
- Frame lỗi phải được đánh dấu invalid và có lý do lỗi rõ ràng.
- Artifact trung gian để trong thư mục local bị ignore như `outputs/` hoặc `artifacts/`.

## 1. Trạng thái hiện tại

Task đang làm: **Task 1 - Video preprocessing với face landmark tracking và ROI extraction**.

Notebook hiện có: `notebooks/task1_video_preprocessing.ipynb`.

Notebook hiện tại đã có các phần chính:

- cấu hình `VIDEO_PATH`, `OUTPUT_DIR`, `RUN_NAME`, `CONFIG`;
- đọc metadata video bằng OpenCV;
- resize frame nếu cấu hình `resize_width`;
- sample FPS nếu cấu hình `target_fps`;
- chạy MediaPipe Face Mesh với `refine_landmarks=True`;
- chuyển landmark sang pixel coordinates;
- định nghĩa 3 ROI: `forehead`, `left_cheek`, `right_cheek`;
- tạo ROI polygon và binary mask theo từng frame;
- lưu `.npz` gồm `frame_indices`, `landmarks_px`, `roi_polygons_px`, `roi_masks`, `valid`, `error_reasons`, `roi_names`;
- lưu `.json` metadata gồm schema version, config, ROI landmark indices, số frame valid/invalid và thống kê lỗi;
- lưu PNG overlay samples; overlay MP4 chỉ bật khi `save_overlay_video=True`.

## 2. Task 1 - Video preprocessing, Face Mesh tracking, Multi-ROI extraction

### 2.1. Mục tiêu

Xây dựng pipeline tiền xử lý video đầu vào để tạo artifact ổn định cho Task 2. Pipeline phải đọc video, giữ được chỉ số frame gốc, phát hiện landmark khuôn mặt, trích xuất ROI trán/má trái/má phải và lưu dữ liệu trung gian đủ để trích xuất tín hiệu RGB theo thời gian.

Task 1 **không** ước lượng nhịp tim. Task 1 chỉ tạo dữ liệu sạch và có kiểm soát lỗi cho các task sau.

### 2.2. Input

| Nhóm | Yêu cầu |
| --- | --- |
| Video | Một file video local, ví dụ video trong UBFC-rPPG. |
| Cấu hình đường dẫn | `VIDEO_PATH` lấy từ biến môi trường hoặc chỉnh trong notebook khi chạy local. Không commit đường dẫn tuyệt đối theo máy cá nhân. |
| Metadata cần đọc | FPS gốc, số frame, width, height, duration. |
| Công cụ | OpenCV để đọc frame/video; MediaPipe Face Mesh để lấy landmark. |
| ROI cần trích xuất | `forehead`, `left_cheek`, `right_cheek`. |

### 2.3. Cấu hình pipeline

Các config cần có và ý nghĩa:

| Config | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `target_fps` | `float \| None` | Nếu `None`, xử lý theo FPS gốc. Nếu có giá trị, sample xuống FPS mục tiêu nhưng vẫn lưu `frame_indices` gốc. |
| `resize_width` | `int \| None` | Nếu có giá trị, resize frame theo width này và giữ aspect ratio. |
| `max_frames` | `int \| None` | Giới hạn số frame khi debug. Không dùng để đánh giá chính thức. |
| `save_overlay_video` | `bool` | Chỉ lưu MP4 overlay khi bật rõ ràng. Mặc định `False`. |
| `overlay_sample_count` | `int` | Số PNG overlay sample cần lưu để kiểm tra ROI trực quan. |
| `min_roi_area_px` | `float` | Diện tích tối thiểu của ROI. ROI nhỏ hơn ngưỡng bị đánh dấu lỗi. |
| `schema_version` | `str` | Version schema artifact, ví dụ `task1_roi_v1`. |

### 2.4. Các bước xử lý bắt buộc

1. **Mở video và đọc metadata**
   - Kiểm tra video mở được bằng `cv2.VideoCapture`.
   - Lưu FPS, frame count, width, height, duration.
   - Nếu không mở được video, báo lỗi rõ ràng.

2. **Duyệt frame theo thứ tự gốc**
   - Mỗi frame đọc từ video có `frame_idx` gốc.
   - Nếu sample FPS, chỉ xử lý các frame được chọn nhưng phải lưu `frame_indices` để biết frame gốc nào đã được xử lý.
   - Không đảo thứ tự frame.

3. **Resize nếu cần**
   - Chỉ resize khi `resize_width` khác `None`.
   - Giữ aspect ratio.
   - Landmark và ROI được tính trên frame sau resize; metadata gốc vẫn phải được lưu trong JSON.

4. **Detect/track Face Mesh landmark**
   - Dùng MediaPipe Face Mesh với `max_num_faces=1`.
   - Landmark lưu dạng pixel coordinates `(x, y)`.
   - Shape khuyến nghị: `(num_processed_frames, 478, 2)` khi dùng `refine_landmarks=True`.

5. **Tạo ROI polygon**
   - Dùng landmark indices cố định cho 3 ROI:
     - `forehead`
     - `left_cheek`
     - `right_cheek`
   - Clip polygon vào biên frame.
   - Tính diện tích polygon; nếu dưới `min_roi_area_px`, đánh dấu lỗi `unstable_roi:<roi_name>`.

6. **Tạo ROI mask**
   - Tạo binary mask cho từng ROI trên frame đã xử lý.
   - Shape khuyến nghị: `(num_processed_frames, 3, height, width)` nếu toàn bộ output có cùng kích thước.
   - Mask dùng `uint8`, giá trị 0/1.

7. **Xử lý lỗi theo frame**
   - Không drop frame lỗi âm thầm.
   - Với mỗi frame đã được chọn xử lý, lưu:
     - `valid=True/False`;
     - `error_reasons`, ví dụ `no_face`, `invalid_landmarks`, `unstable_roi:forehead`.
   - Nếu một frame lỗi, arrays tương ứng nên chứa `NaN` cho landmark/polygon không hợp lệ và mask rỗng.

8. **Lưu artifact**
   - Lưu `.npz` cho arrays lớn.
   - Lưu `.json` cho metadata, config, schema, thống kê lỗi và danh sách overlay.
   - Lưu PNG overlay samples mặc định.
   - Không lưu overlay MP4 trừ khi `save_overlay_video=True`.

### 2.5. Output bắt buộc

#### 2.5.1. File `.npz`

Tên gợi ý: `outputs/task1_preprocessing/<run_name>_roi_data.npz`.

Các field bắt buộc:

| Field | Shape / dtype | Ý nghĩa |
| --- | --- | --- |
| `frame_indices` | `(N,)`, `int32` | Chỉ số frame gốc đã được xử lý. |
| `landmarks_px` | `(N, 478, 2)`, `float32` | Landmark pixel coordinates; frame lỗi dùng `NaN` khi không có landmark hợp lệ. |
| `roi_polygons_px` | `(N, 3, P, 2)`, `float32` | Polygon ROI theo pixel; `P` là số điểm lớn nhất sau padding; điểm padding hoặc ROI lỗi dùng `NaN`. |
| `roi_masks` | `(N, 3, H, W)`, `uint8` | Binary mask của từng ROI. |
| `valid` | `(N,)`, `bool` | Frame có đủ landmark và ROI hợp lệ hay không. |
| `error_reasons` | `(N,)`, object/string | Lý do lỗi theo frame; chuỗi rỗng nếu valid. |
| `roi_names` | `(3,)` | Thứ tự ROI, phải khớp trục thứ 2 của `roi_polygons_px` và `roi_masks`. |

#### 2.5.2. File `.json`

Tên gợi ý: `outputs/task1_preprocessing/<run_name>_metadata.json`.

Các field bắt buộc:

| Field | Ý nghĩa |
| --- | --- |
| `schema_version` | Version schema artifact. |
| `video` | Metadata video gốc: path, fps, frame_count, width, height, duration_seconds. |
| `config` | Config đã dùng để chạy. |
| `roi_names` | Danh sách ROI theo thứ tự. |
| `roi_landmarks` | Landmark indices dùng để tạo từng ROI. |
| `processed_frame_count` | Số frame đã xử lý sau sampling. |
| `valid_frame_count` | Số frame hợp lệ. |
| `invalid_frame_count` | Số frame lỗi. |
| `failure_counts` | Thống kê số lần từng lỗi xuất hiện. |
| `overlay_samples` | Danh sách file PNG overlay đã lưu. |

#### 2.5.3. Visualization

- Lưu PNG overlay samples trong thư mục riêng theo `run_name`.
- Overlay phải vẽ polygon cho 3 ROI và trạng thái `valid`/`invalid`.
- Overlay MP4 là optional, chỉ sinh khi bật config.

### 2.6. Tiêu chí hoàn thành Task 1

Task 1 được xem là hoàn thành khi:

- Notebook chạy được từ đầu đến cuối khi `VIDEO_PATH` trỏ tới một video hợp lệ.
- Khi `VIDEO_PATH` chưa set hoặc file không tồn tại, notebook không crash ở cell chạy chính mà in hướng dẫn rõ ràng.
- `.npz` và `.json` được sinh đúng schema ở trên.
- `frame_indices` giữ được chỉ số frame gốc.
- Frame lỗi vẫn có record trong output đã xử lý và có `error_reasons` rõ ràng.
- Có PNG overlay samples để kiểm tra ROI bằng mắt.
- Artifact của Task 1 đủ để Task 2 đọc lại và tính RGB mean theo từng ROI theo thời gian.

### 2.7. Việc cần kiểm tra tiếp cho Task 1

- Chạy notebook với một video UBFC-rPPG thật hoặc một face video local.
- Kiểm tra overlay sample: ROI có nằm trên vùng da hợp lý không, đặc biệt trán và hai má.
- Kiểm tra shape/dtype của `.npz` sau khi chạy.
- Kiểm tra `failure_counts`: nếu `no_face` hoặc `unstable_roi` quá nhiều, cần xem lại landmark indices, resize, ngưỡng area hoặc chất lượng video.
- Kiểm tra dung lượng `roi_masks`; nếu quá lớn với video dài, cân nhắc lưu polygon trước và chỉ tạo mask khi Task 2 cần, nhưng phải giữ contract rõ ràng.

### 2.8. Artifact dùng cho Task 2

Task 2 sẽ dùng các output sau từ Task 1:

- `frame_indices` để đồng bộ timeline video;
- `roi_names` để biết thứ tự ROI;
- `roi_masks` hoặc `roi_polygons_px` để lấy pixel trong từng ROI;
- `valid` và `error_reasons` để tránh xử lý nhầm frame lỗi;
- metadata FPS để xây dựng trục thời gian và filter tín hiệu.

## 3. Task 2 - Signal extraction và baseline Green/CHROM/POS

### Mục tiêu

Từ artifact Task 1, trích xuất tín hiệu RGB theo thời gian cho từng ROI, tiền xử lý tín hiệu và triển khai baseline rPPG truyền thống.

### Input

- Frame sequence hoặc video gốc kèm `frame_indices` từ Task 1.
- `roi_masks` hoặc `roi_polygons_px` từ Task 1.
- `valid` và `error_reasons` từ Task 1.
- FPS hoặc effective FPS sau sampling.
- Ground truth pulse/heart rate từ UBFC-rPPG.

### Việc cần làm

- Tính mean RGB theo từng ROI và từng frame valid.
- Tạo chuỗi `R(t)`, `G(t)`, `B(t)` cho mỗi ROI.
- Xử lý frame invalid theo policy rõ ràng: mask khỏi window đánh giá hoặc nội suy có kiểm soát ở bước signal, không làm mất timeline.
- Tiền xử lý signal: normalization, detrending, bandpass filtering.
- Triển khai Green channel baseline.
- Triển khai CHROM.
- Triển khai POS.
- Ước lượng HR từ phổ tần số theo từng video hoặc từng sliding window.
- So sánh với ground truth.

### Output

- Per-ROI RGB signal.
- Per-ROI rPPG signal cho Green, CHROM, POS.
- HR prediction theo video/window.
- Metric: MAE, RMSE, Pearson correlation, SNR.
- Biểu đồ signal, spectrum và prediction vs ground truth.

## 4. Task 3 - Multi-ROI fusion

### Mục tiêu

So sánh single ROI, average fusion và quality-weighted fusion để kiểm tra lợi ích của multi-ROI.

### Input

- Per-ROI RGB/rPPG signal từ Task 2.
- HR prediction từ từng ROI.
- Ground truth HR.
- Quality features: SNR, peak clarity, peak stability, landmark jitter, inter-ROI disagreement.

### Việc cần làm

- Thiết kế single-ROI baseline.
- Thiết kế average fusion.
- Thiết kế quality-weighted fusion.
- Tính quality features cho từng ROI/window.
- Chuẩn hóa trọng số để tổng bằng 1.
- Phân tích trường hợp một ROI bị nhiễu, bị che hoặc ánh sáng kém.

### Output

- Fused rPPG signal.
- Weight của từng ROI theo window.
- HR prediction sau fusion.
- Bảng so sánh single ROI vs average fusion vs quality-weighted fusion.
- Biểu đồ weight và metric theo thời gian/window.

## 5. Task 4 - Mô hình học sâu nhẹ

### Mục tiêu

Thử nghiệm mô hình nhẹ như Tiny CNN hoặc TCN trên tín hiệu ROI/fused signal và so sánh với baseline truyền thống.

### Input

- ROI signal hoặc fused signal từ Task 2/3.
- Ground truth HR hoặc pulse signal.
- Training windows đã cắt theo protocol thống nhất.
- Baseline result từ Green, CHROM, POS.

### Việc cần làm

- Chuẩn bị dataset window-level.
- Thiết kế Tiny CNN hoặc TCN nhỏ.
- Huấn luyện và validation theo split rõ ràng.
- Theo dõi loss và dấu hiệu overfitting.
- So sánh với baseline cùng protocol.

### Output

- Model checkpoint nếu có huấn luyện.
- HR prediction từ mô hình nhẹ.
- Loss curve.
- Bảng metric so sánh với baseline.
- Nhận xét mô hình nhẹ có cải thiện accuracy/robustness không.

## 6. Task 5 - Confidence score

### Mục tiêu

Thiết kế confidence score để biết khi nào HR prediction đáng tin.

### Input

- HR prediction từ baseline/fusion/model.
- Ground truth HR để phân tích error.
- SNR, peak stability, motion score, inter-ROI disagreement.
- Error thực tế giữa prediction và ground truth.

### Việc cần làm

- Thiết kế rule-based confidence trước.
- Tính SNR và peak clarity/stability.
- Tính motion score từ landmark jitter hoặc ROI movement.
- Tính disagreement giữa các ROI.
- Kiểm tra tương quan giữa confidence thấp và error cao.
- Nếu dữ liệu đủ, thử mô hình nhỏ để dự đoán confidence hoặc predicted error.

### Output

- Công thức hoặc mô hình confidence.
- Confidence theo video/window.
- Bảng confidence-error.
- Biểu đồ confidence vs error.
- Ngưỡng cảnh báo khi prediction không đáng tin.

## 7. Task 6 - Tổng hợp kết quả và báo cáo

### Mục tiêu

Chuẩn hóa notebook, tổng hợp kết quả thực nghiệm và hoàn thiện báo cáo nghiên cứu.

### Input

- Notebook/script từ Task 1 đến Task 5.
- Bảng kết quả baseline, fusion, model nhẹ và confidence.
- Biểu đồ ROI, signal, spectrum, metric, loss curve.

### Việc cần làm

- Chuẩn hóa notebook để chạy lại được từ đầu đến cuối.
- Ghi rõ cách chuẩn bị dataset local và cách chạy từng bước.
- Tổng hợp kết quả theo cùng protocol đánh giá.
- Viết methodology, experiment setup, result, discussion, limitation và future work.
- Nêu rõ giới hạn: hệ thống không phải thiết bị y tế và không dùng để chẩn đoán.

### Output cuối

- Notebook chính chạy end-to-end.
- `results.csv` hoặc bảng kết quả tương đương.
- `figures/` hoặc thư mục hình minh họa.
- Report/proposal hoàn chỉnh.
- README hướng dẫn chạy lại thí nghiệm.
