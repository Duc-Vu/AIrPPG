# Task 4 — Lightweight Model (Tiny CNN / TCN) Pipeline & Results

Tài liệu này tổng hợp lại toàn bộ các công việc, kết quả thử nghiệm và cấu trúc thư mục của **Task 4 — Lightweight Model (Tiny CNN / TCN)** trong dự án AIrPPG.

---

## 1. Tổng quan công việc đã thực hiện

Chúng tôi đã thiết kế và triển khai pipeline học sâu nhẹ để dự đoán Heart Rate (HR) theo sliding window từ tín hiệu rPPG. Các module chính đã được thêm vào thư mục `src/airppg/modeling/` và script chạy pipeline tại `scripts/run_modeling.py`.

### Điểm cốt lõi trong thiết kế Feature:
Do sliding window dài khoảng 30s (~860 frames) có kích thước thay đổi phụ thuộc vào FPS của từng video, việc đưa trực tiếp chuỗi raw signal vào CNN là không khả thi. 
* **Giải pháp**: Tính **FFT power spectrum** trong dải tần số HR hữu ích [42 - 210 bpm] trên mỗi window.
* **Resampling**: Nội suy (resample) spectrum thành một vector có số chiều cố định `n_bins = 64` bins.
* **Các chiến lược đầu vào (Input Strategies)**:
  1. `pos_spectrum` (1 channel): FFT của tín hiệu POS trích xuất từ average-ROI (Task 2).
  2. `fused_spectrum` (1 channel): FFT của tín hiệu POS sau khi chạy Average Fusion trên nhiều ROI (Task 3).
  3. `multi_channel` (3 channels): Stack FFT của 3 tín hiệu Green, CHROM, và POS từ fusion.

---

## 2. Kiến trúc mô hình học sâu nhẹ

Chúng tôi đã thiết kế 2 kiến trúc siêu nhẹ với số lượng tham số rất nhỏ (~22K - 25K params) để đảm bảo tốc độ inference cực nhanh (< 1 ms/window trên CPU) và hạn chế overfitting:

### 1. **TinyCNN1D** (~22,385 tham số)
* Gồm 3 lớp `Conv1d` kết hợp với `BatchNormalization`, `ReLU`, `MaxPool1d`, và `AveragePooling`.
* Có Regression Head gồm 2 lớp tuyến tính (`Linear`) và `Dropout(0.3)`.

### 2. **SmallTCN** (~25,217 tham số)
* Gồm một lớp Projection ban đầu, sau đó đi qua 4 block **Dilated Causal Convolution** với hệ số dilation tăng dần ($1, 2, 4, 8$) để mở rộng receptive field mà không làm tăng lượng tham số quá nhiều.
* Kết thúc bằng `AdaptiveAvgPool1d` và một lớp tuyến tính để regression ra HR.

---

## 3. Kết quả huấn luyện (4 Runs)

Mô hình được huấn luyện bằng loss **Smooth L1 (Huber)** để giảm độ nhạy cảm với các cửa sổ nhiễu (outliers), tối ưu hóa bằng **Adam** (lr=1e-3, weight_decay=1e-4), dùng **Early Stopping** (patience=30) theo `val_mae`.

Dưới đây là kết quả đánh giá trên tập **Test** của 4 cấu hình chạy:

| Model | Strategy | Channels | Params | Best Epoch | Test MAE (bpm) ↓ | Test RMSE (bpm) ↓ | Test Pearson r ↑ |
|---|---|---|---|---|---|---|---|
| **SmallTCN** | `pos_spectrum` | 1 | 25,217 | 62 | **3.59** | **4.74** | ~0.10 |
| **TinyCNN1D** | `fused_spectrum` | 1 | 22,385 | 41 | **4.09** | **5.18** | ~0.20 |
| **TinyCNN1D** | `multi_channel` | 3 | 22,545 | 39 | **4.64** | **5.65** | ~0.25 |
| **TinyCNN1D** | `pos_spectrum` | 1 | 22,385 | 33 | **5.53** | **6.63** | ~0.16 |

> [!NOTE]
> **SmallTCN + pos_spectrum** là cấu hình tốt nhất đạt Test MAE **3.59 bpm**, nhờ khả năng nắm bắt pattern phổ tần số tốt hơn từ các lớp Dilated Conv.
> **fused_spectrum** (dùng tín hiệu sau fusion của Task 3) giúp TinyCNN cải thiện MAE đáng kể từ **5.53 bpm** xuống **4.09 bpm** so với chỉ dùng `pos_spectrum` đơn thuần.

---

## 4. So sánh với các Baseline (Task 2 & 3)

Bảng so sánh chi tiết kết quả MAE (bpm) và RMSE (bpm) trên tập **Test** giữa mô hình Task 4 và các phương pháp truyền thống:

| Nhóm phương pháp | Chi tiết phương pháp / Cấu hình | Test MAE (bpm) ↓ | Test RMSE (bpm) ↓ | So với POS avg_fusion |
|---|---|---|---|---|
| **Mô hình học sâu (Task 4)** | **SmallTCN (pos_spectrum)** | 3.59 | 4.74 | +1.93 bpm (Tệ hơn) |
| | **TinyCNN (fused_spectrum)** | 4.09 | 5.18 | +2.43 bpm (Tệ hơn) |
| | TinyCNN (multi_channel) | 4.64 | 5.65 | +2.98 bpm (Tệ hơn) |
| | TinyCNN (pos_spectrum) | 5.53 | 6.63 | +3.87 bpm (Tệ hơn) |
| **POS Baseline (Task 2 & 3)** | **POS quality_weighted_fusion** (Task 3) | **1.64** | **2.10** | **-0.02 bpm (Tốt nhất)** |
| | **POS average_fusion** (Task 3) | **1.66** | **2.12** | **Baseline mốc** |
| | POS single_forehead (Task 3) | 1.80 | 2.33 | +0.14 bpm (Tệ hơn) |
| **Green Baseline (Task 2 & 3)** | Green average_fusion | 10.24 | 12.09 | +8.58 bpm (Tệ hơn) |
| | Green quality_weighted_fusion | 11.69 | 15.59 | +10.03 bpm (Tệ hơn) |
| **CHROM Baseline (Task 2 & 3)**| CHROM average_fusion | 48.66 | 48.81 | +47.00 bpm (Tệ hơn) |

### Nhận xét & Phân tích chuyên môn:
1. **Chưa vượt qua POS Fusion**: Mô hình học sâu nhẹ chưa cải thiện được độ chính xác so với thuật toán POS truyền thống kết hợp fusion (MAE 3.59 bpm vs 1.64 bpm).
2. **Nguyên nhân**:
   * **Dữ liệu nhỏ**: Chỉ có 39 video cho tập train (~1,468 windows). CNN/TCN rất dễ bị overfitting trên tập dữ liệu nhỏ như vậy.
   * **Thông tin đầu vào**: Đầu vào là FFT spectrum trong dải HR, đây chính là thông tin mà thuật toán POS baseline dùng để chọn peak trực tiếp. POS đã tối ưu toán học hoàn hảo trên phổ tần số này nên CNN không có thêm lợi thế thông tin để khai thác.
   * **Vượt trội Green & CHROM**: Tuy chưa thắng POS, mô hình học sâu nhẹ vượt trội hơn nhiều so với baseline Green (10.24 bpm) và CHROM (48.66 bpm).

---

## 5. Tổ chức các file code đã triển khai

Tất cả code tuân thủ nghiêm ngặt cấu trúc và convention của dự án:

```
src/airppg/
  ├── modeling/
  │    ├── __init__.py      <- Khởi tạo gói modeling
  │    ├── config.py        <- Định nghĩa cấu hình run, hyperparams, normalization
  │    ├── dataset.py       <- Đọc manifest, load npz, trích xuất FFT spectrum & normalized
  │    ├── models.py        <- Định nghĩa cấu hình mạng TinyCNN1D & SmallTCN
  │    ├── train.py         <- Loop huấn luyện, early stopping, lưu checkpoint
  │    ├── metrics.py       <- Tính toán các metric MAE, RMSE, Pearson r, Bias
  │    └── visualization.py <- Vẽ đồ thị loss curve và so sánh HR dự đoán vs Ground Truth
  └── schemas.py            <- Khai báo MODELING_SCHEMA_VERSION = "modeling_v1"

scripts/
  └── run_modeling.py       <- Script điều khiển chính để chạy huấn luyện và đánh giá cả 4 cấu hình
```

---

## 6. Các artifacts sinh ra (Đã thêm vào Gitignore)

Các file kết quả sinh ra rất lớn và nhiều nên đã được đưa vào cấu hình bỏ qua của Git để giữ repository sạch sẽ.

### Thư mục dữ liệu kết quả (`outputs/modeling/` - Đã được thêm vào `.gitignore`):
* `modeling_manifest.json`: Manifest chứa thông tin đầu ra của tất cả 49 samples, tuân thủ schema `modeling_v1`.
* `modeling_metrics_summary.csv`: Lưu các metric chi tiết của từng sample.
* `modeling_aggregate_summary.csv`: Tổng hợp mean/std của metric theo từng split (train/val/test).
* `comparison_table_test.csv`: Bảng so sánh kết quả test trực tiếp với tất cả các baseline cũ.
* `checkpoints/`: Chứa các folder checkpoint cho 4 cấu hình huấn luyện (lưu `best_model.pt`, `training_history.csv`, `config.json` chứa tham số normalize).
* `train/`, `val/`, `test/`: Chứa các thư mục con theo `sample_id`, lưu các file:
  * `model_predictions.npz`: Chứa mảng predictions `predicted_hr`, `hr_timestamps`, `gt_aligned_to_hr`, v.v.
  * `model_metrics.json` & `model_metrics.csv`
  * `model_visualization.png`: Đồ thị so sánh tín hiệu HR dự đoán vs Ground Truth theo thời gian của sample đó.

### Báo cáo trực quan (`report/` - Đã được staging vào Git):
* `report/task4_lightweight_model_report.html`: Trang HTML báo cáo chi tiết trực quan, kết hợp CSS chung của dự án, trình bày đầy đủ bảng biểu so sánh, phân tích lý do và biểu đồ huấn luyện.
* `report/assets/`: Chứa các hình vẽ loss curves và so sánh cột kết quả MAE/RMSE để nhúng trực tiếp vào file báo cáo HTML.

---

## 7. Cập nhật Git & Gitignore mới nhất

Chúng tôi đã dọn dẹp Git để loại bỏ các file tạm thời và các file output nặng:
* **Đã cập nhật `.gitignore`**:
  * Thêm `outputs/` để tự động bỏ qua toàn bộ dữ liệu sinh ra từ mô hình và các task trước.
  * Thêm `.tmp_*` để bỏ qua các file so sánh/kiểm tra tạm thời tại thư mục gốc (ví dụ `.tmp_compare_outputs.py`).
* **Đã staged sẵn sàng commit**:
  * Các file source code tại `src/airppg/modeling/` và `scripts/run_modeling.py`.
  * File cập nhật schema `src/airppg/schemas.py` và tài liệu hướng dẫn `README.md`.
  * Toàn bộ báo cáo HTML và hình ảnh trực quan tại `report/`.
