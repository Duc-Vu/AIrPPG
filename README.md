# AIrPPG

AIrPPG là dự án thực nghiệm rPPG đa ROI để ước lượng nhịp tim từ video RGB khuôn mặt. Pipeline hiện tại được tổ chức theo từng task:

- **Task 1 - Preprocessing**: đọc video UBFC-rPPG, detect Face Mesh, tạo ROI trán/má trái/má phải, giữ timeline và lưu artifact frame/ROI/ground truth.
- **Task 2 - Signal extraction**: đọc output Task 1, trích xuất RGB/rPPG theo ROI, chạy Green/CHROM/POS, align ground truth và tạo baseline metric.
- **Task 3 - Multi-ROI fusion**: đọc output Task 2, so sánh single ROI, average fusion và quality-weighted fusion; đây là task hiện đã có report chi tiết.
- **Task 4 - Lightweight model**: task kế tiếp, dùng signal/window từ Task 2/3 để huấn luyện mô hình nhẹ như Tiny CNN hoặc TCN và so sánh với baseline truyền thống/fusion.

Dataset chính: **UBFC-rPPG**. Dataset không được commit vào repository.

---

## 1. Cài đặt môi trường bằng `uv`

Dự án dùng **Python 3.11** và [`uv`](https://docs.astral.sh/uv/) để quản lý môi trường.

### Cài `uv` trên Windows PowerShell

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Sau khi cài xong, mở lại terminal và kiểm tra:

```powershell
uv --version
```

### Cài dependency của dự án

Từ thư mục gốc repository:

```powershell
uv sync
```

Nếu cần chạy test/dev tools:

```powershell
uv sync --extra dev
```

Nếu workspace nằm trong đường dẫn Unicode trên Windows và `uv run` không import được `airppg`, cài project dạng wheel thay vì editable:

```powershell
uv sync --extra dev --no-editable
```

### Mở Jupyter Lab

```powershell
uv run jupyter lab
```

Notebook tự tìm project root và thêm `src/` vào `sys.path`, nên có thể chạy khi Jupyter đang đứng ở repo root hoặc trực tiếp trong thư mục `notebooks/`.

Notebook chính:

```text
notebooks/01_preprocessing.ipynb
notebooks/02_signal_extraction.ipynb
notebooks/03_fusion.ipynb
```

---

## 2. Dataset UBFC-rPPG

Đặt dataset cục bộ dưới thư mục:

```text
datasets/UBFC_DATASET/
  DATASET_1/
    10-gt/
      vid.avi
      gtdump.xmp
    11-gt/
      vid.avi
      gtdump.xmp
    ...
  DATASET_2/
    subject1/
      vid.avi
      ground_truth.txt
    subject2/
      vid.avi
      ground_truth.txt
    ...
```

Các file dataset, frame trích xuất, video overlay và artifact trung gian nằm trong `datasets/` hoặc `outputs/` và không nên commit vào git.

## 3. Task 1 - Preprocessing

Task 1 tạo dữ liệu đầu vào ổn định cho toàn bộ pipeline. Notebook chính:

```text
notebooks/01_preprocessing.ipynb
```

Luồng xử lý tóm tắt:

1. Đọc video UBFC-rPPG và metadata: FPS, frame count, resolution, duration.
2. Detect Face Mesh landmarks bằng MediaPipe.
3. Tạo 3 ROI cố định: `forehead`, `left_cheek`, `right_cheek`.
4. Tạo polygon/mask ROI theo từng frame.
5. Giữ nguyên timeline: frame lỗi không bị drop, mà được đánh dấu `valid = False` và có `error_reasons`.
6. Chuẩn hóa UBFC ground truth thành artifact riêng để Task 2 không phải parse raw dataset.
7. Lưu overlay PNG để kiểm tra ROI bằng mắt.

Output chính:

```text
outputs/preprocessing/
  preprocessing_dataset_manifest.json
  train|val|test/
    <sample_id>/
      roi_data.npz
      metadata.json
      ground_truth.npz
      overlay_samples/
```

Các key quan trọng:

| File | Key/trường | Ý nghĩa |
| --- | --- | --- |
| `preprocessing_dataset_manifest.json` | `items[*].outputs` | Entrypoint cho Task 2. |
| `roi_data.npz` | `frame_indices`, `landmarks_px`, `roi_polygons_px`, `roi_masks`, `valid`, `error_reasons`, `roi_names` | ROI/frame-level artifact. |
| `metadata.json` | video metadata, config, ROI names, failure counts | Audit trail của preprocessing. |
| `ground_truth.npz` | `timestamps`, `heart_rate_bpm`, `ppg_signal`, `ppg_timestamps`, `spo2_percent` | Ground truth đã chuẩn hóa. |

Task 1 hoàn thành khi toàn bộ sample xử lý được, manifest dùng path tương đối, ROI overlay nhìn hợp lý, và frame lỗi vẫn có record rõ ràng.

---

## 4. Task 2 - Signal extraction và classical baselines

Task 2 đọc output Task 1, trích xuất tín hiệu RGB/rPPG theo ROI, chạy Green/CHROM/POS và tạo baseline metric cho Task 3. Notebook chính:

```text
notebooks/02_signal_extraction.ipynb
```

Luồng xử lý tóm tắt:

1. Đọc `outputs/preprocessing/preprocessing_dataset_manifest.json`.
2. Với mỗi sample `processed`, load `roi_data.npz`, `metadata.json`, `ground_truth.npz` và video nguồn.
3. Tính mean RGB theo `forehead`, `left_cheek`, `right_cheek`.
4. Giữ timeline: frame invalid vẫn tồn tại, signal lỗi là `NaN`.
5. Tiền xử lý signal: fill/interpolate NaN khi cần, detrend, bandpass HR band, z-score.
6. Chạy baseline Green, CHROM, POS.
7. Ước lượng HR theo sliding FFT window, align với ground truth, tính metric.
8. Lưu artifact và visualization.

Output chính:

```text
outputs/signal_extraction/
  signal_extraction_dataset_manifest.json
  signal_extraction_metrics_summary.csv
  train|val|test/
    <sample_id>/
      signals.npz
      metadata.json
      metrics.json
      metrics.csv
      visualization.png
```

`signals.npz` là input quan trọng nhất cho Task 3:

| Key | Shape | Ý nghĩa |
| --- | --- | --- |
| `timestamps` | `(N,)` | Timeline frame-level. |
| `valid` | `(N,)` | Frame validity từ Task 1. |
| `roi_names` | `(3,)` | `forehead`, `left_cheek`, `right_cheek`. |
| `method_names` | `(3,)` | `green`, `chrom`, `pos`. |
| `mean_rgb` | `(N, 3, 3)` | RGB trung bình theo frame, ROI, channel. |
| `rppg_signals` | `(N, 3)` | rPPG baseline đã average ROI theo từng method. |
| `rppg_roi_signals` | `(3, N, 3)` | rPPG theo method, frame, ROI. |
| `hr_timestamps` | `(W,)` | Tâm sliding window. |
| `hr_estimates` | `(W, 3)` | HR baseline Task 2. Lưu ý: đã average ROI, không phải single ROI. |
| `gt_aligned_to_hr` | `(W,)` | Ground truth HR đã align với window. |

Metric Task 2:

| Metric | Ý nghĩa |
| --- | --- |
| `mae_bpm` | Sai số tuyệt đối trung bình, càng thấp càng tốt. |
| `rmse_bpm` | Phạt lỗi lớn mạnh hơn MAE, càng thấp càng tốt. |
| `pearson_r` | Tương quan trend giữa estimate và ground truth, càng cao càng tốt. |
| `bias_bpm` | Sai lệch trung bình `estimate - GT`. |
| `snr_db` | Độ rõ tín hiệu HR band so với nhiễu. |

Report hiện có:

```text
report/task2_signal_extraction_report.html
report/assets/task2_test_baselines.png
report/assets/task2_mae_by_split.png
```

Kết luận hiện tại: POS là baseline Task 2 tốt nhất trên test split; Green dùng được nhưng kém hơn POS; CHROM có lỗi cao trong protocol hiện tại.

---

## 5. Task 3 - Multi-ROI fusion

Task 3 tập trung vào **fusion đa ROI**: kiểm tra việc kết hợp trán, má trái và má phải có cải thiện kết quả so với từng ROI riêng lẻ hay không.

Task 3 chạy trực tiếp trong notebook:

```text
notebooks/03_fusion.ipynb
```

Entrypoint mặc định là:

```text
outputs/signal_extraction/signal_extraction_dataset_manifest.json
```

### Mục tiêu Task 3

So sánh các chiến lược:

1. **Single-ROI baseline**

   - Dùng từng ROI riêng lẻ: `forehead`, `left_cheek`, `right_cheek`.
   - Ước lượng HR riêng cho từng ROI.
   - Đánh giá ROI nào ổn định hơn theo từng sample/window.
2. **Average fusion**

   - Gộp tín hiệu ROI bằng trung bình đơn giản.
   - Đây là baseline fusion dễ hiểu, không cần quality score.
3. **Quality-weighted fusion**

   - Tính quality score cho từng ROI/window.
   - Gán trọng số cao hơn cho ROI có tín hiệu tốt hơn.
   - Chuẩn hóa trọng số để tổng weight của các ROI bằng 1.

Task 3 cần trả lời:

- Multi-ROI fusion có tốt hơn single ROI không?
- Average fusion có đủ tốt không, hay cần quality-weighted fusion?
- ROI nào thường đáng tin nhất trên UBFC?
- Khi một ROI bị nhiễu, bị che hoặc ánh sáng kém, fusion có giảm lỗi không?
- Quality features nào tương quan tốt với lỗi HR thực tế?

### Input Task 3

Task 3 đọc từ mỗi `signals.npz` của Signal extraction:

| Input                | Cách dùng                                                                         |
| -------------------- | ----------------------------------------------------------------------------------- |
| `rppg_roi_signals` | Input chính cho fusion. Shape `(3, N, 3)` tương ứng `(method, frame, roi)`. |
| `roi_names`        | Thứ tự ROI, ví dụ `forehead`, `left_cheek`, `right_cheek`.                |
| `method_names`     | Thứ tự method, ví dụ `green`, `chrom`, `pos`.                             |
| `timestamps`       | Timeline frame-level.                                                               |
| `valid`            | Mask frame hợp lệ từ Preprocessing.                                                     |
| `hr_timestamps`    | Timeline window-level cho HR estimate.                                              |
| `gt_aligned_to_hr` | Ground truth HR đã align theo `hr_timestamps`.                                  |
| `hr_estimates`     | HR từ Task 2. Lưu ý: đây là baseline đã trung bình 3 ROI, không phải single-ROI baseline. |
| `mean_rgb`         | Có thể dùng để tính motion/color quality bổ sung nếu cần.                  |

Task 3 không cần đọc raw video hoặc raw UBFC ground truth nếu Signal extraction đã chạy đúng.

`signal_extraction_baseline` trong Task 3 chỉ dùng để kiểm tra Task 3 tái tạo đúng baseline Task 2. So sánh chính để trả lời câu hỏi nghiên cứu là `single_*` vs `average_fusion` vs `quality_weighted_fusion`.

### Quality features

Mỗi ROI/window/method có vector quality features với tên cố định:

| Feature                             | Ý nghĩa                                                                 |
| ----------------------------------- | ------------------------------------------------------------------------- |
| `valid_ratio`                     | Tỷ lệ frame hợp lệ trong window.                                      |
| `signal_std`                      | Độ lệch chuẩn của ROI signal trong window.                            |
| `dominant_power_ratio`            | Peak HR-band FFT power chia cho median HR-band power.                  |
| `snr_like_db`                     | `10*log10(signal_power/(noise_power+1e-12))` quanh dominant peak ±0.1 Hz. |
| `inter_roi_disagreement_bpm`      | Lệch tuyệt đối giữa ROI HR và median HR của các ROI cùng method/window. |

Quality score dùng min-max normalize theo ROI cho từng `(window, method)`:

```text
score = 0.35 * snr_norm
      + 0.25 * clarity_norm
      + 0.20 * valid_ratio
      + 0.10 * std_norm
      - 0.10 * disagreement_norm
```

Edge cases:

- ROI có `valid_ratio < 0.8` hoặc score không hữu hạn được gán score 0.
- Score được clip về không âm và normalize để weight hữu hạn, không âm, tổng bằng 1.
- Nếu toàn bộ score bằng 0, fallback về uniform weight trên ROI có `valid_ratio >= 0.8`; nếu không có ROI hợp lệ, dùng uniform trên toàn bộ ROI.

### Output Task 3

```text
outputs/fusion/
  fusion_manifest.json
  fusion_metrics_summary.csv
  fusion_aggregate_summary.csv
  train/
    <sample_id>/
      fusion_signals.npz
      fusion_metrics.json
      fusion_metrics.csv
      fusion_visualization.png
  val/
    <sample_id>/
      fusion_signals.npz
      fusion_metrics.json
      fusion_metrics.csv
      fusion_visualization.png
  test/
    <sample_id>/
      fusion_signals.npz
      fusion_metrics.json
      fusion_metrics.csv
      fusion_visualization.png
```

Schema version: `fusion_v1`.

### `fusion_signals.npz`

| Key                            | Shape        | Ý nghĩa                                                      |
| ------------------------------ | ------------ | -------------------------------------------------------------- |
| `schema_version`             | scalar       | `fusion_v1`.                                                  |
| `sample_id`                  | scalar       | ID sample.                                                     |
| `split`                      | scalar       | Split dữ liệu.                                               |
| `timestamps`                 | `(N,)`       | Timeline frame-level.                                         |
| `valid`                      | `(N,)`       | Frame validity từ preprocessing.                              |
| `roi_names`                  | `(3,)`       | Thứ tự ROI.                                                   |
| `method_names`               | `(3,)`       | Thứ tự method: `green`, `chrom`, `pos`.                       |
| `fusion_strategy_names`      | `(6,)`       | `single_*`, `average_fusion`, `quality_weighted_fusion`, `signal_extraction_baseline`. |
| `feature_names`              | `(5,)`       | Tên quality features.                                         |
| `hr_timestamps`              | `(W,)`       | Timeline window-level.                                        |
| `gt_aligned_to_hr`           | `(W,)`       | Ground truth HR theo window.                                  |
| `baseline_hr`                | `(W, 3)`     | HR baseline từ Task 2; baseline này đã average ROI.            |
| `single_roi_hr`              | `(W, 3, 3)`  | HR theo `(window, method, roi)`.                              |
| `average_fusion_hr`          | `(W, 3)`     | HR sau average fusion theo method.                            |
| `weighted_fusion_hr`         | `(W, 3)`     | HR sau quality-weighted fusion theo method.                   |
| `roi_quality_features`       | `(W, 3, 3, 5)` | Quality features theo `(window, method, roi, feature)`.     |
| `roi_weights`                | `(W, 3, 3)`  | Method-specific ROI weights.                                  |
| `average_fused_rppg_signals` | `(N, 3)`     | Frame-level average-fused rPPG theo method.                   |
| `weighted_fused_rppg_signals` | `(N, 3)`    | Frame-level quality-weighted rPPG theo method.                |

### Metrics cần báo cáo

Task 3 phải so sánh chính:

- single ROI `forehead`
- single ROI `left_cheek`
- single ROI `right_cheek`
- average fusion
- quality-weighted fusion

`signal_extraction_baseline` được báo cáo riêng như baseline Task 2 đã average ROI; không dùng nó thay cho single-ROI baseline.

Metric:

- MAE BPM
- RMSE BPM
- Pearson correlation
- bias BPM
- SNR dB nếu có fused signal

Nên báo cáo theo:

- từng sample
- từng split
- tổng hợp riêng trên `test`

### Visualization cần có

Mỗi sample hoặc một nhóm sample đại diện nên có:

- HR theo từng ROI so với ground truth.
- HR của average fusion và weighted fusion so với ground truth.
- Weight của từng ROI theo thời gian/window.
- Quality features theo thời gian/window.
- Bar chart metric so sánh single ROI vs fusion.

### Tiêu chí hoàn thành Task 3

Task 3 chỉ nên xem là hoàn thành khi:

- Đọc được `outputs/signal_extraction/signal_extraction_dataset_manifest.json`.
- Load được `rppg_roi_signals`, `hr_timestamps`, `gt_aligned_to_hr`, `roi_names`, `method_names`.
- Tính được HR riêng cho từng ROI hoặc dùng được ROI-level signal để tạo HR theo ROI/window.
- Có average fusion.
- Có quality-weighted fusion với weight hữu hạn, không âm, tổng bằng 1.
- Có xử lý fallback khi quality score không hợp lệ.
- So sánh single ROI vs average fusion vs quality-weighted fusion.
- Xuất manifest, `.npz`, metrics `.json/.csv` và visualization.
- Có kết quả tổng hợp trên `test` split.

### Trạng thái output hiện tại

Task 3 hiện đã xử lý đủ toàn bộ 49 sample từ Signal extraction:

| Hạng mục | Giá trị |
| --- | --- |
| Source processed items từ Task 2 | 49 |
| Fusion processed / failed | 49 / 0 |
| Split count | train 39, val 5, test 5 |
| Per-sample metric rows | 882 = 49 samples × 3 methods × 6 strategies |
| Aggregate rows | 54 = 3 splits × 3 methods × 6 strategies |

Report hiện có:

```text
report/task3_fusion_report.html
report/assets/task3_test_mae.png
report/assets/task3_test_rmse_pearson.png
report/assets/task3_sample_visualization.png
```

Kết luận hiện tại: average fusion có ích rõ với Green/POS so với single ROI; quality-weighted fusion chỉ nhỉnh hơn average rất nhẹ với POS và chưa ổn định với Green/CHROM. `signal_extraction_baseline` gần trùng `average_fusion` vì baseline Task 2 đã average ROI.

---

## 6. Task 4 - Lightweight model

Task 4 là bước kế tiếp: huấn luyện mô hình nhẹ trên signal/window đã chuẩn hóa để kiểm tra mô hình học máy có vượt baseline truyền thống và fusion hay không.

### Mục tiêu Task 4

- Input chính: artifact từ Task 2 (`signals.npz`) và/hoặc Task 3 (`fusion_signals.npz`).
- Dự đoán HR theo window, cùng timeline `hr_timestamps` và ground truth `gt_aligned_to_hr`.
- So sánh với baseline:
  - Green/CHROM/POS từ Task 2.
  - `single_*`, `average_fusion`, `quality_weighted_fusion` từ Task 3.
  - Các paper/baseline trong proposal như DeepPhys, PhysNet, TS-CAN, EfficientPhys chỉ được đưa vào bảng khi có artifact hoặc số liệu cùng protocol.
- Ưu tiên mô hình nhẹ, dễ chạy, dễ giải thích: Tiny CNN, TCN, MLP/LightGBM trên feature nếu cần.

### Tổ chức folder đề xuất

```text
notebooks/
  04_lightweight_model.ipynb

src/airppg/modeling/
  __init__.py
  config.py          # dataclass config
  dataset.py         # đọc Task 2/3 artifact và tạo window dataset
  features.py        # feature/window transforms nếu dùng tabular model
  models.py          # TinyCNN, TCN, MLP nhỏ
  train.py           # train/eval loop nhỏ, notebook-friendly
  metrics.py         # wrapper metric nếu cần, ưu tiên reuse airppg.metrics
  visualization.py   # loss curve, prediction-vs-GT, comparison chart

outputs/modeling/
  modeling_manifest.json
  modeling_metrics_summary.csv
  modeling_aggregate_summary.csv
  train|val|test/
    <sample_id>/
      model_predictions.npz
      model_metrics.json
      model_metrics.csv
      model_visualization.png
  checkpoints/
    <run_name>/
      best_model.pt
      training_history.csv
      config.json

report/
  task4_lightweight_model_report.html
  assets/
    task4_*.png
```

### Input dataset cho Task 4

Khuyến nghị tạo window-level dataset từ các signal đã có, không đọc lại raw video:

| Nguồn | Key dùng | Ý nghĩa |
| --- | --- | --- |
| Task 2 `signals.npz` | `rppg_roi_signals` `(method, frame, roi)` | Per-ROI rPPG input. |
| Task 2 `signals.npz` | `rppg_signals` `(frame, method)` | Baseline averaged ROI signal. |
| Task 3 `fusion_signals.npz` | `average_fused_rppg_signals`, `weighted_fused_rppg_signals` | Fusion signals làm input hoặc baseline. |
| Task 3 `fusion_signals.npz` | `roi_quality_features`, `roi_weights` | Feature phụ cho quality-aware model. |
| Task 2/3 | `hr_timestamps`, `gt_aligned_to_hr` | Label và timeline window-level. |

Window geometry phải giống Task 2/3:

```text
window_size = min(max(3, round(WINDOW_SEC * fps)), N)
step_size = max(1, round(STEP_SEC * fps))
starts = range(0, N - window_size + 1, step_size)
```

Không tự tạo split mới nếu không cần. Dùng split từ manifest Task 1/2/3: `train`, `val`, `test`.

### Cách làm đề xuất

1. **Dataset builder**
   - Đọc `outputs/fusion/fusion_manifest.json` hoặc `outputs/signal_extraction/signal_extraction_dataset_manifest.json`.
   - Chỉ lấy item `status == "processed"`.
   - Với mỗi sample, tạo window input và label.
   - Lưu manifest dataset nếu có preprocessing feature nặng.

2. **Baseline trước khi train**
   - Load metric Task 2/3 làm bảng baseline cố định.
   - Không train model nếu chưa có baseline cùng split/protocol.

3. **Model nhỏ trước**
   - Tiny CNN 1D: input có shape gợi ý `(channels, window_size)`, channel có thể là ROI/method/fusion signal.
   - TCN nhỏ: dùng khi cần temporal receptive field dài hơn.
   - MLP/LightGBM: dùng với feature như SNR, dominant peak, ROI disagreement nếu muốn confidence/error prediction.

4. **Training protocol**
   - Train trên `train`, chọn checkpoint bằng `val`.
   - Chỉ báo cáo kết quả cuối trên `test`.
   - Lưu seed, config, model architecture, input channels, window length.
   - Early stopping theo `val_mae_bpm` hoặc `val_rmse_bpm`.

5. **Evaluation**
   - Metric bắt buộc: MAE, RMSE, Pearson, bias, SNR nếu có reconstructed/predicted signal.
   - So sánh cùng bảng với Task 2/3 baseline.
   - Vẽ prediction vs ground truth theo sample đại diện.
   - Vẽ bar chart test MAE/RMSE/Pearson giữa model và baseline.

### Lưu ý quan trọng

- Không leak data: mọi normalization học từ train phải fit trên train, apply cho val/test.
- Không dùng test để chọn epoch, chọn architecture, hoặc tune threshold.
- Không drop window lỗi âm thầm; nếu window không dùng được phải ghi lý do trong manifest hoặc metrics.
- Không đổi `hr_timestamps` hoặc interpolate label nếu không ghi rõ.
- Không so sánh với paper/baseline ngoài repo nếu khác dataset/protocol mà không ghi chú.
- Mô hình nhẹ phải có số tham số, thời gian inference hoặc ít nhất nhận xét chi phí tính toán.
- Nếu model không vượt baseline, vẫn báo cáo trung thực; kết quả âm vẫn có giá trị nghiên cứu.

### Output bắt buộc Task 4

```text
outputs/modeling/
  modeling_manifest.json
  modeling_metrics_summary.csv
  modeling_aggregate_summary.csv
  checkpoints/<run_name>/
    best_model.pt
    training_history.csv
    config.json
  train|val|test/<sample_id>/
    model_predictions.npz
    model_metrics.json
    model_metrics.csv
    model_visualization.png
```

`model_predictions.npz` nên có:

| Key | Shape | Ý nghĩa |
| --- | --- | --- |
| `schema_version` | scalar | Ví dụ `modeling_v1`. |
| `sample_id`, `split`, `model_name` | scalar | Metadata. |
| `hr_timestamps` | `(W,)` | Timeline window-level. |
| `gt_aligned_to_hr` | `(W,)` | Ground truth HR. |
| `predicted_hr` | `(W,)` hoặc `(W, K)` | HR dự đoán. |
| `baseline_hr` | `(W, B)` | Baseline được so sánh nếu lưu kèm. |
| `input_strategy_names` | `(C,)` | Tên input channel/strategy. |
| `valid_windows` | `(W,)` | Window có dùng để tính metric hay không. |
| `confidence` | `(W,)` optional | Nếu Task 4 thử confidence/error-aware output. |

Report Task 4 bắt buộc đặt trong `report/`, dùng `report/report.css`, có giải thích metric, biểu đồ, bảng so sánh baseline, nhận xét dưới từng chart và kết luận cuối.

---

## 7. Kiểm tra nhanh artifact hiện tại

Sau khi chạy Task 2, kiểm tra Signal extraction manifest:

```powershell
uv run python -c "import json; from pathlib import Path; p=Path('outputs/signal_extraction/signal_extraction_dataset_manifest.json'); m=json.loads(p.read_text(encoding='utf-8')); print(m['counts'])"
```

Kiểm tra shape Task 2 của một sample:

```powershell
uv run python -c "import json, numpy as np; from pathlib import Path; root=Path('outputs/signal_extraction'); m=json.loads((root/'signal_extraction_dataset_manifest.json').read_text(encoding='utf-8')); item=next(i for i in m['items'] if i['status']=='processed'); data=np.load(root/item['outputs']['signals_npz']); print(data['mean_rgb'].shape, data['rppg_roi_signals'].shape, data['hr_estimates'].shape)"
```

Sau khi chạy Task 3, kiểm tra Fusion manifest và weight:

```powershell
uv run python -c "import json, numpy as np; from pathlib import Path; root=Path('outputs/fusion'); m=json.loads((root/'fusion_manifest.json').read_text(encoding='utf-8')); print(m['counts']); item=next(i for i in m['items'] if i['status']=='processed'); d=np.load(root/item['outputs']['fusion_npz']); w=d['roi_weights']; print(d['single_roi_hr'].shape, w.shape, np.allclose(w.sum(axis=2), 1.0))"
```

---

## 8. Ghi chú vận hành

- Không commit dataset hoặc output lớn.
- Không hardcode đường dẫn máy cá nhân vào notebook/script commit lên repo.
- Preprocessing và Signal extraction phải giữ path tương đối trong manifest để dễ chuyển máy.
- Nếu thay đổi schema output, cần tăng `schema_version` và cập nhật README tương ứng.
