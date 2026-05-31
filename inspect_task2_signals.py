"""Script kiểm tra nhanh (inspect) kết quả đầu ra của Task 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kiểm tra nhanh kết quả đầu ra của Task 2 (Tín hiệu rPPG và Nhịp tim)."
    )
    parser.add_argument(
        "target_path",
        type=str,
        nargs="?",
        default="outputs/task2_signals/",
        help="Đường dẫn tới file *_rppg_signals.npz hoặc thư mục chứa kết quả của Task 2.",
    )
    args = parser.parse_args()

    target = Path(args.target_path)
    if not target.exists():
        print(f"Lỗi: Đường dẫn không tồn tại: {target}")
        return

    # Nếu truyền vào thư mục, tự động quét tìm file *_rppg_signals.npz
    rppg_file = None
    if target.is_dir():
        print(f"Đang quét thư mục '{target}' để tìm kết quả của Task 2...")
        candidates = sorted(list(target.rglob("*_rppg_signals.npz")))
        if not candidates:
            print("Không tìm thấy tệp *_rppg_signals.npz nào.")
            return
        rppg_file = candidates[-1]  # Lấy file mới nhất hoặc đầu tiên tìm thấy
        print(f"Tìm thấy tệp kết quả: {rppg_file.name}")
    else:
        if not target.name.endswith("_rppg_signals.npz"):
            print("Cảnh báo: Tệp tin nên có định dạng *_rppg_signals.npz")
        rppg_file = target

    # Xác định các file liên quan
    base_name = rppg_file.name.replace("_rppg_signals.npz", "")
    parent_dir = rppg_file.parent

    rgb_file = parent_dir / f"{base_name}_rgb_signals.npz"
    hr_file = parent_dir / f"{base_name}_hr_traces.npz"
    metrics_file = parent_dir / f"{base_name}_metrics.json"

    print("=" * 60)
    print(f"BÁO CÁO NHANH KẾT QUẢ TASK 2 - {base_name.upper()}")
    print("=" * 60)

    # 1. Kiểm tra các tệp tin
    print("1. Trạng thái các tệp tin kết quả:")
    print(f"  - [x] rPPG Signals: {rppg_file} ({rppg_file.stat().st_size / 1024:.1f} KB)")
    print(
        f"  - [{'x' if rgb_file.exists() else ' '}] RGB Mean: {rgb_file.name} "
        f"({rgb_file.stat().st_size / 1024:.1f} KB nếu có)"
        if rgb_file.exists()
        else f"  - [ ] RGB Mean: {rgb_file.name} (Chưa tạo/Không có)"
    )
    print(
        f"  - [{'x' if hr_file.exists() else ' '}] HR Traces: {hr_file.name} "
        f"({hr_file.stat().st_size / 1024:.1f} KB nếu có)"
        if hr_file.exists()
        else f"  - [ ] HR Traces: {hr_file.name} (Chưa tạo/Không có)"
    )
    print(
        f"  - [{'x' if metrics_file.exists() else ' '}] Metrics JSON: {metrics_file.name}"
        if metrics_file.exists()
        else f"  - [ ] Metrics JSON: {metrics_file.name} (Chưa tạo/Không có)"
    )

    # 2. Đọc và phân tích tín hiệu rPPG
    print("\n2. Phân tích Tín hiệu rPPG:")
    try:
        rppg_data = np.load(rppg_file)
        ts = rppg_data["timestamps"]
        print(f"  - Số lượng mẫu thời gian: {len(ts)}")
        print(f"  - Độ dài thời gian: {ts[-1] - ts[0]:.2f} giây (từ {ts[0]:.2f}s đến {ts[-1]:.2f}s)")

        for method in ["green", "chrom", "pos"]:
            if method in rppg_data:
                sig_arr = rppg_data[method]
                print(
                    f"  - Thuật toán {method.upper()}: shape={sig_arr.shape}, "
                    f"std={np.std(sig_arr):.4f}, mean={np.mean(sig_arr):.4f}"
                )
    except Exception as e:
        print(f"  - Lỗi khi đọc dữ liệu rPPG: {e}")

    # 3. Phân tích Nhịp tim ước lượng (Estimated Heart Rate)
    if hr_file.exists():
        print("\n3. Thống kê Nhịp tim ước lượng (Heart Rate - BPM):")
        try:
            hr_data = np.load(hr_file)
            hr_ts = hr_data["hr_timestamps"]
            print(f"  - Số lượng cửa sổ ước lượng: {len(hr_ts)}")

            for key in ["green_hr", "chrom_hr", "pos_hr", "gt_hr"]:
                if key in hr_data:
                    arr = hr_data[key]
                    valid_arr = arr[~np.isnan(arr)]
                    if len(valid_arr) > 0:
                        name = key.replace("_hr", "").upper()
                        print(
                            f"  - {name:6s}: Trung bình = {np.mean(valid_arr):.2f} BPM, "
                            f"Khoảng = [{np.min(valid_arr):.1f} - {np.max(valid_arr):.1f}] BPM"
                        )
                    else:
                        print(f"  - {key.upper()}: Không có dữ liệu hợp lệ (toàn bộ NaN)")
        except Exception as e:
            print(f"  - Lỗi khi đọc dữ liệu nhịp tim: {e}")

    # 4. Hiển thị bảng Metrics đánh giá chất lượng
    if metrics_file.exists():
        print("\n4. Báo cáo sai số so với Ground Truth (Metrics):")
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            print(f"  {'-'*68}")
            print(f"  | {'Phương pháp':15s} | {'MAE (BPM)':10s} | {'RMSE (BPM)':10s} | {'Pearson r':10s} | {'SNR (dB)':10s} |")
            print(f"  {'-'*68}")
            for method in ["green", "chrom", "pos"]:
                if method in metrics:
                    m = metrics[method]
                    mae = f"{m.get('mae', float('nan')):.3f}"
                    rmse = f"{m.get('rmse', float('nan')):.3f}"
                    r = f"{m.get('pearson_r', float('nan')):.3f}"
                    snr = f"{m.get('snr', float('nan')):.3f}"
                    print(f"  | {method.upper():15s} | {mae:10s} | {rmse:10s} | {r:10s} | {snr:10s} |")
            print(f"  {'-'*68}")
        except Exception as e:
            print(f"  - Lỗi khi đọc hoặc hiển thị file metrics: {e}")

    print("=" * 60)


if __name__ == "__main__":
    main()
