"""Thư viện lõi chứa các hàm trích xuất tín hiệu, tiền xử lý, thuật toán baseline rPPG, ước lượng nhịp tim và đánh giá độ chính xác."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.signal as sig
import scipy.signal.windows as windows
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from tqdm.auto import tqdm


def extract_roi_rgb(
    video_path: Path | str,
    npz_data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    progress_bar: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Trích xuất tín hiệu màu RGB trung bình cho từng ROI từ video gốc.

    Args:
        video_path: Đường dẫn tới video gốc.
        npz_data: Dữ liệu .npz từ Task 1 (chứa roi_masks, frame_indices, valid).
        metadata: Siêu dữ liệu từ Task 1.
        progress_bar: Có hiển thị thanh tiến trình tqdm hay không.

    Returns:
        mean_rgb: Mảng shape (N_valid, 3, 3) đại diện cho trung bình màu của 3 ROI (forehead, left_cheek, right_cheek)
                  cho 3 kênh màu (R, G, B).
        timestamps: Mảng shape (N_valid,) đại diện cho thời gian (giây) của từng khung hình tương ứng.
    """
    video_path = Path(video_path)
    frame_indices = npz_data["frame_indices"]
    roi_masks = npz_data["roi_masks"]
    valid = npz_data["valid"]

    # Lấy danh sách các index khung hình hợp lệ
    valid_idxs = np.where(valid)[0]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Không thể mở video nguồn tại: {video_path}")

    # Lấy các cấu hình từ metadata
    fps = metadata["video"]["fps"]
    mask_h, mask_w = roi_masks.shape[2:]

    n_valid = len(valid_idxs)
    mean_rgb = np.zeros((n_valid, 3, 3), dtype=np.float32)
    timestamps = np.zeros(n_valid, dtype=np.float32)

    success_count = 0
    actual_indices = []

    iterator = range(n_valid)
    if progress_bar:
        iterator = tqdm(iterator, desc="Trích xuất màu RGB cho các ROI", unit="khung hình")

    for i in iterator:
        orig_idx = valid_idxs[i]
        frame_idx = frame_indices[orig_idx]

        # Di chuyển tới đúng khung hình mong muốn
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()

        if not ret:
            # Gặp lỗi khi đọc khung hình -> bỏ qua khung hình này
            continue

        # Thay đổi kích thước khung hình về khớp với mặt nạ ROI
        h, w = frame_bgr.shape[:2]
        if w != mask_w or h != mask_h:
            frame_bgr = cv2.resize(frame_bgr, (mask_w, mask_h), interpolation=cv2.INTER_AREA)

        # Chuyển đổi hệ màu BGR sang RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Tính toán giá trị trung bình pixel cho từng ROI
        for roi_idx in range(3):
            mask = roi_masks[orig_idx, roi_idx]
            if mask.sum() > 0:
                mean_rgb[success_count, roi_idx] = frame_rgb[mask > 0].mean(axis=0)
            else:
                mean_rgb[success_count, roi_idx] = np.array([np.nan, np.nan, np.nan])

        timestamps[success_count] = frame_idx / fps
        actual_indices.append(orig_idx)
        success_count += 1

    cap.release()

    # Chỉ giữ lại phần mảng đã đọc thành công
    mean_rgb = mean_rgb[:success_count]
    timestamps = timestamps[:success_count]

    return mean_rgb, timestamps


def preprocess_signal(
    signal: np.ndarray,
    fps: float,
    detrend: bool = True,
    normalize: bool = True,
    bandpass: tuple[float, float] | None = (0.7, 3.5),
) -> np.ndarray:
    """Tiền xử lý chuỗi tín hiệu thời gian (Detrend, Lọc thông dải, Chuẩn hóa z-score).

    Args:
        signal: Chuỗi tín hiệu đầu vào (1D array).
        fps: Tần số quét khung hình (Hz).
        detrend: Có loại bỏ xu hướng tuyến tính (detrend) hay không.
        normalize: Có chuẩn hóa z-score (zero-mean, unit-variance) hay không.
        bandpass: Khoảng tần số lọc thông dải (Hz). Mặc định là (0.7, 3.5) tương ứng 42-210 BPM.

    Returns:
        Chuỗi tín hiệu sau khi đã tiền xử lý.
    """
    if len(signal) == 0:
        return signal

    proc_sig = np.array(signal, dtype=np.float32)

    # Xử lý các giá trị NaN nếu có bằng phương pháp nội suy tuyến tính
    nan_mask = np.isnan(proc_sig)
    if np.any(nan_mask):
        non_nan_indices = np.where(~nan_mask)[0]
        if len(non_nan_indices) > 0:
            indices = np.arange(len(proc_sig))
            proc_sig[nan_mask] = np.interp(
                indices[nan_mask], non_nan_indices, proc_sig[~nan_mask]
            )
        else:
            return np.zeros_like(proc_sig)

    # 1. Khử xu hướng tuyến tính (Detrend)
    if detrend and len(proc_sig) > 1:
        proc_sig = sig.detrend(proc_sig, type="linear")

    # 2. Lọc thông dải (Bandpass filter) Butterworth bậc 4
    if bandpass is not None and len(proc_sig) > 9:  # Yêu cầu chiều dài tín hiệu tối thiểu
        low, high = bandpass
        nyquist = 0.5 * fps
        low_norm = low / nyquist
        high_norm = high / nyquist

        # Đảm bảo tần số chuẩn hóa nằm trong khoảng (0, 1) để tránh lỗi Butterworth
        low_norm = max(0.001, min(0.999, low_norm))
        high_norm = max(0.002, min(0.999, high_norm))

        b, a = sig.butter(4, [low_norm, high_norm], btype="bandpass")
        proc_sig = sig.filtfilt(b, a, proc_sig)

    # 3. Chuẩn hóa z-score (Zero-mean, Unit variance)
    if normalize:
        std = np.std(proc_sig)
        if std > 0:
            proc_sig = (proc_sig - np.mean(proc_sig)) / std
        else:
            proc_sig = proc_sig - np.mean(proc_sig)

    return proc_sig


def green_channel(
    mean_rgb: np.ndarray, roi_idx: int = 0, fps: float = 30.0, return_rois: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Thuật toán Green Channel baseline.

    Args:
        mean_rgb: Mảng shape (N, 3, 3) đại diện cho RGB của 3 ROI qua thời gian.
        roi_idx: Chỉ số ROI cần trích xuất (Mặc định 0 = forehead).
        fps: Tần số quét (Hz).
        return_rois: Có trả về tín hiệu của từng ROI riêng lẻ hay không.

    Returns:
        Nếu return_rois=False: Tín hiệu rPPG trung bình đã tiền xử lý.
        Nếu return_rois=True: Tuple (tín hiệu rPPG trung bình, mảng các tín hiệu ROI shape (N, N_rois)).
    """
    n_frames, n_rois, _ = mean_rgb.shape

    if return_rois:
        rppg_rois = []
        for i in range(n_rois):
            sig_roi = preprocess_signal(mean_rgb[:, i, 1], fps)
            rppg_rois.append(sig_roi)
        rppg_rois_arr = np.stack(rppg_rois, axis=1)
        rppg_avg = np.mean(rppg_rois_arr, axis=1)
        # Chuẩn hóa z-score cho rppg_avg
        std_avg = np.std(rppg_avg)
        if std_avg > 0:
            rppg_avg = (rppg_avg - np.mean(rppg_avg)) / std_avg
        return rppg_avg, rppg_rois_arr

    signal = mean_rgb[:, roi_idx, 1]  # index 1 đại diện cho Green channel
    return preprocess_signal(signal, fps)


def chrom_rppg(
    mean_rgb: np.ndarray, fps: float, return_rois: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Thuật toán CHROM (Chrominance-based) theo de Haan & Jeanne (2013).

    Sử dụng kỹ thuật cửa sổ trượt Overlap-Add với độ dài cửa sổ khoảng 1.6 giây.

    Args:
        mean_rgb: Mảng shape (N, 3, 3) chứa dữ liệu RGB của các ROI qua thời gian.
        fps: Tần số quét (Hz).
        return_rois: Có trả về tín hiệu của từng ROI riêng lẻ hay không.

    Returns:
        Nếu return_rois=False: Tín hiệu rPPG tổng hợp (trung bình qua các ROI).
        Nếu return_rois=True: Tuple (tín hiệu rPPG tổng hợp, mảng các tín hiệu ROI shape (N, N_rois)).
    """
    n_frames, n_rois, _ = mean_rgb.shape
    L = int(round(1.6 * fps))
    if L < 3:
        L = 3

    rppg_rois = []

    for roi_idx in range(n_rois):
        s_global = np.zeros(n_frames, dtype=np.float32)
        w_global = np.zeros(n_frames, dtype=np.float32)

        # Trượt cửa sổ bước 1 frame
        for i in range(n_frames - L + 1):
            window = mean_rgb[i : i + L, roi_idx, :]  # shape (L, 3)

            # Chuẩn hóa tạm thời theo trung bình của cửa sổ
            mean_c = np.mean(window, axis=0)
            mean_c = np.where(mean_c == 0, 1.0, mean_c)
            c_norm = window / mean_c

            R_n = c_norm[:, 0]
            G_n = c_norm[:, 1]
            B_n = c_norm[:, 2]

            # Tính các tín hiệu màu sắc (chrominance signals)
            xs = 3.0 * R_n - 2.0 * G_n
            ys = 1.5 * R_n + G_n - 1.5 * B_n

            std_xs = np.std(xs)
            std_ys = np.std(ys)

            if std_xs == 0 or std_ys == 0:
                s_win = np.zeros(L, dtype=np.float32)
            else:
                alpha = std_xs / std_ys
                s_win = xs - alpha * ys

            # Áp dụng cửa sổ Hanning để làm mượt tại các biên khi ghép
            win = np.hanning(L)
            s_global[i : i + L] += s_win * win
            w_global[i : i + L] += win

        w_global = np.where(w_global == 0, 1.0, w_global)
        s_roi = s_global / w_global

        # Tiền xử lý lọc thông dải cho từng ROI
        s_roi_filtered = preprocess_signal(
            s_roi, fps, detrend=True, normalize=True, bandpass=(0.7, 3.5)
        )
        rppg_rois.append(s_roi_filtered)

    rppg_rois_arr = np.stack(rppg_rois, axis=1)

    # Trung bình hóa tín hiệu qua các ROI
    rppg_avg = np.mean(rppg_rois_arr, axis=1)

    # Chuẩn hóa z-score đầu ra cuối cùng
    std_avg = np.std(rppg_avg)
    if std_avg > 0:
        rppg_avg = (rppg_avg - np.mean(rppg_avg)) / std_avg

    if return_rois:
        # Chuẩn hóa từng ROI rPPG trước khi trả về
        normalized_rois = np.zeros_like(rppg_rois_arr)
        for i in range(n_rois):
            std_i = np.std(rppg_rois_arr[:, i])
            if std_i > 0:
                normalized_rois[:, i] = (rppg_rois_arr[:, i] - np.mean(rppg_rois_arr[:, i])) / std_i
            else:
                normalized_rois[:, i] = rppg_rois_arr[:, i] - np.mean(rppg_rois_arr[:, i])
        return rppg_avg, normalized_rois

    return rppg_avg


def pos_rppg(
    mean_rgb: np.ndarray, fps: float, return_rois: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Thuật toán POS (Plane-Orthogonal-to-Skin) theo Wang et al. (2017).

    Sử dụng kỹ thuật cửa sổ trượt Overlap-Add với độ dài cửa sổ khoảng 1.6 giây.

    Args:
        mean_rgb: Mảng shape (N, 3, 3) chứa dữ liệu RGB của các ROI qua thời gian.
        fps: Tần số quét (Hz).
        return_rois: Có trả về tín hiệu của từng ROI riêng lẻ hay không.

    Returns:
        Nếu return_rois=False: Tín hiệu rPPG tổng hợp (trung bình qua các ROI).
        Nếu return_rois=True: Tuple (tín hiệu rPPG tổng hợp, mảng các tín hiệu ROI shape (N, N_rois)).
    """
    n_frames, n_rois, _ = mean_rgb.shape
    L = int(round(1.6 * fps))
    if L < 3:
        L = 3

    rppg_rois = []

    # Ma trận hình chiếu POS
    P = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float32)

    for roi_idx in range(n_rois):
        s_global = np.zeros(n_frames, dtype=np.float32)
        w_global = np.zeros(n_frames, dtype=np.float32)

        for i in range(n_frames - L + 1):
            window = mean_rgb[i : i + L, roi_idx, :]  # shape (L, 3)

            # Chuẩn hóa tạm thời theo trung bình của cửa sổ
            mean_c = np.mean(window, axis=0)
            mean_c = np.where(mean_c == 0, 1.0, mean_c)
            c_norm = window / mean_c

            # Chiếu tín hiệu lên mặt phẳng POS phẳng giao với da
            # H_raw có kích thước (2, L)
            H_raw = P @ c_norm.T

            std_0 = np.std(H_raw[0])
            std_1 = np.std(H_raw[1])

            if std_1 == 0:
                s_win = np.zeros(L, dtype=np.float32)
            else:
                alpha = std_0 / std_1
                s_win = H_raw[0] + alpha * H_raw[1]

            # Áp dụng cửa sổ Hanning để ghép nối cửa sổ trượt
            win = np.hanning(L)
            s_global[i : i + L] += s_win * win
            w_global[i : i + L] += win

        w_global = np.where(w_global == 0, 1.0, w_global)
        s_roi = s_global / w_global

        # Tiền xử lý lọc thông dải cho từng ROI
        s_roi_filtered = preprocess_signal(
            s_roi, fps, detrend=True, normalize=True, bandpass=(0.7, 3.5)
        )
        rppg_rois.append(s_roi_filtered)

    rppg_rois_arr = np.stack(rppg_rois, axis=1)

    # Trung bình hóa qua các ROI
    rppg_avg = np.mean(rppg_rois_arr, axis=1)

    # Chuẩn hóa z-score đầu ra cuối cùng
    std_avg = np.std(rppg_avg)
    if std_avg > 0:
        rppg_avg = (rppg_avg - np.mean(rppg_avg)) / std_avg

    if return_rois:
        # Chuẩn hóa từng ROI rPPG trước khi trả về
        normalized_rois = np.zeros_like(rppg_rois_arr)
        for i in range(n_rois):
            std_i = np.std(rppg_rois_arr[:, i])
            if std_i > 0:
                normalized_rois[:, i] = (rppg_rois_arr[:, i] - np.mean(rppg_rois_arr[:, i])) / std_i
            else:
                normalized_rois[:, i] = rppg_rois_arr[:, i] - np.mean(rppg_rois_arr[:, i])
        return rppg_avg, normalized_rois

    return rppg_avg


def estimate_hr(
    signal: np.ndarray,
    fps: float,
    window_sec: float = 30.0,
    step_sec: float = 1.0,
    hr_range: tuple[float, float] = (42.0, 210.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Ước lượng nhịp tim (BPM) từ phổ tần số sử dụng biến đổi Fourier cửa sổ trượt (FFT).

    Args:
        signal: Chuỗi tín hiệu thời gian rPPG.
        fps: Tần số quét (Hz).
        window_sec: Độ rộng cửa sổ phân tích FFT (giây). Mặc định 30 giây.
        step_sec: Bước nhảy cửa sổ (giây). Mặc định 1 giây.
        hr_range: Khoảng nhịp tim sinh lý plausbile (BPM). Mặc định (42, 210) BPM.

    Returns:
        hr_trace: Mảng 1D chứa các giá trị nhịp tim ước lượng (BPM).
        timestamps: Mảng 1D chứa mốc thời gian trung tâm của từng cửa sổ (giây).
    """
    N = len(signal)
    W_size = int(round(window_sec * fps))
    S_step = int(round(step_sec * fps))

    if W_size > N:
        # Nếu chuỗi tín hiệu quá ngắn, thu nhỏ cửa sổ về toàn bộ chiều dài
        W_size = N
        S_step = N

    hr_trace = []
    timestamps = []

    f_min, f_max = hr_range[0] / 60.0, hr_range[1] / 60.0

    start = 0
    while start + W_size <= N:
        end = start + W_size
        t_center = (start + end - 1) / (2.0 * fps)

        s_win = signal[start:end]

        # Khử DC (mean)
        s_win = s_win - np.mean(s_win)

        # Áp dụng cửa sổ Hann
        win = windows.hann(W_size)
        s_win = s_win * win

        # Thực hiện biến đổi Fourier nhanh (FFT) cho tín hiệu thực
        fft_vals = np.fft.rfft(s_win)
        freqs = np.fft.rfftfreq(W_size, 1.0 / fps)
        power_spec = np.abs(fft_vals) ** 2

        # Tìm tần số đỉnh trong dải nhịp tim cho phép
        valid_idx = (freqs >= f_min) & (freqs <= f_max)
        if np.sum(valid_idx) > 0:
            peak_idx = np.argmax(power_spec[valid_idx])
            peak_freq = freqs[valid_idx][peak_idx]
            hr = peak_freq * 60.0
        else:
            hr = np.nan

        hr_trace.append(hr)
        timestamps.append(t_center)

        start += S_step
        if S_step <= 0:
            break

    return np.array(hr_trace, dtype=np.float32), np.array(timestamps, dtype=np.float32)


def load_ground_truth(
    gt_path: Path | str, fps_gt: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Tải và parse dữ liệu ground truth nhịp tim, hỗ trợ cả 2 split của UBFC (.txt và .xmp).

    Args:
        gt_path: Đường dẫn tới file ground truth (.txt hoặc .xmp).
        fps_gt: Tần số quét lấy mẫu của tín hiệu gốc (nếu là dạng raw signal cần chuyển đổi).

    Returns:
        gt_hr: Mảng nhịp tim gốc (BPM).
        gt_timestamps: Mảng mốc thời gian tương ứng (giây).
    """
    gt_path = Path(gt_path)

    # 1. Xử lý file .xmp (DATASET_1)
    if gt_path.suffix.lower() == ".xmp":
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(gt_path)
            root = tree.getroot()

            pulse_data = []
            hr_data = []

            # Đệ quy tìm kiếm tag chứa dữ liệu nhịp tim hoặc sóng xung
            def find_elements(element: ET.Element):
                tag = element.tag.lower()
                if "hr" in tag or "heartrate" in tag or "bpm" in tag:
                    text = element.text
                    if text:
                        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
                        hr_data.extend([float(x) for x in nums])
                elif (
                    "pulse" in tag
                    or "signal" in tag
                    or "wave" in tag
                    or "value" in tag
                    or "data" in tag
                ):
                    text = element.text
                    if text:
                        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
                        pulse_data.extend([float(x) for x in nums])

                for child in element:
                    find_elements(child)

            find_elements(root)

            if len(hr_data) > 0:
                gt_hr = np.array(hr_data, dtype=np.float32)
                fps = fps_gt if fps_gt is not None else 1.0
                gt_ts = np.arange(len(gt_hr)) / fps
                return gt_hr, gt_ts
            elif len(pulse_data) > 0:
                pulse_signal = np.array(pulse_data, dtype=np.float32)
                fps = fps_gt if fps_gt is not None else 30.0
                gt_hr = _estimate_running_hr_from_ppg(pulse_signal, fps)
                gt_ts = np.arange(len(pulse_signal)) / fps
                return gt_hr, gt_ts
        except Exception:
            pass  # Nếu lỗi khi parse XML thì tự động nhảy sang nạp dạng bảng (loadtxt)

    # 2. Xử lý file .txt (DATASET_2)
    try:
        try:
            data = np.loadtxt(gt_path, delimiter=",")
        except Exception:
            data = np.loadtxt(gt_path)

        if data.ndim == 1:
            val = data
            fps = fps_gt if fps_gt is not None else 30.0
            ts = np.arange(len(val)) / fps
        else:
            # Kiểm tra cấu trúc cột
            if data.shape[1] == 2:
                # Thường là [time, BPM] hoặc [PPG, time]
                mean_0 = np.nanmean(data[:, 0])
                mean_1 = np.nanmean(data[:, 1])
                if mean_0 > 40.0:
                    val = data[:, 0]
                    ts = data[:, 1]
                elif mean_1 > 40.0:
                    val = data[:, 1]
                    ts = data[:, 0]
                else:
                    val = data[:, 0]
                    ts = data[:, 1]
            elif data.shape[1] >= 3:
                # Cấu trúc chuẩn của UBFC DATASET_2:
                # Cột 0: Sóng PPG, Cột 1: Nhịp tim BPM, Cột 2: Thời gian (giây)
                val = data[:, 1]
                ts = data[:, 2]
            else:
                val = data[:, 0]
                ts = np.arange(len(val)) / (fps_gt if fps_gt is not None else 30.0)

        # Phát hiện loại tín hiệu: Nếu mean > 40 thì là nhịp tim BPM trực tiếp
        if np.nanmean(val) > 40.0:
            return val, ts
        else:
            # Ngược lại là tín hiệu PPG thô -> Cần chạy FFT cửa sổ trượt để ra chuỗi BPM
            fps = fps_gt if fps_gt is not None else 30.0
            gt_hr = _estimate_running_hr_from_ppg(val, fps)
            return gt_hr, ts
    except Exception as e:
        raise ValueError(f"Không thể đọc file ground truth {gt_path}: {e}")


def _estimate_running_hr_from_ppg(ppg: np.ndarray, fps: float) -> np.ndarray:
    """Ước lượng nhịp tim cục bộ mẫu-qua-mẫu (sample-by-sample) từ tín hiệu sóng PPG thô."""
    win_size = int(round(8.0 * fps))
    half_win = win_size // 2
    n = len(ppg)
    hr = np.zeros(n, dtype=np.float32)

    for i in range(n):
        start = max(0, i - half_win)
        end = min(n, i + half_win)
        w = ppg[start:end]
        if len(w) < 3:
            hr[i] = np.nan
            continue

        w = w - np.mean(w)
        w = w * windows.hann(len(w))

        fft_vals = np.fft.rfft(w)
        freqs = np.fft.rfftfreq(len(w), 1.0 / fps)
        power = np.abs(fft_vals) ** 2

        valid_idx = (freqs >= 0.7) & (freqs <= 3.5)
        if np.sum(valid_idx) > 0:
            peak_idx = np.argmax(power[valid_idx])
            hr[i] = freqs[valid_idx][peak_idx] * 60.0
        else:
            hr[i] = np.nan

    return hr


def calculate_snr(signal: np.ndarray, fps: float, gt_hr: float, band_width: float = 0.1) -> float:
    """Tính toán chỉ số SNR (Signal-to-Noise Ratio) của tín hiệu rPPG so với nhịp tim Ground Truth (BPM).

    Sử dụng phương pháp tính tổng công suất trong dải tần số nhịp tim (tần số cơ bản và hài bậc 2)
    chia cho công suất của các tần số nhiễu khác trong dải sinh lý [0.7, 3.5] Hz.
    """
    sig_detrend = sig.detrend(signal - np.mean(signal))
    N = len(sig_detrend)
    if N < 3:
        return np.nan

    # Thực hiện biến đổi Fourier
    fft_vals = np.fft.rfft(sig_detrend)
    freqs = np.fft.rfftfreq(N, 1.0 / fps)
    power_spec = np.abs(fft_vals) ** 2

    # Tần số nhịp tim Ground Truth (Hz)
    f_gt = gt_hr / 60.0

    # Dải tần sinh lý [0.7, 3.5] Hz (tương ứng [42, 210] BPM)
    f_min, f_max = 0.7, 3.5

    # Định nghĩa dải tần tín hiệu (fundamental frequency và harmonics)
    fundamental_mask = (freqs >= (f_gt - band_width)) & (freqs <= (f_gt + band_width))
    harmonic_mask = (freqs >= (2 * f_gt - 2 * band_width)) & (freqs <= (2 * f_gt + 2 * band_width))
    signal_mask = fundamental_mask | harmonic_mask

    total_band_mask = (freqs >= f_min) & (freqs <= f_max)

    p_signal = np.sum(power_spec[signal_mask & total_band_mask])
    p_noise = np.sum(power_spec[(~signal_mask) & total_band_mask])

    if p_noise <= 0 or p_signal <= 0:
        return np.nan

    snr = 10 * np.log10(p_signal / p_noise)
    return float(snr)


def evaluate(
    pred_hr: np.ndarray,
    pred_ts: np.ndarray,
    gt_hr: np.ndarray,
    gt_ts: np.ndarray,
    rppg_signal: np.ndarray | None = None,
    fps: float | None = None,
) -> dict[str, float]:
    """Căn khớp thời gian và tính toán các chỉ số đánh giá độ chính xác ước lượng nhịp tim (có tính SNR).

    Args:
        pred_hr: Nhịp tim dự đoán từ thuật toán (BPM).
        pred_ts: Mốc thời gian của nhịp tim dự đoán (giây).
        gt_hr: Nhịp tim gốc Ground Truth (BPM).
        gt_ts: Mốc thời gian của Ground Truth (giây).
        rppg_signal: Tín hiệu rPPG toàn cục (dùng để tính SNR).
        fps: Tần số quét của tín hiệu rPPG.

    Returns:
        Từ điển chứa các chỉ số: MAE, RMSE, pearson_r, mean_error (bias), snr.
    """
    if len(pred_hr) == 0 or len(gt_hr) == 0:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "pearson_r": np.nan,
            "mean_error": np.nan,
            "snr": np.nan,
        }

    try:
        # Sử dụng nội suy tuyến tính để đưa Ground Truth về cùng các mốc thời gian của dự đoán
        f_gt = interp1d(gt_ts, gt_hr, kind="linear", bounds_error=False, fill_value="extrapolate")
        gt_aligned = f_gt(pred_ts)

        # Lọc bỏ các giá trị NaN để tính toán chính xác
        mask = (~np.isnan(pred_hr)) & (~np.isnan(gt_aligned))
        if np.sum(mask) < 2:
            return {
                "mae": np.nan,
                "rmse": np.nan,
                "pearson_r": np.nan,
                "mean_error": np.nan,
                "snr": np.nan,
            }

        pred_clean = pred_hr[mask]
        gt_clean = gt_aligned[mask]

        mae = np.mean(np.abs(pred_clean - gt_clean))
        rmse = np.sqrt(np.mean((pred_clean - gt_clean) ** 2))

        # Tính hệ số tương quan Pearson
        try:
            r, _ = pearsonr(pred_clean, gt_clean)
        except Exception:
            r = np.nan

        mean_error = np.mean(pred_clean - gt_clean)

        # Tính SNR
        if rppg_signal is not None and fps is not None and len(gt_clean) > 0:
            mean_gt_hr = float(np.nanmean(gt_clean))
            snr = calculate_snr(rppg_signal, fps, mean_gt_hr)
        else:
            snr = np.nan

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "pearson_r": float(r),
            "mean_error": float(mean_error),
            "snr": float(snr),
        }
    except Exception as e:
        print(f"Lỗi khi đánh giá chỉ số: {e}")
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "pearson_r": np.nan,
            "mean_error": np.nan,
            "snr": np.nan,
        }
