import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import pearsonr

class Task3MultiROIFusion:
    def __init__(self, task2_manifest_path, output_dir, fps=30):
        self.manifest_path = task2_manifest_path
        self.output_dir = output_dir
        self.fps = fps
        self.min_hr_f, self.max_hr_f = 45 / 60.0, 240 / 60.0
        
        # Hyperparameters cho Quality Score (Rule-based)
        self.w_a = 1.0   # snr
        self.w_b = 0.5   # peak clarity
        self.w_c = 2.0   # valid ratio
        self.w_d = 1.5   # inter-roi disagreement
        self.w_e = 0.5   # landmark jitter

    def _extract_hr_and_psd(self, signal):
        f, pxx = welch(signal, fs=self.fps, nperseg=len(signal))
        valid_idx = np.where((f >= self.min_hr_f) & (f <= self.max_hr_f))[0]
        if len(valid_idx) == 0:
            return 0.0, f, pxx, 0
        
        valid_f, valid_pxx = f[valid_idx], pxx[valid_idx]
        peak_idx = np.argmax(valid_pxx)
        hr = valid_f[peak_idx] * 60.0
        return hr, f, pxx, valid_f[peak_idx]

    def calc_quality_features(self, signals, valid_mask, prev_hrs=None, jitters=None):
        num_rois = signals.shape[0]
        features, hrs = [], []
        
        for i in range(num_rois):
            sig = signals[i]
            hr, f, pxx, peak_f = self._extract_hr_and_psd(sig)
            hrs.append(hr)
            
            # Tính SNR & Peak Clarity
            signal_band = np.where((f >= peak_f - 0.1) & (f <= peak_f + 0.1))[0]
            sig_power = np.sum(pxx[signal_band])
            noise_power = np.sum(pxx) - sig_power
            snr = 10 * np.log10(sig_power / noise_power) if noise_power > 0 else 0
            
            valid_idx = np.where((f >= self.min_hr_f) & (f <= self.max_hr_f))[0]
            valid_pxx = pxx[valid_idx]
            noise_pxx = np.delete(valid_pxx, np.argmax(valid_pxx))
            clarity = np.max(valid_pxx) / (np.max(noise_pxx) + 1e-6) if len(noise_pxx) > 0 else 1.0
            
            stability = 1.0 / (abs(hr - prev_hrs[i]) + 0.1) if prev_hrs else 1.0
            sig_std = np.std(sig)
            valid_ratio = np.mean(valid_mask)
            jitter = jitters[i] if jitters else 0.0
            
            features.append({
                'snr_db': max(snr, 0),
                'peak_clarity': clarity,
                'peak_stability': stability,
                'signal_std': sig_std,
                'valid_ratio': valid_ratio,
                'landmark_jitter': jitter
            })
            
        disagreement = np.var(hrs)
        for feat in features:
            feat['inter_roi_disagreement'] = disagreement
            
        return features, hrs

    def calculate_weights(self, features):
        weights = []
        for feat in features:
            if feat['valid_ratio'] < 0.3: # Fallback: ROI bị che -> weight = 0
                weights.append(0.0)
                continue
                
            q_score = (self.w_a * feat['snr_db'] + 
                       self.w_b * feat['peak_clarity'] + 
                       self.w_c * feat['valid_ratio'] - 
                       self.w_d * feat['inter_roi_disagreement'] - 
                       self.w_e * feat['landmark_jitter'])
            weights.append(max(q_score, 0.0))
            
        weights = np.array(weights)
        total_w = np.sum(weights)
        
        # Edge Case Fallback: Toàn bộ quality hỏng -> Average fusion
        if total_w == 0 or np.isnan(total_w):
            return np.ones(len(features)) / len(features)
        return weights / total_w

    def process_sample(self, sample_data, sample_id, split):
        rppg = sample_data['rppg_roi_signals']
        roi_names = sample_data['roi_names']
        method_names = sample_data['method_names']
        gt_hr = sample_data['gt_aligned_to_hr']
        valid_mask = sample_data['valid']
        
        num_methods, frames, num_rois = rppg.shape
        window_size = self.fps * 10
        num_windows = frames // window_size
        
        single_roi_hr = np.zeros((num_windows, num_methods, num_rois))
        avg_fusion_hr = np.zeros((num_windows, num_methods))
        weight_fusion_hr = np.zeros((num_windows, num_methods))
        roi_weights_all = np.zeros((num_windows, num_methods, num_rois))
        
        # Array lưu trữ toàn bộ fused signals (dùng cho Task 4)
        fused_rppg_signals = np.zeros((num_methods, frames))
        roi_features_all = []
        
        prev_hrs = [[70]*num_rois] * num_methods
        
        for w in range(num_windows):
            start, end = w * window_size, (w + 1) * window_size
            w_valid = valid_mask[start:end]
            
            for m in range(num_methods):
                signals_window = rppg[m, start:end, :].T 
                
                # 1. Single ROI
                features, s_hrs = self.calc_quality_features(signals_window, w_valid, prev_hrs[m])
                single_roi_hr[w, m, :] = s_hrs
                prev_hrs[m] = s_hrs
                
                # 2. Average Fusion
                avg_sig = np.mean(signals_window, axis=0)
                avg_fusion_hr[w, m], _, _, _ = self._extract_hr_and_psd(avg_sig)
                
                # 3. Quality-weighted Fusion
                weights = self.calculate_weights(features)
                roi_weights_all[w, m, :] = weights
                
                weighted_sig = np.zeros_like(avg_sig)
                for r in range(num_rois):
                    weighted_sig += weights[r] * signals_window[r]
                    
                weight_fusion_hr[w, m], _, _, _ = self._extract_hr_and_psd(weighted_sig)
                fused_rppg_signals[m, start:end] = weighted_sig # Lưu dạng sóng gộp
                
                if m == 0:
                    roi_features_all.append(features)

        metrics = self._calculate_metrics(single_roi_hr, avg_fusion_hr, weight_fusion_hr, gt_hr[:num_windows], num_methods, method_names, roi_names)
        
        self._save_outputs(sample_id, split, single_roi_hr, avg_fusion_hr, weight_fusion_hr, roi_weights_all, roi_features_all, fused_rppg_signals, metrics, gt_hr, method_names, roi_names)
        return metrics

    def _calculate_metrics(self, single, avg, weighted, gt, num_methods, m_names, r_names):
        metrics = []
        for m in range(num_methods):
            for r in range(len(r_names)):
                metrics.append(self._get_metric_dict(f"Single_{r_names[r]}_{m_names[m]}", single[:, m, r], gt))
            metrics.append(self._get_metric_dict(f"Average_{m_names[m]}", avg[:, m], gt))
            metrics.append(self._get_metric_dict(f"Weighted_{m_names[m]}", weighted[:, m], gt))
        return pd.DataFrame(metrics)

    def _get_metric_dict(self, name, pred, gt):
        valid = ~np.isnan(pred) & ~np.isnan(gt)
        if sum(valid) < 2: return {'Method': name, 'MAE': 0, 'RMSE': 0, 'Pearson': 0, 'Bias': 0}
        p, g = pred[valid], gt[valid]
        mae = np.mean(np.abs(p - g))
        rmse = np.sqrt(np.mean((p - g)**2))
        pearson, _ = pearsonr(p, g)
        bias = np.mean(p - g)
        return {'Method': name, 'MAE': mae, 'RMSE': rmse, 'Pearson': pearson, 'Bias': bias}

    def _save_outputs(self, sample_id, split, single, avg, weighted, weights, features, fused_signals, metrics_df, gt, m_names, r_names):
        save_path = os.path.join(self.output_dir, split, sample_id)
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Save NPZ (Đã bổ sung fused_rppg_signals)
        np.savez(os.path.join(save_path, 'fusion_signals.npz'),
                 sample_id=sample_id, split=split, roi_names=r_names, method_names=m_names,
                 single_roi_hr=single, average_fusion_hr=avg, weighted_fusion_hr=weighted,
                 roi_weights=weights, fused_rppg_signals=fused_signals, gt_aligned_to_hr=gt)
        
        # 2. Save Metrics
        metrics_df.to_csv(os.path.join(save_path, 'fusion_metrics.csv'), index=False)
        metrics_df.to_json(os.path.join(save_path, 'fusion_metrics.json'), orient='records')
        
        # 3. Visualization
        self._plot_visualizations(save_path, sample_id, single, avg, weighted, weights, features, metrics_df, gt, m_names, r_names)

    def _plot_visualizations(self, path, sample_id, single, avg, weighted, weights, features, metrics_df, gt, m_names, r_names):
        windows = range(len(single))
        m_idx = 0 
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 20))
        
        # Plot 1: HR Tracking
        axes[0].plot(windows, gt[:len(windows)], 'k--', linewidth=2, label='Ground Truth')
        for r in range(len(r_names)):
            axes[0].plot(windows, single[:, m_idx, r], alpha=0.5, label=f'Single: {r_names[r]}')
        axes[0].plot(windows, avg[:, m_idx], 'g-.', label='Avg Fusion')
        axes[0].plot(windows, weighted[:, m_idx], 'r-', linewidth=2, label='Weighted Fusion')
        axes[0].set_title(f'HR Tracking - {sample_id}')
        axes[0].set_ylabel('BPM')
        axes[0].legend()
        
        # Plot 2: Weights
        for r in range(len(r_names)):
            axes[1].plot(windows, weights[:, m_idx, r], marker='o', label=f'Weight: {r_names[r]}')
        axes[1].set_title('ROI Weights Over Time')
        axes[1].set_ylabel('Weight')
        axes[1].legend()
        
        # Plot 3: Quality Features (SNR tracking)
        for r in range(len(r_names)):
            snr_values = [feat[r]['snr_db'] for feat in features]
            axes[2].plot(windows, snr_values, marker='s', label=f'SNR: {r_names[r]}')
        axes[2].set_title('ROI Quality (SNR) Over Time')
        axes[2].set_ylabel('SNR (dB)')
        axes[2].legend()
        
        # Plot 4: Bar Chart so sánh MAE
        methods_to_plot = [f"Single_forehead_{m_names[m_idx]}", f"Average_{m_names[m_idx]}", f"Weighted_{m_names[m_idx]}"]
        display_names = ['Single (Forehead)', 'Average Fusion', 'Weighted Fusion']
        maes = []
        for method in methods_to_plot:
            match = metrics_df[metrics_df['Method'] == method]['MAE']
            maes.append(match.values[0] if not match.empty else 0)
            
        axes[3].bar(display_names, maes, color=['blue', 'green', 'red'])
        axes[3].set_title('Mean Absolute Error (MAE) Comparison')
        axes[3].set_ylabel('MAE (BPM) - Lower is better')
        for i, v in enumerate(maes):
            axes[3].text(i, v + 0.1, f'{v:.2f}', ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(path, 'fusion_visualization.png'))
        plt.close()

    def run_pipeline(self):
        """Pipeline chuẩn: Đọc từ JSON manifest và loop qua toàn bộ dataset"""
        print(f"Reading manifest from: {self.manifest_path}")
        with open(self.manifest_path, 'r') as f:
            manifest = json.load(f)
            
        all_metrics = []
        
        for split, samples in manifest.items():
            for sample_id, path in samples.items():
                print(f"Processing {split} / {sample_id}...")
                data = np.load(path, allow_pickle=True)
                sample_data = {
                    'rppg_roi_signals': data['rppg_roi_signals'],
                    'roi_names': data['roi_names'],
                    'method_names': data['method_names'],
                    'gt_aligned_to_hr': data['gt_aligned_to_hr'],
                    'valid': data['valid']
                }
                
                metrics_df = self.process_sample(sample_data, sample_id, split)
                metrics_df['Sample_ID'] = sample_id
                metrics_df['Split'] = split
                all_metrics.append(metrics_df)
                
        # Tổng hợp toàn bộ kết quả của Task 3
        final_df = pd.concat(all_metrics)
        summary_path = os.path.join(self.output_dir, 'task3_fusion_metrics_summary.csv')
        final_df.to_csv(summary_path, index=False)
        print(f"\n[SUCCESS] Hoàn thành Pipeline! Tổng kết lưu tại: {summary_path}")

