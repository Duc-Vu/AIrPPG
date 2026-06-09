import json, sys, traceback
from pathlib import Path
import numpy as np
try:
    old_pre=Path('outputs/task1_preprocessing'); new_pre=Path('outputs/preprocessing'); old_sig=Path('outputs/task2_signals'); new_sig=Path('outputs/signal_extraction')
    def load_json(p): return json.loads(p.read_text(encoding='utf-8'))
    def stat(a):
        a=np.asarray(a); out=f'shape={a.shape} dtype={a.dtype}'
        if a.dtype.kind in 'biufc' and a.size:
            if a.dtype.kind == 'b': vals=a.reshape(-1); out += f' true={int(vals.sum())} false={int(vals.size-vals.sum())}'
            else:
                finite=np.isfinite(a) if a.dtype.kind in 'fc' else np.ones(a.shape,bool); vals=a[finite] if a.dtype.kind in 'fc' else a.reshape(-1)
                if vals.size: out += f' min={float(np.min(vals)):.6g} max={float(np.max(vals)):.6g} mean={float(np.mean(vals)):.6g} nan={int((~finite).sum()) if a.dtype.kind in "fc" else 0}'
        return out
    def cmp_array(label,a,b):
        a=np.asarray(a); b=np.asarray(b); same_shape=a.shape==b.shape; same_dtype=str(a.dtype)==str(b.dtype); diff='n/a'
        if a.dtype.kind in 'SUO' or b.dtype.kind in 'SUO': same=same_shape and np.array_equal(a.astype(str), b.astype(str))
        elif a.dtype.kind == 'b' or b.dtype.kind == 'b' or a.dtype == np.uint8 or b.dtype == np.uint8:
            same=same_shape and np.array_equal(a.astype(np.uint8), b.astype(np.uint8)); diff=0 if same else ('n/a' if not same_shape else int(np.count_nonzero(a.astype(np.uint8)!=b.astype(np.uint8))))
        else:
            same=same_shape and np.allclose(a,b,equal_nan=True); diff='n/a' if not same_shape else (float(np.nanmax(np.abs(a.astype(np.float32)-b.astype(np.float32)))) if a.size else 0.0)
        print(f'{label}: same_shape={same_shape} same_dtype={same_dtype} same_values={same} diff={diff}')
        if not same: print('  old',stat(a)); print('  new',stat(b))
    def load_npz(p):
        with np.load(p, allow_pickle=True) as d: return {k:d[k] for k in d.files}
    old_pre_m=load_json(old_pre/'task1_dataset_manifest.json'); new_pre_m=load_json(new_pre/'preprocessing_dataset_manifest.json')
    print('PRE counts old/new:', old_pre_m['counts'], new_pre_m['counts'])
    sample=sorted(set(i['sample_id'] for i in old_pre_m['items']) & set(i['sample_id'] for i in new_pre_m['items']))[0]
    oi={i['sample_id']:i for i in old_pre_m['items']}[sample]; ni={i['sample_id']:i for i in new_pre_m['items']}[sample]
    print('PRE sample:', sample)
    old_roi=load_npz(old_pre/oi['roi_npz']); new_roi=load_npz(new_pre/ni['roi_npz'])
    print('PRE roi keys old-only:', sorted(set(old_roi)-set(new_roi))); print('PRE roi keys new-only:', sorted(set(new_roi)-set(old_roi)))
    for label,ok,nk in [('frame_indices','frame_indices','frame_indices'),('valid','valid','valid'),('roi_names','roi_names','roi_names'),('landmarks alias','landmarks_px','landmarks'),('roi_polygons alias','roi_polygons_px','roi_polygons'),('roi_masks','roi_masks','roi_masks')]: cmp_array('pre.'+label, old_roi[ok], new_roi[nk])
    old_gt=load_npz(old_pre/oi['ground_truth_npz']); new_gt=load_npz(new_pre/ni['ground_truth_npz'])
    for k in ['timestamps','heart_rate_bpm','ppg_signal','ppg_timestamps','spo2_percent']: cmp_array('gt.'+k, old_gt[k], new_gt[k])
    old_meta=load_json(old_pre/oi['metadata_json']); new_meta=load_json(new_pre/ni['metadata_json'])
    for k in ['valid_frame_count','invalid_frame_count','failure_counts','roi_names']:
        print(f'metadata.{k}: old={old_meta.get(k)} new={new_meta.get(k)} same={old_meta.get(k)==new_meta.get(k)}')
    print('metadata.schema old/new', old_meta.get('schema_version'), new_meta.get('schema_version'))

    old_sig_m=load_json(old_sig/'task2_dataset_manifest.json'); new_sig_m=load_json(new_sig/'signal_extraction_dataset_manifest.json')
    print('SIG counts old/new:', old_sig_m['counts'], new_sig_m['counts'])
    sample=sorted(set(i['sample_id'] for i in old_sig_m['items']) & set(i['sample_id'] for i in new_sig_m['items']))[0]
    os={i['sample_id']:i for i in old_sig_m['items']}[sample]; ns={i['sample_id']:i for i in new_sig_m['items']}[sample]
    print('SIG sample:', sample)
    old_s=load_npz(old_sig/os['outputs']['signals_npz']); new_s=load_npz(new_sig/ns['outputs']['signals_npz'])
    for k in ['frame_indices','timestamps','valid','roi_names','method_names','mean_rgb','rppg_signals','rppg_roi_signals','hr_timestamps','hr_estimates','gt_timestamps','gt_heart_rate_bpm','gt_aligned_to_hr','ppg_timestamps','ppg_signal','spo2_percent']: cmp_array('sig.'+k, old_s[k], new_s[k])
    for method in ['green','chrom','pos']:
        print('metric',method)
        for m in ['mae_bpm','rmse_bpm','pearson_r','bias_bpm','snr_db','n_windows']:
            ov=os['metrics'][method][m]; nv=ns['metrics'][method][m]; same=(ov==nv) or (isinstance(ov,float) and isinstance(nv,float) and abs(ov-nv)<1e-9)
            print(f'  {m}: old={ov} new={nv} same={same}')
except Exception:
    traceback.print_exc(); sys.exit(1)
