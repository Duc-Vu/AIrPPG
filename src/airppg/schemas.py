"""Shared schema constants for AIrPPG artifacts."""

PREPROCESSING_SCHEMA_VERSION = "preprocessing_roi_v1"
PREPROCESSING_GT_SCHEMA_VERSION = "preprocessing_ground_truth_v1"
PREPROCESSING_MANIFEST_SCHEMA_VERSION = "preprocessing_dataset_manifest_v1"
SIGNAL_EXTRACTION_SCHEMA_VERSION = "signal_extraction_rppg_v1"
FUSION_SCHEMA_VERSION = "fusion_v1"
MODELING_SCHEMA_VERSION = "modeling_v1"


ROI_NAMES = ("forehead", "left_cheek", "right_cheek")
METHOD_NAMES = ("green", "chrom", "pos")
SPLIT_NAMES = ("train", "val", "test")
VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")
DATASET_NAME = "UBFC-rPPG"
