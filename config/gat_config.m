function cfg = gat_config()
% GCN_CONFIG  Central configuration for the MATLAB (SPM/MarsBaR) pipeline.
%
%   cfg = GCN_CONFIG() returns a struct with every machine-specific path used
%   by the MATLAB scripts. Edit the values in the "EDIT THESE FOR YOUR MACHINE"
%   block below; no other MATLAB file should contain a hard-coded absolute path.
%
%   Add the folder that contains this file to your MATLAB path, e.g.
%       addpath('/path/to/repo/config');
%   and then call gat_config() from any script.

% -------------------------------------------------------------------------
% EDIT THESE FOR YOUR MACHINE
% -------------------------------------------------------------------------
% Root of the study data on disk.
cfg.data_root = '/path/to/Functional_connectivity_study';

% Root for all pipeline outputs.
cfg.output_root = fullfile(cfg.data_root, 'GCN_output_files');

% SPM first-level directory (one sub-* folder per subject, each with SPM.mat).
cfg.first_level_dir = fullfile(cfg.data_root, 'analysis', 'SPM_FirstLevel');

% Per-subject anatomical / tissue-density data (GMV / WMV images).
cfg.participants_data = fullfile(cfg.data_root, 'Participants_Data');

% Folder with per-subject behavioural spreadsheets (<subject>.xlsx).
cfg.behavioural_dir = fullfile(cfg.data_root, 'behaviourals');

% Demographics table (loads a variable 'demo' with ID, SEX, EDU, AGE).
cfg.demographics_mat = fullfile(cfg.data_root, 'demographics.mat');

% Pre-extracted BOLD signal .mat files (one per subject; see preprocessing/).
cfg.bold_dir = fullfile(cfg.output_root, 'Subjects_BOLD_signals');

% Reference atlas / group ROI image used to seed subject-specific ROIs
% (precomputed-consensus mode; see spm_2level_gat for details).
cfg.atlas_roi_file = fullfile(cfg.output_root, 'ROIs', 'consensus_labeled_rois.nii');

% Anatomical parcellation used to refine peak ROIs (image + label text file).
cfg.parcellation_atlas  = fullfile(cfg.data_root, 'atlas', 'Schaefer100_HOsub_Hybrid.nii');
cfg.parcellation_labels = fullfile(cfg.data_root, 'atlas', 'Schaefer100_HOsub_Hybrid_labels.txt');

% Minimum ROI / parcellation overlap (percent) to keep a sub-ROI.
cfg.overlap_percent = 20;

% Thresholded second-level activation map used to mask the final ROIs.
cfg.thresholded_activation = fullfile(cfg.output_root, 'thresholded.nii');
cfg.activation_threshold   = 0;

% Reference image used to reslice tissue-density maps to a common space
% (gmv_wmv_calc). Any image in the target space works.
cfg.reslice_reference = cfg.atlas_roi_file;

% -------------------------------------------------------------------------
% DERIVED PATHS (usually no need to edit)
% -------------------------------------------------------------------------
% Two-sample t-test working directory.
cfg.two_sample_dir = fullfile(cfg.output_root, 'two-sample t-tests');

% Per-subject extracted ROI masks.
cfg.subjects_rois_dir = fullfile(cfg.output_root, 'Subjects_ROIs');

% Group-level ROI directory written by SPM_ROI_extraction_GCN.
cfg.group_rois_dir = fullfile(cfg.output_root, 'ROIs');

% Contrast name used at the first level.
cfg.contrast_name = 'Flanker > Rest - All Sessions';

% Default MarsBaR sphere radius (mm).
cfg.sphere_radius = 4;

end
