function spm_roi_extraction_gat(peakTable, radius)
% SPM_ROI_EXTRACTION_GCN  Build a merged ROI image from second-level peaks.
%
%   SPM_ROI_EXTRACTION_GCN(PEAKTABLE, RADIUS) draws a MarsBaR sphere of the
%   given RADIUS (mm) around each peak coordinate in PEAKTABLE (columns X,Y,Z),
%   converts each sphere to a NIfTI image, and merges them into a single
%   labelled image (merged_roi.nii) in the group ROI directory defined by
%   gat_config (cfg.group_rois_dir). Each peak receives a distinct label.

    cfg     = gat_config();
    roi_dir = cfg.group_rois_dir;

    marsbar('on');
    peak_coords = table2array(peakTable);
    peak_coords = peak_coords(:, 1:3);

    if exist(roi_dir, 'dir'); rmdir(roi_dir, 's'); end
    mkdir(roi_dir);

    % --- One spherical ROI per peak ----------------------------------
    for i = 1:size(peak_coords, 1)
        roi       = maroi_sphere(struct('centre', peak_coords(i, :), 'radius', radius));
        roi_fname = fullfile(roi_dir, sprintf('peak_%d_roi.mat', i));

        roi_data.index = i;
        roi_data.label = sprintf('Peak %d', i);
        roi_data.roi   = roi;
        save(roi_fname, 'roi_data');
    end

    % --- Convert each .mat ROI to NIfTI -------------------------------
    roi_files = dir(fullfile(roi_dir, 'peak_*_roi.mat'));
    for i = 1:length(roi_files)
        roi_name    = fullfile(roi_dir, roi_files(i).name);
        [~, name]   = fileparts(roi_name);
        nii_name    = fullfile(roi_dir, [name, '.nii']);
        loaded_data = load(roi_name);
        mars_rois2img(loaded_data.roi_data.roi, nii_name, [], 'i');
        fprintf('Converted: %s -> %s\n', roi_name, nii_name);
    end

    % --- Merge all peak ROIs into one labelled image ------------------
    roi_files_struct = dir(fullfile(roi_dir, 'peak*.nii'));
    roi_files = fullfile({roi_files_struct.folder}, {roi_files_struct.name});

    V           = spm_vol(roi_files{1});
    merged_data = spm_read_vols(V);
    merged_data(merged_data > 0) = 1;

    roi_label = 2;
    for i = 2:length(roi_files)
        V_i  = spm_vol(roi_files{i});
        data = spm_read_vols(V_i);
        merged_data(data > 0) = roi_label;
        roi_label = roi_label + 1;
    end

    V.fname = fullfile(roi_dir, 'merged_roi.nii');
    spm_write_vol(V, merged_data);
end
