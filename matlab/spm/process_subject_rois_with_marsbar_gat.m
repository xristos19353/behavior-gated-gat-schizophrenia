function process_subject_rois_with_marsbar_gat(roi_file)
% PROCESS_SUBJECT_ROIS_WITH_MARSBAR_GCN  Build subject-specific peak ROIs.
%
%   PROCESS_SUBJECT_ROIS_WITH_MARSBAR_GCN(ROI_FILE) uses ROI_FILE as the group
%   ROI image. When called with no argument it falls back to the precomputed
%   consensus atlas in gat_config (cfg.atlas_roi_file).
%
%   For every subject and every label in the group atlas, this function finds
%   the voxel of peak first-level activation inside that ROI, draws a MarsBaR
%   sphere around it, intersects the sphere with the group ROI, and finally
%   merges all per-ROI masks into a single labelled NIfTI per subject
%   (Merged_ROI_<subject>.nii). These subject-specific ROIs are then used by
%   adjacency_matrix_calc and roi_effectsize.

    cfg = gat_config();
    if nargin < 1 || isempty(roi_file)
        roi_file = cfg.atlas_roi_file;
    end
    contrast_name   = cfg.contrast_name;
    first_level_dir = cfg.first_level_dir;
    sphere_radius   = cfg.sphere_radius;
    output_dir      = cfg.subjects_rois_dir;

    marsbar('on');

    if exist(output_dir, 'dir')
        rmdir(output_dir, 's');   % start from a clean output directory
    end
    mkdir(output_dir);

    % Load the group atlas and list its labels (excluding background).
    roi_nii  = spm_vol(roi_file);
    roi_data = spm_read_vols(roi_nii);
    unique_rois = unique(roi_data);
    unique_rois(unique_rois == 0) = [];

    subject_dirs = dir(fullfile(first_level_dir, 'sub-*'));

    for s = 1:length(subject_dirs)
        subject_id        = subject_dirs(s).name;
        subject_output_dir = fullfile(output_dir, subject_id, 'Peak_ROIs');
        if ~exist(subject_output_dir, 'dir')
            mkdir(subject_output_dir);
        end

        spm_mat_file = fullfile(first_level_dir, subject_id, 'SPM.mat');
        if ~exist(spm_mat_file, 'file')
            fprintf('Skipping %s: no SPM.mat file found.\n', subject_id);
            continue;
        end

        load(spm_mat_file, 'SPM');
        contrast_idx = find(strcmp({SPM.xCon.name}, contrast_name));
        if isempty(contrast_idx)
            fprintf('Skipping %s: contrast "%s" not found.\n', subject_id, contrast_name);
            continue;
        end

        contrast_img = fullfile(first_level_dir, subject_id, ...
                                sprintf('spmT_%04d.nii', contrast_idx));
        if ~exist(contrast_img, 'file')
            fprintf('Skipping %s: contrast image not found (%s).\n', subject_id, contrast_img);
            continue;
        end

        contrast_nii  = spm_vol(contrast_img);
        contrast_data = spm_read_vols(contrast_nii);

        % --- One spherical ROI per atlas label ---------------------------
        for i = 1:length(unique_rois)
            roi_label = unique_rois(i);
            roi_mask  = (roi_data == roi_label);
            roi_values = contrast_data(roi_mask);

            if isempty(roi_values) || all(roi_values == 0)
                fprintf('No activation in ROI %d for subject %s.\n', roi_label, subject_id);
                continue;
            end

            [~, peak_idx] = max(roi_values);
            roi_indices   = find(roi_mask);
            [x, y, z]     = ind2sub(size(roi_data), roi_indices(peak_idx));
            peak_coords   = roi_nii.mat * [x; y; z; 1];   % voxel -> MNI

            roi_name = sprintf('ROI_%d_peak_%s.mat', roi_label, subject_id);
            roi_path = fullfile(subject_output_dir, roi_name);

            roi = maroi_sphere(struct('centre', peak_coords(1:3)', 'radius', sphere_radius));
            roi = label(roi, sprintf('ROI_%d', roi_label));

            % Intersect the sphere with the original group ROI.
            roi_masked = roi & roi_mask;
            saveroi(roi_masked, roi_path);
            fprintf('Saved ROI for subject %s, ROI %d at [%0.2f, %0.2f, %0.2f]\n', ...
                    subject_id, roi_label, peak_coords(1), peak_coords(2), peak_coords(3));

            % Convert the MarsBaR ROI to a NIfTI image.
            nii_name     = fullfile(subject_output_dir, [erase(roi_name, '.mat'), '.nii']);
            loaded_roi   = load(roi_path);
            mars_rois2img(loaded_roi.roi, nii_name, [], 'i');
            fprintf('Converted: %s -> %s\n', roi_path, nii_name);
        end

        % --- Merge per-ROI NIfTI masks into one labelled image -----------
        roi_files = dir(fullfile(subject_output_dir, 'ROI*.nii'));
        roi_files = reshape({roi_files.name}, 1, []);
        if isempty(roi_files)
            fprintf('No ROI files generated for %s; skipping merge.\n', subject_id);
            continue;
        end

        first_roi_path = fullfile(subject_output_dir, roi_files{1});
        try
            V = spm_vol(first_roi_path);
        catch ME
            disp(['Error loading file: ', first_roi_path]);
            rethrow(ME);
        end

        merged_data = spm_read_vols(V);
        first_roi_number = sscanf(roi_files{1}, 'ROI_%d');
        merged_data(merged_data > 0) = first_roi_number;

        for i = 2:length(roi_files)
            roi_file_path = fullfile(subject_output_dir, roi_files{i});
            try
                V_i  = spm_vol(roi_file_path);
                data = spm_read_vols(V_i);
            catch
                disp(['Error loading file: ', roi_file_path]);
                continue;
            end
            roi_number = sscanf(roi_files{i}, 'ROI_%d');
            % Label only voxels not already assigned (avoid overwriting).
            merged_data(data > 0 & merged_data == 0) = roi_number;
        end

        V.fname = fullfile(subject_output_dir, sprintf('Merged_ROI_%s.nii', subject_id));
        spm_write_vol(V, merged_data);
    end
end
