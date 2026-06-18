function T = adjacency_matrix_calc(roi_labels_all)
% ADJACENCY_MATRIX_CALC  Build per-subject functional connectivity and metrics.
%
%   T = ADJACENCY_MATRIX_CALC(ROI_LABELS_ALL) computes, for every subject, the
%   ROI-by-ROI functional connectivity matrix (Fisher-z transformed Pearson
%   correlation of mean BOLD time series), together with sparsified binary and
%   weighted adjacency matrices and three node-level metrics (ALFF, ReHo and
%   weighted degree centrality).
%
%   Input:
%       roi_labels_all - vector of ROI label values defining the common node set.
%
%   Output:
%       T - table with one row per subject and the columns:
%           SubjectID, ZInfo, Zbinary, Zweighted, ALFFInfo, ReHoInfo, DCInfo.
%
%   Requires pre-extracted BOLD .mat files (see preprocessing/bold_signal_creation.m,
%   alff_calculation.m, reho_calculation.m) and per-subject merged ROI masks.

    cfg = gat_config();
    output_dir      = cfg.subjects_rois_dir;
    first_level_dir = cfg.first_level_dir;
    mat_dir         = cfg.bold_dir;

    % Clean, sorted, unique ROI label set.
    roi_labels_all = sort(roi_labels_all(:));
    roi_labels_all = roi_labels_all(~isnan(roi_labels_all) & roi_labels_all > 0);
    roi_labels_all = unique(roi_labels_all, 'stable');
    roi_names_all  = arrayfun(@(x) sprintf('ROI_%d', x), roi_labels_all, ...
                              'UniformOutput', false);

    subject_dirs = dir(fullfile(first_level_dir, 'sub-*'));
    n_subjects   = length(subject_dirs);

    subject_ids          = strings(n_subjects, 1);
    Z_all_binary         = cell(n_subjects, 1);
    Z_all_weighted       = cell(n_subjects, 1);
    Z_all                = cell(n_subjects, 1);
    ALFF_all             = cell(n_subjects, 1);
    ReHo_all             = cell(n_subjects, 1);
    DegreeCentrality_all = cell(n_subjects, 1);

    for s = 1:n_subjects
        subject_id = subject_dirs(s).name;
        mat_file   = fullfile(mat_dir, [subject_id '.mat']);

        if ~isfile(mat_file)
            warning('Missing BOLD .mat file: %s', mat_file);
            continue
        end

        roi_file = fullfile(output_dir, subject_id, 'Peak_ROIs', ...
                            ['Merged_ROI_' subject_id '.nii']);
        if ~isfile(roi_file)
            warning('Missing ROI file for %s', subject_id);
            continue
        end

        S            = load(mat_file);
        Y_bold       = S.Y_bold;          % 4D volume: X x Y x Z x T
        T_len        = size(Y_bold, 4);
        ALFF_voxel_z = S.ALFF_voxel_z;
        reho_z       = S.reho_z;

        V_roi      = spm_vol(roi_file);
        roi_labels = spm_read_vols(V_roi);

        unique_labels = unique(roi_labels(:));
        unique_labels = unique_labels(unique_labels > 0 & ~isnan(unique_labels));
        n_rois        = length(roi_labels_all);

        roi_names = strings(n_rois, 1);
        for r = 1:n_rois
            roi_names(r) = "ROI_" + roi_labels_all(r);
        end

        % --- Mean BOLD time series per ROI -------------------------------
        roi_mask    = roi_labels > 0 & ~isnan(roi_labels);
        roi_indices = find(roi_mask);
        Y_2D        = reshape(Y_bold, [], T_len);
        Y_roi       = Y_2D(roi_indices, :);
        roi_voxel_labels = roi_labels(roi_indices);
        clear Y_2D Y_bold;

        bold_avg = zeros(T_len, n_rois);
        for r = 1:n_rois
            roi_label = roi_labels_all(r);
            r_idx     = roi_voxel_labels == roi_label;
            if any(r_idx(:))
                bold_avg(:, r) = mean(Y_roi(r_idx, :)', 2, 'omitnan');
            else
                bold_avg(:, r) = 0;
            end
        end
        clear Y_roi;

        % --- Functional connectivity (Fisher-z) --------------------------
        R = corr(bold_avg, 'Rows', 'pairwise');
        R = max(min(R, 0.9999), -0.9999);   % clip to avoid +/-Inf after atanh
        Z = atanh(R);
        Z(isnan(Z)) = 0;
        Z(logical(eye(size(Z)))) = 0;

        Z_binary   = sparsify_adjacency(Z, true);
        Z_weighted = sparsify_adjacency(Z, false);

        Z_all{s}          = struct('Z', Z, 'ROINames', roi_names);
        Z_all_binary{s}   = struct('Zbinary', Z_binary, 'ROINames', roi_names);
        Z_all_weighted{s} = struct('Zweighted', Z_weighted, 'ROINames', roi_names);

        % --- Weighted degree centrality ----------------------------------
        DC_weighted = sum(abs(Z_weighted), 2);
        clear Z;

        % --- ROI-wise ALFF, ReHo and degree centrality -------------------
        alff_struct = struct();
        reho_struct = struct();
        dc_struct   = struct();

        for i = 1:n_rois
            roi_label = roi_labels_all(i);
            roi_name  = roi_names_all{i};
            roi_mask_r = roi_labels == roi_label;

            if any(roi_mask_r(:))
                alff_struct.(roi_name) = mean(ALFF_voxel_z(roi_mask_r), 'omitnan');
                reho_struct.(roi_name) = mean(reho_z(roi_mask_r), 'omitnan');
                if isnan(reho_struct.(roi_name))
                    reho_struct.(roi_name) = 0;
                end

                match_idx = find(unique_labels == roi_label, 1);
                if ~isempty(match_idx)
                    dc_struct.(roi_name) = DC_weighted(i);
                else
                    dc_struct.(roi_name) = 0;
                end
            else
                alff_struct.(roi_name) = 0;
                reho_struct.(roi_name) = 0;
                dc_struct.(roi_name)   = 0;
            end
        end

        ALFF_all{s}             = alff_struct;
        ReHo_all{s}             = reho_struct;
        DegreeCentrality_all{s} = dc_struct;
        subject_ids(s)          = subject_id;

        fprintf('Processed subject %d (%s): connectivity + ALFF + ReHo\n', ...
                s, subject_id);
    end

    T = table(subject_ids, Z_all, Z_all_binary, Z_all_weighted, ...
              ALFF_all, ReHo_all, DegreeCentrality_all, ...
              'VariableNames', {'SubjectID', 'ZInfo', 'Zbinary', 'Zweighted', ...
                                'ALFFInfo', 'ReHoInfo', 'DCInfo'});

end
