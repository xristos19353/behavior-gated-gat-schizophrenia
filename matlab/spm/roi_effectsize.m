function newTable = roi_effectsize(roi_labels_all)
% ROI_EFFECTSIZE  Collect per-subject ROI effect sizes, tissue density and behaviour.
%
%   newTable = ROI_EFFECTSIZE(ROI_LABELS_ALL) loops over every subject and, for
%   each ROI in ROI_LABELS_ALL, extracts:
%       * the mean first-level contrast value (effect size),
%       * mean grey-matter volume (GMV) and white-matter volume (WMV),
%   then merges these with the subject's behavioural metrics and demographics.
%
%   The result is one row per subject, with behavioural columns plus struct
%   columns (EffectSize, GMV, WMV, Demographics, Behavior) holding ROI-wise and
%   subject-level values, ready to be consumed by the Python pipeline.

    cfg = gat_config();
    first_level_dir   = cfg.first_level_dir;
    output_dir        = cfg.subjects_rois_dir;
    behavioural_dir   = cfg.behavioural_dir;
    gmv_base_path     = cfg.participants_data;
    contrast_name     = cfg.contrast_name;

    load(cfg.demographics_mat);  % provides 'demo' table (ID, SEX, EDU, AGE)

    subject_dirs   = dir(fullfile(first_level_dir, 'sub-*'));
    roi_labels_all = sort(roi_labels_all(:));

    roi_effect_size_table = table();
    behavioral_data_table = table();

    % ===================== 1. Loop over subjects =========================
    for s = 1:length(subject_dirs)
        subject_id  = subject_dirs(s).name;
        subject_dir = fullfile(first_level_dir, subject_id);
        spm_mat_file = fullfile(subject_dir, 'SPM.mat');

        if ~exist(spm_mat_file, 'file')
            warning('Missing SPM.mat: %s', subject_id); continue;
        end

        % --- Locate first-level contrast image ---------------------------
        load(spm_mat_file, 'SPM');
        contrast_idx = find(strcmp({SPM.xCon.name}, contrast_name), 1);
        if isempty(contrast_idx)
            warning('Contrast not found: %s', subject_id); continue;
        end
        con_image = fullfile(subject_dir, sprintf('con_%04d.nii', contrast_idx));
        if ~exist(con_image, 'file')
            warning('Missing contrast image: %s', subject_id); continue;
        end

        % --- Subject-specific merged ROI mask ----------------------------
        mask_file = fullfile(output_dir, subject_id, 'Peak_ROIs', ...
                             ['Merged_ROI_' subject_id '.nii']);
        if ~exist(mask_file, 'file')
            warning('Missing mask: %s', subject_id); continue;
        end

        V_con  = spm_vol(con_image);  Y_con  = spm_read_vols(V_con);
        V_mask = spm_vol(mask_file);  Y_mask = spm_read_vols(V_mask);
        if ~isequal(size(Y_con), size(Y_mask))
            warning('Dimension mismatch: %s', subject_id); continue;
        end

        % --- GMV / WMV images (subject id without analysis suffix) --------
        subject_id2 = regexprep(subject_id, '_[A-Za-z]+$', '');
        gmv_file = fullfile(gmv_base_path, subject_id2, ...
                            ['Resliced_GMV_mwc1_' subject_id2 '.nii']);
        wmv_file = fullfile(gmv_base_path, subject_id2, ...
                            ['Resliced_WMV_mwc2_' subject_id2 '.nii']);

        if ~exist(gmv_file, 'file')
            warning('Missing GMV file: %s', gmv_file); continue;
        end
        if ~exist(wmv_file, 'file')
            warning('Missing WMV file: %s', wmv_file); continue;
        end

        V_gmv = spm_vol(gmv_file);  Y_gmv = spm_read_vols(V_gmv);
        V_wmv = spm_vol(wmv_file);  Y_wmv = spm_read_vols(V_wmv);

        if ~isequal(size(Y_mask), size(Y_gmv)) || ~isequal(size(Y_mask), size(Y_wmv))
            warning('Mask / tissue image size mismatch: %s', subject_id); continue;
        end

        % --- Per-ROI effect size and tissue density ----------------------
        for roi_label = roi_labels_all'
            roi_mask = (Y_mask == roi_label);

            if any(roi_mask(:))
                roi_vals = Y_con(roi_mask);
                mean_effect_size = mean(roi_vals(~isnan(roi_vals)), 'omitnan');
                if isnan(mean_effect_size); mean_effect_size = 0; end

                gmv_vals  = Y_gmv(roi_mask);
                total_gmv = mean(gmv_vals(~isnan(gmv_vals)), 'omitnan');
                if isempty(gmv_vals(~isnan(gmv_vals))); total_gmv = 0; end

                wmv_vals  = Y_wmv(roi_mask);
                total_wmv = mean(wmv_vals(~isnan(wmv_vals)), 'omitnan');
                if isempty(wmv_vals(~isnan(wmv_vals))); total_wmv = 0; end
            else
                warning('ROI %d not found in subject %s. Assigning zeros.', ...
                        roi_label, subject_id);
                mean_effect_size = 0;
                total_gmv        = 0;
                total_wmv        = 0;
            end

            new_entry = table({subject_id}, roi_label, mean_effect_size, ...
                              total_gmv, total_wmv, ...
                              'VariableNames', {'Subject', 'ROI_Label', ...
                                                'MeanEffectSize', 'GMV', 'WMV'});
            roi_effect_size_table = [roi_effect_size_table; new_entry]; %#ok<AGROW>
        end

        % --- Behavioural metrics -----------------------------------------
        group     = 2 - (startsWith(subject_id, 'sub-4') || startsWith(subject_id, 'sub-3'));
        file_path = fullfile(behavioural_dir, [subject_id2, '.xlsx']);
        if ~isfile(file_path)
            warning('Missing behavioural file: %s', subject_id); continue;
        end

        metric_names  = readcell(file_path, 'Range', 'K4:K9');
        metric_values = readmatrix(file_path, 'Range', 'L4:L9');
        if length(metric_values) ~= length(metric_names)
            warning('Mismatch metrics: %s', subject_id); continue;
        end

        sub_data = array2table(metric_values', ...
            'VariableNames', matlab.lang.makeValidName(metric_names));
        sub_data.Subject = {subject_id};
        sub_data.Group   = group;
        behavioral_data_table = [behavioral_data_table; sub_data]; %#ok<AGROW>
    end

    % ===================== 2. Z-score effect sizes per ROI ================
    roi_effect_size_table.Subject = categorical(roi_effect_size_table.Subject);
    unique_rois = unique(roi_effect_size_table.ROI_Label);
    roi_effect_size_table.ZScoredEffectSize = zeros(height(roi_effect_size_table), 1);
    for roi = unique_rois'
        idx   = roi_effect_size_table.ROI_Label == roi;
        vals  = roi_effect_size_table.MeanEffectSize(idx);
        mu    = mean(vals);
        sigma = std(vals);
        roi_effect_size_table.ZScoredEffectSize(idx) = (sigma > 0) .* ((vals - mu) / sigma);
    end

    % ===================== 3. Merge and reshape ===========================
    behavioral_data_table.Subject = categorical(behavioral_data_table.Subject);
    merged_data = innerjoin(roi_effect_size_table, behavioral_data_table, 'Keys', 'Subject');

    subjects            = string(unique(merged_data.Subject));
    merged_data.Subject = string(merged_data.Subject);

    newTable = table();
    newTable.Subject = subjects;

    metric_columns = setdiff(merged_data.Properties.VariableNames, ...
        {'ROI_Label', 'ZScoredEffectSize', 'Subject', 'MeanEffectSize', 'GMV', 'WMV'});

    for m = 1:numel(metric_columns)
        col = metric_columns{m};
        newTable.(col) = arrayfun(@(s) ...
            merged_data.(col)(find(strcmp(merged_data.Subject, s), 1)), subjects);
    end

    % --- Build per-subject structs -----------------------------------
    EffectSizeStructs = cell(length(subjects), 1);
    GMVStructs        = cell(length(subjects), 1);
    WMVStructs        = cell(length(subjects), 1);
    Demographics      = cell(length(subjects), 1);
    BehaviorStructs   = cell(length(subjects), 1);

    for i = 1:length(subjects)
        subj      = subjects(i);
        subj_data = merged_data(strcmp(merged_data.Subject, subj), :);

        % Demographics by matching the numeric subject id.
        subj_numeric = str2double(regexp(subj, '\d+', 'match', 'once'));
        subj_idx     = find(demo.ID == subj_numeric);
        if isempty(subj_idx)
            warning('Subject %s not found in demographics.', subj); continue;
        end

        demo_struct     = struct();
        demo_struct.AGE = demo.AGE(subj_idx);
        demo_struct.SEX = demo.SEX(subj_idx);
        demo_struct.EDU = demo.EDU(subj_idx);

        eff_struct = struct();
        gmv_struct = struct();
        wmv_struct = struct();
        for r = 1:height(subj_data)
            roi_name = sprintf('ROI_%d', subj_data.ROI_Label(r));
            eff_struct.(roi_name) = subj_data.MeanEffectSize(r);
            gmv_struct.(roi_name) = subj_data.GMV(r);
            wmv_struct.(roi_name) = subj_data.WMV(r);
        end

        beh_struct = struct();
        for m = 1:numel(metric_columns)
            col = metric_columns{m};
            beh_struct.(col) = subj_data.(col)(1);
        end

        EffectSizeStructs{i} = eff_struct;
        GMVStructs{i}        = gmv_struct;
        WMVStructs{i}        = wmv_struct;
        Demographics{i}      = demo_struct;
        BehaviorStructs{i}   = beh_struct;
    end

    newTable.EffectSize   = EffectSizeStructs;
    newTable.GMV          = GMVStructs;
    newTable.WMV          = WMVStructs;
    newTable.Demographics = Demographics;
    newTable.Behavior     = BehaviorStructs;

    % Drop columns that are not used as graph-level behavioural features.
    drop_cols = intersect(newTable.Properties.VariableNames, ...
        {'mu', 'MeanRT', 'RTCV', 'RTSD', 'sigma', 'tau', 'Group'});
    newTable  = removevars(newTable, drop_cols);

end
