function [peakTable, T] = spm_2level_gat(train_subject_ids, sphere_radius, output_dir)
% SPM_2LEVEL_GCN  Second-level SPM analysis and graph-feature extraction.
%
%   [peakTable, T] = SPM_2LEVEL_GCN(TRAIN_SUBJECT_IDS, SPHERE_RADIUS, OUTPUT_DIR)
%   runs the second-level (group) analysis that defines the ROIs used to build
%   the subject graphs, then extracts every node/edge feature consumed by the
%   GNN, and saves the result as results.mat.
%
%   Steps:
%     1. Two-sample t-test (schizophrenia vs. non-schizophrenia) on the
%        first-level contrast images of the TRAINING subjects only, with age,
%        sex and education as covariates.
%     2. Threshold the contrast (p < 0.001 uncorrected, k = 20 voxels) and
%        detect cluster peaks (up to 3 per cluster, >= 8 mm apart).
%     3. Build spherical ROIs around the peaks and refine them against an
%        anatomical atlas (see spm_roi_extraction_gat / mask_merged_rois_with_aal_gat).
%     4. Extract subject-specific peak ROIs (process_subject_rois_with_marsbar_gat),
%        compute connectivity + node metrics (adjacency_matrix_calc) and effect
%        sizes / tissue density / behaviour (roi_effectsize).
%     5. Save flag, peakTable and the feature table T to results.mat.
%
%   Inputs:
%       train_subject_ids - cell/string array of training subject ids. Only these
%                           subjects contribute to the second-level t-test, which
%                           keeps the ROI definition free of test-set leakage.
%       sphere_radius     - MarsBaR sphere radius in mm (default cfg.sphere_radius).
%       output_dir        - directory where results.mat is written
%                           (default cfg.two_sample_dir).
%
%   Notes:
%       For the final paper runs the ROIs were precomputed once on a consensus
%       atlas (see config gat_config.atlas_roi_file) and the Python side loaded
%       results.mat directly (USE_PRECOMPUTED_SPM = True). Set the atlas path in
%       gat_config to reproduce that mode; run this function per fold to
%       reproduce the leakage-free per-fold ROI definition.

    cfg = gat_config();
    if nargin < 2 || isempty(sphere_radius)
        sphere_radius = cfg.sphere_radius;
    end
    if nargin < 3 || isempty(output_dir)
        output_dir = cfg.two_sample_dir;
    end

    current_folder = pwd;
    spm('Defaults', 'fMRI');
    spm_jobman('initcfg');
    spm_get_defaults('cmdline', true);

    load(cfg.demographics_mat);   % provides 'demo' (ID, SEX, EDU, AGE)

    first_level_dir = cfg.first_level_dir;
    if ~exist(output_dir, 'dir'); mkdir(output_dir); end

    % ===================== 1. Build the two groups =======================
    train_subject_ids = string(train_subject_ids);
    subjects      = dir(fullfile(first_level_dir, 'sub-*'));
    subject_names = string({subjects.name});
    keep          = ismember(subject_names, train_subject_ids);
    subjects      = subjects(keep);

    group1_scans = {};   % schizophrenia (SZ)
    group2_scans = {};   % non-schizophrenia (non-SZ)
    contrast_name = cfg.contrast_name;

    for i = 1:length(subjects)
        subject_id   = subjects(i).name;
        spm_mat_path = fullfile(first_level_dir, subject_id, 'SPM.mat');
        if ~exist(spm_mat_path, 'file'); continue; end

        load(spm_mat_path, 'SPM');
        cidx = find(strcmp({SPM.xCon.name}, contrast_name), 1);
        if isempty(cidx); continue; end

        contrast_img = fullfile(first_level_dir, subject_id, sprintf('con_%04d.nii', cidx));
        if ~exist(contrast_img, 'file'); continue; end

        % SZ subjects have ids beginning with sub-3 / sub-4.
        subject_numeric = regexprep(subject_id, '_[A-Za-z]+$', '');
        is_sz = startsWith(subject_numeric, 'sub-3') || startsWith(subject_numeric, 'sub-4');
        if is_sz
            group1_scans{end+1} = contrast_img; %#ok<AGROW>
        else
            group2_scans{end+1} = contrast_img; %#ok<AGROW>
        end
    end

    % --- Covariates (sex, education, age) in scan order ------------------
    all_scans = [group1_scans, group2_scans];
    SEX = []; EDU = []; AGE = [];
    for i = 1:length(all_scans)
        sub_id     = regexp(all_scans{i}, 'sub-\d+', 'match', 'once');
        numeric_id = str2double(regexp(sub_id, '\d+', 'match', 'once'));
        idx        = find(demo.ID == numeric_id);
        if isempty(idx)
            error('Subject %s (numeric ID %d) not found in demo.ID', sub_id, numeric_id);
        end
        SEX(end+1, 1) = demo.SEX(idx); %#ok<AGROW>
        EDU(end+1, 1) = demo.EDU(idx); %#ok<AGROW>
        AGE(end+1, 1) = demo.AGE(idx); %#ok<AGROW>
    end

    % ===================== 2. Two-sample t-test ==========================
    if exist(output_dir, 'dir'); rmdir(output_dir, 's'); end
    mkdir(output_dir);

    matlabbatch = {};
    matlabbatch{1}.spm.stats.factorial_design.dir            = {output_dir};
    matlabbatch{1}.spm.stats.factorial_design.des.t2.scans1  = group1_scans(:);
    matlabbatch{1}.spm.stats.factorial_design.des.t2.scans2  = group2_scans(:);
    matlabbatch{1}.spm.stats.factorial_design.des.t2.variance = 1;
    matlabbatch{1}.spm.stats.factorial_design.des.t2.gmsca    = 0;
    matlabbatch{1}.spm.stats.factorial_design.des.t2.ancova   = 0;

    covs = {SEX, 'Sex'; EDU, 'Education'; AGE, 'Age'};
    for c = 1:size(covs, 1)
        matlabbatch{1}.spm.stats.factorial_design.cov(c).c     = covs{c, 1};
        matlabbatch{1}.spm.stats.factorial_design.cov(c).cname = covs{c, 2};
        matlabbatch{1}.spm.stats.factorial_design.cov(c).iCFI  = 1;
        matlabbatch{1}.spm.stats.factorial_design.cov(c).iCC   = 1;
    end

    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im         = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.em         = {''};
    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit     = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm    = 1;

    matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(output_dir, 'SPM.mat')};
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

    matlabbatch{3}.spm.stats.con.spmmat = {fullfile(output_dir, 'SPM.mat')};
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.name    = 'Non-SZ > SZ';
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [-1 1];
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

    spm_jobman('run', matlabbatch);

    % ===================== 3. Threshold and find peaks ===================
    cd(output_dir);
    load('SPM.mat');

    Ic        = 1;        % contrast index
    p_unc     = 0.001;    % uncorrected p threshold
    k         = 20;       % cluster extent threshold (voxels)
    min_dist_mm = 8;      % minimum distance between peaks
    max_peaks_per_cluster = 3;

    STAT = SPM.xCon(Ic).STAT;
    df   = [SPM.xCon(Ic).eidf, SPM.xX.erdf];
    u    = spm_u(p_unc, df, STAT);

    V   = spm_vol(fullfile(SPM.swd, SPM.xCon(Ic).Vspm.fname));
    XYZ = SPM.xVol.XYZ;
    Z   = spm_get_data(V, XYZ);

    I = find(Z > u);
    if isempty(I)
        [peakTable, T] = save_empty(current_folder, output_dir);
        return;
    end

    % Drop clusters smaller than k voxels.
    c    = spm_clusters(XYZ(:, I));
    clus = unique(c);
    for i = 1:length(clus)
        if sum(c == clus(i)) < k
            I(c == clus(i)) = [];
            c(c == clus(i)) = [];
        end
    end
    if isempty(I)
        [peakTable, T] = save_empty(current_folder, output_dir);
        return;
    end

    XYZ_thr = XYZ(:, I);
    Z_thr   = Z(I);
    [~, Zmax, M, A, ~] = spm_max(Z_thr, XYZ_thr);

    peakList = [];
    for i = 1:max(A)
        vox_idx = find(A == i);
        if isempty(vox_idx); continue; end

        [~, sort_idx] = sort(Zmax(vox_idx), 'descend');
        coords_mm = [];
        for j = 1:length(sort_idx)
            this_vox = vox_idx(sort_idx(j));
            coord_mm = SPM.xVol.M(1:3, :) * [M(:, this_vox); 1];

            if isempty(coords_mm)
                is_far = true;
            else
                dists  = sqrt(sum((coords_mm - coord_mm).^2, 1));
                is_far = all(dists >= min_dist_mm);
            end

            if is_far
                coords_mm(:, end+1) = coord_mm; %#ok<AGROW>
                peakList = [peakList; coord_mm', Zmax(this_vox)]; %#ok<AGROW>
            end
            if size(coords_mm, 2) == max_peaks_per_cluster; break; end
        end
    end

    peakTable = array2table(peakList, 'VariableNames', {'X', 'Y', 'Z', 'Stat'});
    if size(peakTable, 1) <= 1
        warning('One or zero peaks found. Skipping fold.');
        [peakTable, T] = save_empty(current_folder, output_dir);
        return;
    end

    % ===================== 4. ROI definition and features ================
    spm_roi_extraction_gat(peakTable, sphere_radius);

    flag = mask_merged_rois_with_aal_gat( ...
        fullfile(cfg.group_rois_dir, 'merged_roi.nii'), ...
        cfg.parcellation_atlas, ...
        cfg.parcellation_labels, ...
        fullfile(cfg.group_rois_dir, 'merged_rois_masked_atlas.nii'), ...
        fullfile(cfg.group_rois_dir, 'ROIS.csv'), ...
        fullfile(cfg.group_rois_dir, 'final_ROIS.csv'), ...
        cfg.overlap_percent, ...
        cfg.thresholded_activation, ...
        cfg.activation_threshold);
    if flag ~= 0
        warning('ROI/atlas masking produced no ROIs. Skipping fold.');
        [peakTable, T] = save_empty(current_folder, output_dir);
        return;
    end

    process_subject_rois_with_marsbar_gat( ...
        fullfile(cfg.group_rois_dir, 'merged_rois_masked_atlas.nii'));

    roi_table      = readtable(fullfile(cfg.group_rois_dir, 'final_ROIS.csv'));
    roi_labels_all = roi_table.FinalROIAfterMerging;

    Zmetrics = adjacency_matrix_calc(roi_labels_all);
    T        = roi_effectsize(roi_labels_all);

    T.Z         = Zmetrics.ZInfo;
    T.Zbinary   = Zmetrics.Zbinary;
    T.Zweighted = Zmetrics.Zweighted;
    T.ALFF      = Zmetrics.ALFFInfo;
    T.ReHo      = Zmetrics.ReHoInfo;
    T.DC        = Zmetrics.DCInfo;

    T          = table2struct(T, 'ToScalar', true);
    T.Subject  = cellstr(T.Subject);

    flag = 0; %#ok<NASGU>
    cd(current_folder);
    save(fullfile(output_dir, 'results.mat'), 'flag', 'peakTable', 'T');
end


function [peakTable, T] = save_empty(current_folder, output_dir)
% Write an empty results file with flag = 1 so the Python side skips the fold.
    flag = 1; peakTable = []; T = []; %#ok<NASGU>
    cd(current_folder);
    save(fullfile(output_dir, 'results.mat'), 'flag', 'peakTable', 'T');
end
