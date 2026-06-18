function gmv_wmv_calc()
% GMV_WMV_CALC  Modulate and reslice grey/white-matter tissue maps.
%
%   For every subject this:
%     1. Computes the Jacobian of the SPM deformation field (y_*.nii).
%     2. Modulates the native-space tissue segments (wc1 -> mwc1, wc2 -> mwc2).
%     3. Reslices the modulated and unmodulated maps to a common reference space.
%     4. Writes Resliced_GMV_*.nii / Resliced_WMV_*.nii into each subject folder
%        (consumed by roi_effectsize).
%
%   Expects, per subject, the SPM segmentation outputs wc1c*, wc2c* and y_c* in
%   an "anat" subfolder of gat_config.participants_data.

    cfg          = gat_config();
    base_dir     = cfg.participants_data;
    ref_image    = cfg.reslice_reference;   % target space for reslicing
    subject_dirs = dir(fullfile(base_dir, 'sub-*'));
    subject_dirs = subject_dirs([subject_dirs.isdir]);

    fprintf('Found %d subjects\n', length(subject_dirs));

    if ~exist(ref_image, 'file')
        error('Missing reslice reference image: %s', ref_image);
    end

    for s = 1:length(subject_dirs)
        subject_id = subject_dirs(s).name;
        anat_dir   = fullfile(base_dir, subject_id, 'anat');
        subj_dir   = fullfile(base_dir, subject_id);

        wc1_file = fullfile(anat_dir, ['wc1c' subject_id '_T1w.nii']);
        wc2_file = fullfile(anat_dir, ['wc2c' subject_id '_T1w.nii']);
        def_file = fullfile(anat_dir, ['y_c'  subject_id '_T1w.nii']);

        if ~exist(wc1_file, 'file') || ~exist(wc2_file, 'file') || ~exist(def_file, 'file')
            fprintf('SKIP (missing wc1/wc2/y_): %s\n', subject_id);
            continue
        end

        % --- Jacobian of the deformation field ---------------------------
        Nii  = nifti(def_file);
        Ydef = squeeze(Nii.dat(:, :, :, 1, :));
        vx   = sqrt(sum(Nii.mat(1:3, 1:3).^2));

        [du_dx, du_dy, du_dz] = gradient(Ydef(:, :, :, 1), vx(1), vx(2), vx(3));
        [dv_dx, dv_dy, dv_dz] = gradient(Ydef(:, :, :, 2), vx(1), vx(2), vx(3));
        [dw_dx, dw_dy, dw_dz] = gradient(Ydef(:, :, :, 3), vx(1), vx(2), vx(3));

        Yjac = du_dx .* (dv_dy .* dw_dz - dv_dz .* dw_dy) ...
             - du_dy .* (dv_dx .* dw_dz - dv_dz .* dw_dx) ...
             + du_dz .* (dv_dx .* dw_dy - dv_dy .* dw_dx);

        V_wc1    = spm_vol(wc1_file);
        Yjac_rs  = imresize3(Yjac, V_wc1.dim, 'linear');
        fprintf('  Jacobian range: [%.3f, %.3f]\n', min(Yjac_rs(:)), max(Yjac_rs(:)));

        % --- Modulate wc1 -> mwc1 and wc2 -> mwc2 ------------------------
        mwc1_file = fullfile(anat_dir, ['mwc1' subject_id '_T1w.nii']);
        mwc2_file = fullfile(anat_dir, ['mwc2' subject_id '_T1w.nii']);

        for tissue = {{'wc1c', 'mwc1'}, {'wc2c', 'mwc2'}}
            in_file  = fullfile(anat_dir, [tissue{1}{1} subject_id '_T1w.nii']);
            out_file = fullfile(anat_dir, [tissue{1}{2} subject_id '_T1w.nii']);

            V              = spm_vol(in_file);
            Y_mod          = spm_read_vols(V) .* Yjac_rs;
            Y_mod(Y_mod < 0) = 0;

            V_out       = V;
            V_out.fname = out_file;
            V_out.dt    = [spm_type('float32') 0];
            spm_write_vol(V_out, Y_mod);
        end

        % --- Reslice everything to the reference space -------------------
        flags = struct('interp', 1, 'wrap', [0 0 0], 'mask', 0, 'which', 1, 'mean', 0);
        spm_reslice(char(ref_image, mwc1_file), flags);
        spm_reslice(char(ref_image, mwc2_file), flags);
        spm_reslice(char(ref_image, wc1_file), flags);
        spm_reslice(char(ref_image, wc2_file), flags);

        % --- Rename resliced outputs into the subject folder -------------
        move_resliced(anat_dir, mwc1_file, subj_dir, ['Resliced_GMV_mwc1_' subject_id '.nii']);
        move_resliced(anat_dir, mwc2_file, subj_dir, ['Resliced_WMV_mwc2_' subject_id '.nii']);
        move_resliced(anat_dir, wc1_file,  subj_dir, ['Resliced_GMV_wc1_'  subject_id '.nii']);
        move_resliced(anat_dir, wc2_file,  subj_dir, ['Resliced_WMV_wc2_'  subject_id '.nii']);

        % --- Sanity check ------------------------------------------------
        gmv = sum(spm_read_vols(spm_vol(mwc1_file)), 'all', 'omitnan') / 1000;
        wmv = sum(spm_read_vols(spm_vol(mwc2_file)), 'all', 'omitnan') / 1000;
        fprintf('%s - GMV: %.1f mL | WMV: %.1f mL\n', subject_id, gmv, wmv);
    end

    disp('=== All done ===');
end


function move_resliced(anat_dir, source_file, dest_dir, dest_name)
% Move an spm_reslice output ('r' prefix) to dest_dir/dest_name.
    [~, name, ext] = fileparts(source_file);
    auto_file = fullfile(anat_dir, ['r' name ext]);
    if exist(auto_file, 'file')
        movefile(auto_file, fullfile(dest_dir, dest_name));
    end
end
