function bold_signal_creation()
% BOLD_SIGNAL_CREATION  Cache preprocessed 4D BOLD volumes as .mat files.
%
%   Reads the preprocessed functional NIfTI for every subject and stores the
%   header (V_bold) and voxel data (Y_bold, single precision) in one .mat file
%   per subject under gat_config.bold_dir. These files are the input for the
%   ALFF, ReHo and connectivity steps.
%
%   The functional images are expected to be named
%       niftiDATA_Subject<NNN>_Condition000.nii
%   inside gat_config.bold_dir, where <NNN> is the subject index.

    cfg = gat_config();
    first_level_dir = cfg.first_level_dir;
    bold_dir        = cfg.bold_dir;
    if ~exist(bold_dir, 'dir'); mkdir(bold_dir); end

    subject_dirs = dir(fullfile(first_level_dir, 'sub-*'));

    for s = 1:length(subject_dirs)
        bold_file = fullfile(bold_dir, ...
            sprintf('niftiDATA_Subject%03d_Condition000.nii', s));
        if ~exist(bold_file, 'file')
            warning('Missing BOLD image: %s', bold_file);
            continue;
        end

        [~, fname] = fileparts(subject_dirs(s).name);
        V_bold = spm_vol(bold_file);
        Y_bold = single(spm_read_vols(V_bold)); %#ok<NASGU>

        save(fullfile(bold_dir, [fname '.mat']), 'V_bold', 'Y_bold', '-v7.3');
        fprintf('Saved subject %d: %s\n', s, fname);
    end
end
