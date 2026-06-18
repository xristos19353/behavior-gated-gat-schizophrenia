function alff_calculation()
% ALFF_CALCULATION  Compute voxel-wise z-scored ALFF for every subject.
%
%   For each cached BOLD .mat file (see bold_signal_creation), this computes the
%   Amplitude of Low-Frequency Fluctuations (ALFF) in the 0.01-0.08 Hz band,
%   z-scores it within the brain mask, and appends ALFF_voxel_z to the file.
%
%   Assumes a repetition time (TR) of 2 s; change Fs below for other TRs.

    cfg      = gat_config();
    base_dir = cfg.bold_dir;
    Fs       = 1 / 2;      % sampling frequency (Hz); TR = 2 s
    f_min    = 0.01;
    f_max    = 0.08;

    mat_files = dir(fullfile(base_dir, '*.mat'));
    fprintf('Found %d .mat files\n', numel(mat_files));

    for k = 1:numel(mat_files)
        file_path = fullfile(base_dir, mat_files(k).name);
        fprintf('\nProcessing %s\n', mat_files(k).name);

        try
            S = load(file_path);
            if ~isfield(S, 'Y_bold') || ~isfield(S, 'V_bold')
                fprintf('SKIP: missing Y_bold or V_bold\n');
                continue;
            end

            Y_bold = S.Y_bold;
            V_bold = S.V_bold; %#ok<NASGU>
            T_len  = size(Y_bold, 4);

            % Reshape to voxels x time and replace NaNs with the voxel mean.
            Y_reshaped = reshape(Y_bold, [], T_len);
            nan_idx    = isnan(Y_reshaped);
            Y_reshaped(nan_idx) = 0;

            valid_counts = sum(~nan_idx, 2);
            voxel_means  = sum(Y_reshaped, 2) ./ valid_counts;
            voxel_means(valid_counts == 0) = 0;
            voxel_mean_matrix = repmat(voxel_means, 1, T_len);
            Y_reshaped(nan_idx) = voxel_mean_matrix(nan_idx);

            % Remove the per-voxel temporal mean.
            Y_detrended = Y_reshaped - mean(Y_reshaped, 2);

            NFFT = 2^nextpow2(T_len);
            f    = Fs / 2 * linspace(0, 1, NFFT / 2 + 1);
            freq_idx = (f >= f_min) & (f <= f_max);

            Y_fft      = fft(Y_detrended, NFFT, 2);
            P2         = abs(Y_fft(:, 1:NFFT/2 + 1)).^2 / NFFT;
            ALFF_voxel = sqrt(sum(P2(:, freq_idx), 2));

            % Z-score within the brain mask.
            brain_mask = ~isnan(Y_reshaped(:, 1)) & (mean(Y_reshaped, 2) ~= 0);
            ALFF_brain = ALFF_voxel(brain_mask);
            ALFF_mean  = mean(ALFF_brain);
            ALFF_std   = std(ALFF_brain);

            if ALFF_std == 0 || isnan(ALFF_std)
                fprintf('SKIP: ALFF std is zero/NaN\n');
                continue;
            end

            ALFF_voxel_z = (ALFF_voxel - ALFF_mean) / ALFF_std; %#ok<NASGU>
            save(file_path, 'Y_bold', 'V_bold', 'ALFF_voxel_z', '-v7.3');
            fprintf('DONE: %s\n', mat_files(k).name);

        catch ME
            fprintf('ERROR in %s: %s\n', mat_files(k).name, ME.message);
        end
    end

    fprintf('\nAll done.\n');
end
