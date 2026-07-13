function reho_calculation()
% REHO_CALCULATION  Compute voxel-wise z-scored ReHo for every subject.
%
%   For each cached BOLD .mat file (see bold_signal_creation / alff_calculation),
%   this computes Regional Homogeneity (ReHo) as Kendall's coefficient of
%   concordance over each voxel's 3x3x3 neighbourhood (27 voxels), z-scores it
%   within the brain mask, and appends reho_z to the file.

    cfg         = gat_config();
    folder_path = cfg.bold_dir;
    mat_files   = dir(fullfile(folder_path, '*.mat'));

    for i = 1:length(mat_files)
        file_path = fullfile(folder_path, mat_files(i).name);
        fprintf('Processing file: %s\n', mat_files(i).name);

        vars   = load(file_path, 'Y_bold', 'V_bold', 'ALFF_voxel_z');
        Y_bold = vars.Y_bold;
        T_len  = size(Y_bold, 4);

        % Replace NaNs in the time series with the voxel-wise mean.
        Y_reshaped = reshape(Y_bold, [], T_len);
        nan_idx    = isnan(Y_reshaped);
        Y_reshaped(nan_idx) = 0;
        valid_counts = sum(~nan_idx, 2);
        voxel_means  = sum(Y_reshaped, 2) ./ valid_counts;
        voxel_means(valid_counts == 0) = 0;
        voxel_mean_matrix = repmat(voxel_means, 1, T_len);
        Y_reshaped(nan_idx) = voxel_mean_matrix(nan_idx);
        Y_clean = reshape(Y_reshaped, size(Y_bold));

        reho_map = nan(size(Y_clean, 1), size(Y_clean, 2), size(Y_clean, 3));

        for x = 2:size(Y_clean, 1) - 1
            for y = 2:size(Y_clean, 2) - 1
                for z = 2:size(Y_clean, 3) - 1
                    block = Y_clean(x-1:x+1, y-1:y+1, z-1:z+1, :);
                    block = reshape(block, [], T_len);

                    if sum(all(block == 0, 2)) > 13
                        reho_map(x, y, z) = 0;
                        continue;
                    end
                    valid_voxels = block(~all(block == 0, 2), :);
                    if size(valid_voxels, 1) < 5
                        reho_map(x, y, z) = 0;
                        continue;
                    end

                    % Kendall's W over the neighbourhood time series.
                    ranks = tiedrank(valid_voxels')';
                    Sdev  = sum((sum(ranks, 1) - size(ranks, 1) * (size(ranks, 1) + 1) / 2).^2);
                    K     = size(ranks, 1);
                    reho_map(x, y, z) = 12 * Sdev / (K^2 * (T_len^3 - T_len));
                end
            end
        end

        reho_map(isnan(reho_map)) = 0;

        % Z-score within the brain mask.
        brain_mask = ~isnan(reho_map) & (mean(Y_clean, 4) ~= 0);
        reho_vals  = reho_map(brain_mask);
        mean_reho  = mean(reho_vals, 'omitnan');
        std_reho   = std(reho_vals, 'omitnan');

        reho_z = nan(size(reho_map)); 
        if std_reho == 0
            reho_z = zeros(size(reho_map));
        else
            reho_z = nan(size(reho_map));
            reho_z(brain_mask) = (reho_vals - mean_reho) / std_reho;
        end

        save(file_path, 'reho_z', '-append', '-v7.3');
        fprintf('Saved: %s\n\n', mat_files(i).name);
    end
end
