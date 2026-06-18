function spm_2level_gat_wrapper(root_path, radius)
% SPM_2LEVEL_GCN_WRAPPER  Precompute second-level SPM results for every fold.
%
%   SPM_2LEVEL_GCN_WRAPPER(ROOT_PATH, RADIUS) walks the seed/outer-fold folder
%   tree produced by the Python pipeline and, for each outer fold that does not
%   yet have a results.mat, loads the saved train/trainval subject ids and runs
%   spm_2level_gat to generate that fold's ROI definition and feature table.
%
%   Expected layout:
%       ROOT_PATH/seed*/outer*/trainval_ids_*.mat
%   A results.mat is written next to each trainval_ids_*.mat file.
%
%   Inputs:
%       root_path - root of the seed/outer fold tree.
%       radius    - MarsBaR sphere radius in mm (default gat_config sphere_radius).

    cfg = gat_config();
    if nargin < 2 || isempty(radius)
        radius = cfg.sphere_radius;
    end

    seed_dirs = dir(fullfile(root_path, 'seed*'));
    seed_dirs = seed_dirs([seed_dirs.isdir]);

    fprintf('Starting precomputation over %d seeds...\n', length(seed_dirs));

    for s = 1:length(seed_dirs)
        seed_path = fullfile(root_path, seed_dirs(s).name);
        fprintf('\n=== Seed %s (%d/%d) ===\n', seed_dirs(s).name, s, length(seed_dirs));

        outer_dirs = dir(fullfile(seed_path, 'outer*'));
        outer_dirs = outer_dirs([outer_dirs.isdir]);

        for o = 1:length(outer_dirs)
            outer_path = fullfile(seed_path, outer_dirs(o).name);
            fprintf('  Outer fold: %s ... ', outer_dirs(o).name);

            if exist(fullfile(outer_path, 'results.mat'), 'file')
                fprintf('[already done, skipping]\n');
                continue;
            end

            mat_files = dir(fullfile(outer_path, 'trainval_ids_*.mat'));
            if isempty(mat_files)
                fprintf('[no train-ids file found, skipping]\n');
                continue;
            end

            data      = load(fullfile(outer_path, mat_files(1).name));
            var_names = fieldnames(data);
            train_ids = data.(var_names{1});

            try
                spm_2level_gat(train_ids, radius, outer_path);
                fprintf('[completed]\n');
            catch ME
                fprintf('\n  ERROR in %s: %s\n', outer_dirs(o).name, ME.message);
            end
        end
    end

    fprintf('\nAll folds processed.\n');
end
