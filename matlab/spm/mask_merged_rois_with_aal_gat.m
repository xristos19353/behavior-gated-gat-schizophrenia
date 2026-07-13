function flag = mask_merged_rois_with_aal_gat(merged_roi_file, atlas_file, ...
    atlas_labels_file, output_file, output_csv, output_csv_final, ...
    overlap_per, thresholded_activation_file, activation_threshold)
% MASK_MERGED_ROIS_WITH_AAL_GCN  Refine peak ROIs against an anatomical atlas.
%
%   FLAG = MASK_MERGED_ROIS_WITH_AAL_GCN(...) splits each merged peak ROI by the
%   anatomical parcellation it overlaps, keeps only sub-ROIs whose overlap with
%   a parcel exceeds OVERLAP_PER percent (and at least 3 voxels), merges
%   sub-ROIs that fall in the same parcel, and finally masks the result with a
%   thresholded activation map. Two CSV mappings (before / after merging) and a
%   labelled NIfTI of the final ROIs are written.
%
%   Inputs:
%       merged_roi_file             - labelled ROI image from spm_roi_extraction_gat
%       atlas_file                  - anatomical parcellation image
%       atlas_labels_file           - text file with one parcel label per line
%       output_file                 - output labelled ROI NIfTI
%       output_csv                  - ROI-to-parcel mapping before merging
%       output_csv_final            - final ROI-to-parcel mapping
%       overlap_per                 - minimum overlap percentage to keep a sub-ROI
%       thresholded_activation_file - activation map used as a final mask
%       activation_threshold        - threshold applied to the activation map
%
%   Output:
%       flag - 0 on success, 1 if no ROI survived (so the fold is skipped).

    flag = 0;

    roi_nii  = spm_vol(merged_roi_file);
    roi_data = spm_read_vols(roi_nii);

    atlas_nii  = spm_vol(atlas_file);
    atlas_data = spm_read_vols(atlas_nii);
    atlas_labels = read_atlas_labels(atlas_labels_file);

    unique_rois = unique(roi_data);
    unique_rois(unique_rois == 0) = [];

    final_masked_rois = zeros(size(roi_data));
    atlas_roi_mapping = containers.Map('KeyType', 'double', 'ValueType', 'any');
    roi_atlas_mapping = {};
    final_roi_atlas_mapping = {};
    sub_roi_counter = 1000;

    for i = 1:length(unique_rois)
        roi_label   = unique_rois(i);
        current_roi = (roi_data == roi_label);

        overlapping_regions = unique(atlas_data(current_roi));
        overlapping_regions(overlapping_regions == 0) = [];

        if isempty(overlapping_regions)
            fprintf('Warning: ROI %d does not overlap any parcel. Skipping.\n', roi_label);
            roi_atlas_mapping(end+1, 1:4) = {roi_label, 'No overlap', '0%', 0}; 
            continue;
        end

        total_roi_voxels = sum(current_roi(:));

        for j = 1:length(overlapping_regions)
            region_label   = overlapping_regions(j);
            region_mask    = (atlas_data == region_label);
            overlap_voxels = sum(region_mask(:) & current_roi(:));
            overlap_percentage = (overlap_voxels / total_roi_voxels) * 100;

            if overlap_percentage >= overlap_per && overlap_voxels >= 3
                new_roi_label   = sub_roi_counter;
                sub_roi_counter = sub_roi_counter + 1;

                constrained_roi = zeros(size(roi_data));
                constrained_roi(region_mask & current_roi) = new_roi_label;

                if region_label <= length(atlas_labels)
                    region_name = atlas_labels{region_label};
                else
                    region_name = sprintf('Unknown Region %d', region_label);
                end

                roi_atlas_mapping(end+1, 1:4) = ...
                    {roi_label, region_name, sprintf('%.2f%%', overlap_percentage), overlap_voxels}; 

                if isKey(atlas_roi_mapping, region_label)
                    atlas_roi_mapping(region_label) = atlas_roi_mapping(region_label) + constrained_roi;
                else
                    atlas_roi_mapping(region_label) = constrained_roi;
                end

                final_roi_atlas_mapping(end+1, 1:5) = ...
                    {roi_label, new_roi_label, region_label, region_name, overlap_voxels}; 
            end
        end
    end

    if isempty(final_roi_atlas_mapping)
        flag = 1;
        return;
    end

    % Merge all sub-ROIs that fall in the same parcel.
    atlas_keys = keys(atlas_roi_mapping);
    for k = 1:length(atlas_keys)
        region_label = atlas_keys{k};
        merged_roi   = atlas_roi_mapping(region_label);
        final_masked_rois(merged_roi > 0) = region_label;
    end

    % Mask with the thresholded activation map.
    activation_nii  = spm_vol(thresholded_activation_file);
    activation_data = spm_read_vols(activation_nii);
    activation_mask = activation_data > activation_threshold;
    final_masked_rois = final_masked_rois .* activation_mask;

    valid_rois = unique(final_masked_rois);
    valid_rois(valid_rois == 0) = [];
    filtered_mapping = final_roi_atlas_mapping( ...
        ismember(cell2mat(final_roi_atlas_mapping(:, 3)), valid_rois), :);

    roi_nii.fname = output_file;
    spm_write_vol(roi_nii, final_masked_rois);
    fprintf('Masked and thresholded ROIs saved as: %s\n', output_file);

    writecell([{'ROI', 'Parcel', 'Overlap Percentage', 'Voxel Count'}; roi_atlas_mapping], ...
              output_csv);
    writecell([{'OriginalROI', 'NewROIAfterSplitting', 'FinalROIAfterMerging', ...
                'Parcel', 'VoxelCount'}; filtered_mapping], output_csv_final);

    final_roi_data            = struct();
    final_roi_data.masked_rois = final_masked_rois;
    final_roi_data.mapping     = filtered_mapping;
    final_roi_data.atlas_labels = atlas_labels;
    save(strrep(output_file, '.nii', '.mat'), 'final_roi_data');

    if size(filtered_mapping, 1) == 0
        flag = 1;
    end
end


function atlas_labels = read_atlas_labels(atlas_labels_file)
% Read parcellation labels (one per line) from a text file.
    fid = fopen(atlas_labels_file, 'r');
    if fid == -1
        error('Could not open atlas labels file: %s', atlas_labels_file);
    end
    atlas_labels = {};
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line)
            atlas_labels{end+1} = strtrim(line); 
        end
    end
    fclose(fid);
end
