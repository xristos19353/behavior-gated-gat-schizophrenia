function A_sparse = sparsify_adjacency(A, binary)
% SPARSIFY_ADJACENCY  Sparsify a symmetric matrix using Erdos-Renyi theory.
%
%   A_sparse = SPARSIFY_ADJACENCY(A, BINARY) keeps only the strongest edges of
%   the symmetric matrix A. The number of retained edges is derived from the
%   Erdos-Renyi connectivity threshold p_min = log(N) / N, which guarantees a
%   (near) connected graph on average.
%
%   Inputs:
%       A      - symmetric matrix (e.g. correlation / Fisher-z connectivity)
%       binary - (optional, default true) if true the output is binary (0/1);
%                if false the original edge weights are retained.
%
%   Output:
%       A_sparse - sparsified, symmetric adjacency matrix.

    if nargin < 2
        binary = true;
    end

    % Force symmetry and zero the diagonal.
    A = (A + A') / 2;
    A(1:size(A, 1) + 1:end) = 0;

    N = size(A, 1);

    % Minimum edge density from Erdos-Renyi theory (factor 1; use 2 for a
    % stricter connectivity guarantee).
    p_min = 1 * log(N) / N;

    num_possible_edges = N * (N - 1) / 2;
    num_edges_to_keep  = round(p_min * num_possible_edges);

    % Work on the upper triangle only (undirected graph).
    upper_idx    = find(triu(ones(N), 1));
    edge_weights = A(upper_idx);

    % Keep the strongest edges by absolute weight.
    [~, sorted_idx] = sort(abs(edge_weights), 'descend');
    selected_idx    = upper_idx(sorted_idx(1:num_edges_to_keep));

    % Nodes that have at least one valid (non-zero, non-NaN) connection.
    valid_nodes = any(A ~= 0 & ~isnan(A), 2);
    valid_idx   = find(valid_nodes);

    A_sparse = zeros(N);

    if binary
        A_sparse(selected_idx) = 1;
        % Drop rows/columns for nodes that were entirely empty.
        invalid_idx = setdiff(1:N, valid_idx);
        A_sparse(invalid_idx, :) = 0;
        A_sparse(:, invalid_idx) = 0;
    else
        A_sparse(selected_idx) = A(selected_idx);
    end

    % Restore symmetry.
    A_sparse = A_sparse + A_sparse';

end
