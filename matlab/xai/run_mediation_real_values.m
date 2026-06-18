function results = run_mediation_real_values(T, system_rois, n_boot)
% RUN_MEDIATION_REAL_VALUES  Mediation: group -> system activation -> behaviour.
%
%   results = RUN_MEDIATION_REAL_VALUES(T, SYSTEM_ROIS, N_BOOT) tests whether the
%   group difference (HC vs. SZ) in behavioural severity is mediated by the mean
%   first-level activation (effect size) of a set of system ROIs, controlling for
%   age and sex.
%
%   Model:
%       X = group (0 = HC, 1 = SZ)
%       M = mean effect size over SYSTEM_ROIS
%       Y = mean behavioural severity (RTSD, sigma, tau)
%       a: M ~ X + age + sex ;  b, c': Y ~ X + M + age + sex ;  c: Y ~ X + age + sex
%   Indirect effect ab with a bootstrap CI (N_BOOT resamples).
%
%   Inputs:
%       T           - feature table (struct) with Subject, Behavior, EffectSize,
%                     Demographics (as produced by roi_effectsize / the pipeline).
%       system_rois - cell array of ROI field names (e.g. {'ROI_23', ...}).
%       n_boot      - bootstrap resamples (default 5000).
%
%   Output:
%       results - struct with path coefficients, p-values, bootstrap CI, the
%                 indirect effect, partial residuals and a per-subject table.

    if nargin < 3
        n_boot = 5000;
    end

    MIN_SUBJECTS = 0;
    RNG_SEED = 98;   % discovery: 98 (1000 boot); replication: 52 (5000 boot)

    n_subjects = length(T.Subject);

    % --- 1. Group (0 = HC, 1 = SZ) ---------------------------------------
    group = NaN(n_subjects, 1);
    for s = 1:n_subjects
        tok = regexp(T.Subject{s}, 'sub-(\d+)_', 'tokens');
        if isempty(tok); continue; end
        first_digit = str2double(tok{1}{1}(1));
        if ismember(first_digit, [1, 2])
            group(s) = 0;
        elseif ismember(first_digit, [3, 4])
            group(s) = 1;
        end
    end
    fprintf('HC: %d | SZ: %d | Unknown: %d\n', ...
        sum(group == 0), sum(group == 1), sum(isnan(group)));

    % --- 2. Outcome Y, mediator M, covariates ----------------------------
    Y = NaN(n_subjects, 1); M = NaN(n_subjects, 1);
    age = NaN(n_subjects, 1); sex = NaN(n_subjects, 1);
    for s = 1:n_subjects
        try
            beh = T.Behavior{s, 1};
            Y(s) = mean([beh.RTSD, beh.sigma, beh.tau]);
        catch
            Y(s) = NaN;
        end
        try
            eff = T.EffectSize{s, 1};
            roi_vals = NaN(1, length(system_rois));
            for r = 1:length(system_rois)
                roi_vals(r) = eff.(system_rois{r});
            end
            M(s) = mean(roi_vals, 'omitnan');
        catch
            M(s) = NaN;
        end
        try
            dem = T.Demographics{s, 1};
            age(s) = dem.AGE;
            sex(s) = dem.SEX;
        catch
            age(s) = NaN; sex(s) = NaN;
        end
    end
    X = group;

    % --- 3. Drop NaNs ----------------------------------------------------
    valid = ~isnan(X) & ~isnan(Y) & ~isnan(M) & ~isnan(age) & ~isnan(sex);
    valid_idx = find(valid);
    X = X(valid); Y = Y(valid); M = M(valid); age = age(valid); sex = sex(valid);
    n = length(X);
    fprintf('Valid subjects: %d | HC: %d | SZ: %d\n', n, sum(X == 0), sum(X == 1));

    % --- 4. Z-score variables and covariates -----------------------------
    Y = zscore_safe(Y); M = zscore_safe(M);
    age_z = zscore_safe(age); sex_z = zscore_safe(sex);
    C = [age_z, sex_z];

    if n < MIN_SUBJECTS || length(unique(X)) < 2
        warning('Skipped: n=%d < MIN_SUBJECTS=%d or single group', n, MIN_SUBJECTS);
        results.status = 'skipped_low_n_or_one_group';
        results.n_subjects = n;
        return;
    end

    % --- 5. Mediation paths with covariates ------------------------------
    tbl_a = fitlm([X, C], M);
    a = tbl_a.Coefficients.Estimate(2); se_a = tbl_a.Coefficients.SE(2);
    t_a = tbl_a.Coefficients.tStat(2);  p_a = tbl_a.Coefficients.pValue(2);

    tbl_b = fitlm([X, M, C], Y);
    c_p = tbl_b.Coefficients.Estimate(2); se_cp = tbl_b.Coefficients.SE(2);
    t_cp = tbl_b.Coefficients.tStat(2);   p_cp = tbl_b.Coefficients.pValue(2);
    b = tbl_b.Coefficients.Estimate(3);   se_b = tbl_b.Coefficients.SE(3);
    t_b = tbl_b.Coefficients.tStat(3);    p_b = tbl_b.Coefficients.pValue(3);

    tbl_c = fitlm([X, C], Y);
    c = tbl_c.Coefficients.Estimate(2); se_c = tbl_c.Coefficients.SE(2);
    t_c = tbl_c.Coefficients.tStat(2);  p_c = tbl_c.Coefficients.pValue(2);

    ab = a * b;

    % --- 6. Bootstrap CI for the indirect effect -------------------------
    rng(RNG_SEED);
    ab_boot = zeros(n_boot, 1);
    for i = 1:n_boot
        idx = randsample(n, n, true);
        a_b = fitlm([X(idx), C(idx, :)], M(idx)).Coefficients.Estimate(2);
        b_b = fitlm([X(idx), M(idx), C(idx, :)], Y(idx)).Coefficients.Estimate(3);
        ab_boot(i) = a_b * b_b;
    end
    ci_low = prctile(ab_boot, 2.5);
    ci_high = prctile(ab_boot, 97.5);
    p_boot = 2 * min(mean(ab_boot >= 0), mean(ab_boot <= 0));
    sig_indirect = (ci_low > 0) || (ci_high < 0);
    prop_med = ab / c;

    if sign(ab) ~= sign(c)
        warning('Inconsistent mediation: opposite signs for ab and c');
    end

    % --- 7. Print --------------------------------------------------------
    fprintf('\n%-12s %8s %8s %8s %8s %10s %10s\n', 'Path', 'coef', 'se', 't', 'p', 'CI 2.5%', 'CI 97.5%');
    fprintf('%-12s %8.4f %8.4f %8.4f %8.4f %10s %10s\n', 'a (X->M)', a, se_a, t_a, p_a, '-', '-');
    fprintf('%-12s %8.4f %8.4f %8.4f %8.4f %10s %10s\n', 'b (M->Y)', b, se_b, t_b, p_b, '-', '-');
    fprintf('%-12s %8.4f %8.4f %8.4f %8.4f %10s %10s\n', 'Direct', c_p, se_cp, t_cp, p_cp, '-', '-');
    fprintf('%-12s %8.4f %8.4f %8.4f %8.4f %10s %10s\n', 'Total', c, se_c, t_c, p_c, '-', '-');
    fprintf('%-12s %8.4f %8s %8s %8.4f %10.4f %10.4f  %s\n', 'Indirect', ab, '-', '-', ...
        p_boot, ci_low, ci_high, sel(sig_indirect, 'SIGNIFICANT', 'ns'));
    fprintf('Proportion mediated: %.4f (%.1f%%)\n', prop_med, prop_med * 100);

    % --- 8. Output struct ------------------------------------------------
    results.status = 'ok';
    results.n_subjects = n; results.n_HC = sum(X == 0); results.n_SZ = sum(X == 1);
    results.a = a; results.se_a = se_a;
    results.b = b; results.se_b = se_b;
    results.direct = c_p; results.se_direct = se_cp;
    results.total = c; results.se_total = se_c;
    results.indirect = ab;
    results.p_a = p_a; results.p_b = p_b; results.p_direct = p_cp;
    results.p_total = p_c; results.p_indirect = p_boot;
    results.ci_low = ci_low; results.ci_high = ci_high;
    results.sig_indirect = sig_indirect; results.prop_mediated = prop_med;
    results.ab_boot = ab_boot; results.system_rois = system_rois;
    results.M_HC_mean = mean(M(X == 0)); results.M_SZ_mean = mean(M(X == 1));
    results.Y_HC_mean = mean(Y(X == 0)); results.Y_SZ_mean = mean(Y(X == 1));
    results.age_HC_mean_raw = mean(age(X == 0)); results.age_SZ_mean_raw = mean(age(X == 1));
    results.sex_HC_mean_raw = mean(sex(X == 0)); results.sex_SZ_mean_raw = mean(sex(X == 1));

    % --- 9. Per-subject table (z-scored + partial residuals) -------------
    M_partial = fitlm(C, M).Residuals.Raw;   % M with age/sex regressed out
    Y_partial = fitlm(C, Y).Residuals.Raw;   % Y with age/sex regressed out

    results.subject_table = table(T.Subject(valid_idx), X, M, Y, M_partial, Y_partial, ...
        age_z, sex_z, age, sex, ...
        'VariableNames', {'Subject', 'Group', 'Flanker_z', 'BehavSeverity_z', ...
                          'Flanker_partial', 'BehavSeverity_partial', ...
                          'Age_z', 'Sex_z', 'Age_raw', 'Sex_raw'});
    group_label = repmat({'HC'}, n, 1);
    group_label(X == 1) = {'SZ'};
    results.subject_table.GroupLabel = group_label;

    [slope_p, intercept_p] = compute_regression_ci(M_partial, Y_partial, 50);
    results.M_partial = M_partial;
    results.Y_partial = Y_partial;
    results.slope_partial = slope_p;
    results.intercept_partial = intercept_p;
end


% --- Helpers -------------------------------------------------------------
function xz = zscore_safe(x)
    mu = mean(x, 'omitnan');
    sd = std(x, 'omitnan');
    if sd == 0 || isnan(sd)
        xz = x - mu;
    else
        xz = (x - mu) ./ sd;
    end
end


function s = sel(cond, a, b)
    if cond; s = a; else; s = b; end
end


function [slope, intercept, x_range, upper, lower] = compute_regression_ci(x, y, n_pts)
% Ordinary-least-squares regression with a 95% confidence band.
    n = length(x);
    slope = (n * sum(x .* y) - sum(x) * sum(y)) / (n * sum(x.^2) - sum(x)^2);
    intercept = (sum(y) - slope * sum(x)) / n;

    x_range = linspace(min(x), max(x), n_pts)';
    x_mean = mean(x);
    x_ss = sum((x - x_mean).^2);
    y_hat = slope * x_range + intercept;
    mse = sum((y - (slope * x + intercept)).^2) / (n - 2);
    se_fit = sqrt(mse * (1 / n + (x_range - x_mean).^2 / x_ss));
    t_crit = tinv(0.975, n - 2);
    upper = y_hat + t_crit * se_fit;
    lower = y_hat - t_crit * se_fit;
end
