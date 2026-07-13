function results = run_mediation_real_values(T, system_rois, n_boot)
% RUN_MEDIATION  Mediation of the diagnostic-group difference in the ISV
% composite through system-level task activation, adjusting for age and sex.
%
%   X = diagnostic group (HC=0, SZ=1)
%   M = mean Flanker effect size across system_rois
%   Y = ISV composite = mean of z-scored RTSD, sigma, tau
%
% Indirect-effect 95% CI and p-value use bias-corrected (BC) bootstrap;
% the BC interval excludes 0 iff p_boot < 0.05.
%
%   T           table with fields Subject, Behavior, EffectSize, Demographics
%   system_rois cell array of ROI field names in T.EffectSize
%   n_boot      bootstrap resamples (default 5000)

if nargin < 3 || isempty(n_boot), n_boot = 5000; end

nS = numel(T.Subject);

% Diagnostic group from subject ID prefix (3xxx = HC, 4xxx = SZ)
group = NaN(nS,1);
for s = 1:nS
    tok = regexp(T.Subject{s}, 'sub-(\d+)_', 'tokens');
    if isempty(tok), continue; end
    d1 = str2double(tok{1}{1}(1));
    if d1 == 3, group(s) = 0; elseif d1 == 4, group(s) = 1; end
end

% Extract behavior, activation, covariates
RTSD = NaN(nS,1); sig = NaN(nS,1); tau = NaN(nS,1);
M = NaN(nS,1); age = NaN(nS,1); sex = NaN(nS,1);
for s = 1:nS
    try
        beh = T.Behavior{s,1};
        RTSD(s) = beh.RTSD; sig(s) = beh.sigma; tau(s) = beh.tau;
    catch
    end
    try
        eff = T.EffectSize{s,1};
        roi_vals = NaN(1, numel(system_rois));
        for r = 1:numel(system_rois)
            roi_vals(r) = eff.(system_rois{r});
        end
        M(s) = mean(roi_vals, 'omitnan');
    catch
    end
    try
        dem = T.Demographics{s,1};
        age(s) = dem.AGE; sex(s) = dem.SEX;
    catch
    end
end
X = group;

% Complete cases
valid = ~isnan(X) & ~isnan(RTSD) & ~isnan(sig) & ~isnan(tau) ...
      & ~isnan(M) & ~isnan(age) & ~isnan(sex);
valid_idx = find(valid);
X = X(valid); RTSD = RTSD(valid); sig = sig(valid); tau = tau(valid);
M = M(valid); age = age(valid); sex = sex(valid);
n = numel(X);

if n < 2 || numel(unique(X)) < 2
    results.status = 'skipped_low_n_or_one_group';
    results.n_subjects = n;
    return;
end

% Standardize (ISV composite = mean of z-scored metrics, then z-scored)
Y = zscore_safe(mean([zscore_safe(RTSD), zscore_safe(sig), zscore_safe(tau)], 2));
M = zscore_safe(M);
C = [zscore_safe(age), zscore_safe(sex)];

% Mediation paths (covariate-adjusted)
tbl_a = fitlm([X, C], M);          % a: X -> M
a = tbl_a.Coefficients.Estimate(2); se_a = tbl_a.Coefficients.SE(2); p_a = tbl_a.Coefficients.pValue(2);
tbl_b = fitlm([X, M, C], Y);       % b: M -> Y ; c': direct
c_p = tbl_b.Coefficients.Estimate(2); se_cp = tbl_b.Coefficients.SE(2); p_cp = tbl_b.Coefficients.pValue(2);
b = tbl_b.Coefficients.Estimate(3); se_b = tbl_b.Coefficients.SE(3); p_b = tbl_b.Coefficients.pValue(3);
tbl_c = fitlm([X, C], Y);          % c: total
c = tbl_c.Coefficients.Estimate(2); se_c = tbl_c.Coefficients.SE(2); p_c = tbl_c.Coefficients.pValue(2);
ab = a * b;                        % indirect

% Bias-corrected bootstrap of the indirect effect
rng(42);
ab_boot = zeros(n_boot,1);
for i = 1:n_boot
    idx = randsample(n, n, true);
    Xb = X(idx); Yb = Y(idx); Mb = M(idx); Cb = C(idx,:);
    a_b = subsref(fitlm([Xb, Cb], Mb).Coefficients.Estimate, struct('type','()','subs',{{2}}));
    b_b = subsref(fitlm([Xb, Mb, Cb], Yb).Coefficients.Estimate, struct('type','()','subs',{{3}}));
    ab_boot(i) = a_b * b_b;
end

alpha = 0.05;
lo = 1/(2*n_boot); hi = 1 - lo;

z0 = norminv(min(max(mean(ab_boot < ab), lo), hi));            % bias correction
ci_low  = prctile(ab_boot, 100*normcdf(2*z0 + norminv(alpha/2)));
ci_high = prctile(ab_boot, 100*normcdf(2*z0 + norminv(1-alpha/2)));

G0 = min(max(mean(ab_boot < 0), lo), hi);                      % BC p-value
p_one  = normcdf(norminv(G0) - 2*z0);
p_boot = 2 * min(p_one, 1 - p_one);
sig_indirect = (ci_low > 0) || (ci_high < 0);

prop_med = ab / c;
if sign(ab) ~= sign(c)
    warning('Inconsistent mediation: opposite signs for indirect and total effect.');
end

% Console summary
fprintf('HC: %d | SZ: %d | n: %d\n', sum(X==0), sum(X==1), n);
fprintf('%-10s %8s %8s %8s %10s %10s\n', 'Path','coef','se','p','CI2.5%','CI97.5%');
fprintf('%-10s %8.4f %8.4f %8.4f %10s %10s\n', 'a (X->M)', a, se_a, p_a, '-','-');
fprintf('%-10s %8.4f %8.4f %8.4f %10s %10s\n', 'b (M->Y)', b, se_b, p_b, '-','-');
fprintf('%-10s %8.4f %8.4f %8.4f %10s %10s\n', 'Direct',  c_p, se_cp, p_cp, '-','-');
fprintf('%-10s %8.4f %8.4f %8.4f %10s %10s\n', 'Total',   c, se_c, p_c, '-','-');
fprintf('%-10s %8.4f %8s %8.4f %10.4f %10.4f  %s\n', 'Indirect', ab, '-', p_boot, ci_low, ci_high, ...
    sel(sig_indirect,'SIGNIFICANT','ns'));
fprintf('Proportion mediated: %.1f%%\n', 100*prop_med);

% Output
results.status = 'ok';
results.n_subjects = n; results.n_HC = sum(X==0); results.n_SZ = sum(X==1);
results.a = a; results.se_a = se_a; results.p_a = p_a;
results.b = b; results.se_b = se_b; results.p_b = p_b;
results.direct = c_p; results.se_direct = se_cp; results.p_direct = p_cp;
results.total = c; results.se_total = se_c; results.p_total = p_c;
results.indirect = ab; results.p_indirect = p_boot;
results.ci_low = ci_low; results.ci_high = ci_high;
results.sig_indirect = sig_indirect; results.prop_mediated = prop_med;
results.z0 = z0; results.ab_boot = ab_boot; results.system_rois = system_rois;

% Partial residuals (age/sex removed) for plotting
M_partial = fitlm(C, M).Residuals.Raw;
Y_partial = fitlm(C, Y).Residuals.Raw;
results.M_partial = M_partial;
results.Y_partial = Y_partial;
results.subject_table = table(T.Subject(valid_idx), X, M, Y, M_partial, Y_partial, ...
    'VariableNames', {'Subject','Group','Flanker_z','ISV_z','Flanker_partial','ISV_partial'});

end


function xz = zscore_safe(x)
mu = mean(x, 'omitnan'); sd = std(x, 'omitnan');
if sd == 0 || isnan(sd), xz = x - mu; else, xz = (x - mu) ./ sd; end
end

function s = sel(cond, a, b)
if cond, s = a; else, s = b; end
end