# -*- coding: utf-8 -*-
"""Nature-style polish pass: tighten prose, sharpen contribution, reduce length."""
import os
import re
from pathlib import Path

path = Path(os.environ['MANUSCRIPT_PATH'])
t = path.read_text(encoding='utf-8')

# ---------- Abstract ----------
new_abstract = r'''\begin{abstract}
Industrial edge controllers impose strict limits on memory and inference latency. Fault-prediction models for production time series must therefore be uncertainty-aware, cost-sensitive, and explainable. We present a recurrent attention framework built around a two-layer LSTM, normalized per-step Sigmoid attention, Monte Carlo Dropout inference, and a cost-sensitive threshold. A Proposed-Lite variant reduces parameters to 16,374 on average. We evaluate both variants under a unified 10-seed protocol on UR3 CobotOps, NASA C-MAPSS FD001/FD003, and the complete 15-run XJTU-SY bearing dataset, with LSTM, BiLSTM, GRU, Transformer, TCN, PatchTST, TimesNet, random forest, and HistGBM baselines. The full model reaches AUROC 0.996/0.998 and F2 0.866/0.889 on C-MAPSS FD001/FD003, and completes a single CPU forward pass in about 0.66 ms. On XJTU-SY, isotonic calibration reduces ECE from 0.510 to 0.040, and rejecting the 10\% most uncertain windows raises F2 from 0.550 to 0.601. Streaming alerting detects all failure units on FD001, FD003, and XJTU. The framework does not claim universal ranking superiority; its contribution is a quantified reliability, explainability, and edge-deployment layer for industrial fault prediction.
\end{abstract}'''
t = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', lambda m: new_abstract, t, flags=re.S)

# ---------- Intro bullets ----------
new_bullets = r'''Our main findings are as follows:
\begin{itemize}
\item Under the unified 10-seed protocol, the proposed model reached AUROC 0.996/0.998 and F2 0.866/0.889 on C-MAPSS FD001/FD003, and remained competitive on UR3 and XJTU-SY.
\item Proposed-Lite averages 16,374 parameters and retains most of the full-model F2, giving a concrete parameter advantage over nearly every compared deep baseline.
\item MC-Dropout and class weighting provide the most consistent reliability gains; attention is beneficial on gradual degradation and harmful on short windows or strong domain shift.
\item On XJTU-SY, isotonic calibration reduces ECE from 0.510 to 0.040, and MC-based selective prediction raises F2 from 0.550 to 0.601.
\item Streaming alerting detects all failure units on FD001, FD003, and XJTU with false-alarm rates below 0.04 on C-MAPSS, and the full model runs in about 0.66 ms per CPU window.
\end{itemize}'''
start = t.index('Our main findings are as follows:')
end = t.index('\\end{itemize}', start) + len('\\end{itemize}')
t = t[:start] + new_bullets + t[end:]

# ---------- Main Results paragraphs ----------
old_main_start = 'Table~\\ref{tab:main} reports the full comparison with published methods re-implemented under our protocol.'
old_main_end = 'rather than as a universal ranking champion.'
i0 = t.index(old_main_start)
i1 = t.index(old_main_end, i0) + len(old_main_end)
new_main = r'''Table~\ref{tab:main} reports the full comparison under the unified protocol. On C-MAPSS, GRU, Transformer, TCN, PatchTST, and TimesNet reached the highest ranking metrics, while the proposed model stayed within 0.021--0.030 of the best F2 and added uncertainty and attention. On UR3, HistGBM, random forest, Transformer, and TimesNet achieved higher AUROC/AUPRC. On XJTU-SY, all recurrent models degraded under unseen-bearing domain shift; TCN achieved the best AUPRC (0.635) and TimesNet the best F2 (0.655), while the proposed model reached F2 0.550 and AUROC 0.661.

A paired bootstrap over test windows (2,000 resamples) showed that the proposed model's AUROC was lower than the best baseline on all four benchmarks: UR3 versus HistGBM (-0.024, 95\% CI [-0.048, -0.001]), FD001 versus Transformer (-0.002, 95\% CI [-0.003, -0.001]), FD003 versus PatchTST (-0.002, 95\% CI [-0.003, -0.001]), and XJTU versus TCN (-0.058, 95\% CI [-0.076, -0.041]). All intervals excluded zero. We therefore do not claim ranking superiority.

The framework is configurable. A deployment variant selected on validation F2 among the full model and three ablations improves F2 and AUPRC relative to plain LSTM on several benchmarks (Table~\ref{tab:sel}). We present the framework as a reliability-oriented design rather than as a ranking champion.'''
t = t[:i0] + new_main + t[i1:]

# ---------- Discussion ----------
old_disc_start = 'The cross-benchmark results reveal a more nuanced picture than any single-dataset evaluation can provide.'
old_disc_end = 'would likely improve them further.'
i0 = t.index(old_disc_start)
i1 = t.index(old_disc_end, i0) + len(old_disc_end)
new_disc = r'''The cross-benchmark results show where the framework is preferable. Its value is not raw ranking. It lies in three measurable properties: per-window MC-Dropout uncertainty with selective prediction, temporal attention explanations, and a 16k-parameter variant that retains most of the full-model F2. On C-MAPSS, the F2 gap to the best baseline is 0.021--0.030. On UR3 it is 0.036, and on XJTU it is 0.105. For applications where missed faults are costlier than false alarms, and where per-window confidence and timing explanations are required, this trade-off is acceptable.

The XJTU-SY results identify the main boundary. The test bearings come from unseen operating conditions, and the raw predictive uncertainty remains overconfident (ECE 0.510). Isotonic calibration fitted on validation reduces ECE to 0.040, and MC-based rejection raises F2 to 0.601 at 88.8\% coverage. Calibration does not recover ranking, so the framework should be operated with calibration and selective prediction under domain shift. TCN's fixed receptive field transfers better to high-frequency bearing patterns; this remains a plausible mechanism that a dedicated feature-transfer study would need to isolate.

Several limitations remain. UR3 is small and episode-structured; C-MAPSS is simulated; XJTU ranking still lags the best baselines after calibration; hyperparameters were fixed across datasets; deployment metrics are desktop CPU/GPU prototype measurements; and PatchTST and TimesNet are compact re-implementations whose official training schedules may improve them further.'''
t = t[:i0] + new_disc + t[i1:]

# Replace the long "One further limitation" paragraph with a concise version
old_one_start = 'One further limitation deserves explicit discussion: why the proposed model is not the best on every benchmark.'
old_one_end = '16k-parameter edge deployability.'
i0 = t.index(old_one_start)
i1 = t.index(old_one_end, i0) + len(old_one_end)
new_one = r'''One further question is why the proposed model is not the best on every benchmark. On UR3, short repeated cycles favor tree ensembles and Transformer, which exploit phase and threshold information that a 32-dimensional attention context compresses. A plain LSTM preserves the latest transient directly, whereas normalized attention can dilute it when weights are near-uniform. On XJTU-SY, the recurrent attention learns training-bearing trajectories, and MC-Dropout cannot detect out-of-distribution shift; TCN's local kernels transfer better to unseen high-frequency patterns. These mechanisms explain the ranking gaps without contradicting the reliability claims.'''
t = t[:i0] + new_one + t[i1:]

# ---------- Conclusion ----------
i0 = t.index('\\section{Conclusion}')
i1 = t.index('\\section*{Data and Code Availability}', i0)
new_conclusion = r'''\section{Conclusion}
We developed a lightweight, configurable fault-prediction framework for edge industrial time series. Under a unified 10-seed protocol, the full model reached F2 0.866/0.889 and AUROC 0.996/0.998 on C-MAPSS FD001/FD003, and Proposed-Lite retained most of this performance with 16,374 parameters on average. Isotonic calibration reduced the XJTU ECE from 0.510 to 0.040, and MC-based selective prediction raised XJTU F2 from 0.550 to 0.601. Streaming alerting detected all failure units on FD001, FD003, and XJTU. The framework is positioned as a reliability and explainability layer for resource-constrained industrial informatics, with its domain-shift boundary stated explicitly.'''
t = t[:i0] + new_conclusion + t[i1:]

path.write_text(t, encoding='utf-8')
print('nature polish pass done')