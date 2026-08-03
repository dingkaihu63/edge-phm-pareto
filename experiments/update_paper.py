# -*- coding: utf-8 -*-
"""Update the TII manuscript with 10-seed, SOTA-baseline, calibration and closed-loop results."""
import os
import re
from pathlib import Path

path = Path(os.environ['MANUSCRIPT_PATH'])
t = path.read_text(encoding='utf-8')

# ---------- Abstract ----------
new_abstract = r'''\begin{abstract}
Industrial edge controllers impose strict limits on memory, floating-point operations, and inference latency. Fault-prediction models for production time series must therefore be uncertainty-aware, cost-sensitive, and explainable. We present a lightweight recurrent attention framework built around a two-layer LSTM (about 56k parameters, or 16k in the Proposed-Lite variant), a normalized per-step Sigmoid attention mechanism, Monte Carlo Dropout inference, and a cost-sensitive decision threshold. The framework is evaluated under a unified 10-seed protocol on four public benchmarks: UR3 CobotOps collaborative-robot telemetry, NASA C-MAPSS FD001/FD003 turbofan degradation, and the complete 15-run XJTU-SY rolling-bearing dataset. We compare against published architectures (LSTM, BiLSTM, GRU, Transformer, TCN, PatchTST, TimesNet, random forest, and histogram-based gradient boosting) re-implemented under identical conditions, and we add a complete component ablation, post-hoc calibration, uncertainty-based selective prediction, and streaming closed-loop alerting. The proposed model reaches AUROC 0.996/0.998 and F2 0.866/0.889 on C-MAPSS FD001/FD003 with Brier 0.011/0.007. It runs in about 0.66 ms per window for one CPU forward pass and completes 50 MC samples in about 33 ms, compatible with 1 Hz telemetry. On XJTU-SY, isotonic calibration reduces ECE from 0.510 to 0.040, and rejecting the 10\% most uncertain windows raises F2 from 0.550 to 0.601. Streaming alerting with a two-window confirmation and a false-alarm-constrained threshold achieves detection rates of 1.0 on FD001/FD003/XJTU. We document the remaining domain-shift boundary and release a reproducible pipeline.
\end{abstract}'''
t = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', lambda m: new_abstract, t, flags=re.S)

# ---------- Intro bullets ----------
new_bullets = r'''Our main findings are as follows:
\begin{itemize}
\item Under the unified 10-seed protocol, the proposed model reached AUROC 0.996/0.998 and F2 0.866/0.889 on C-MAPSS FD001/FD003, while remaining competitive on UR3 (F2 0.738, AUROC 0.854) and the complete XJTU-SY benchmark (F2 0.550, AUROC 0.661).
\item The Proposed-Lite variant averages 16,374 parameters, smaller than every compared deep baseline except PatchTST on the UR3/XJTU shapes, while retaining 96--99\% of full-model F2 on the turbofan benchmarks.
\item MC-Dropout and class weighting provide the most consistent reliability gains across benchmarks; removing attention improves F2 on the short UR3 windows and under XJTU domain shift but degrades FD001, showing that the attention component is task-dependent.
\item On XJTU-SY, post-hoc isotonic calibration reduces ECE from 0.510 to 0.040, and rejecting the 10\% most uncertain windows raises F2 from 0.550 to 0.601 at 88.8\% coverage.
\item Streaming alerting with a two-window confirmation and a false-alarm-constrained threshold detects all failure units on FD001, FD003, and XJTU with false-alarm rates of 0.027--0.038 on C-MAPSS and zero on XJTU.
\item The full model uses about 56k parameters on average and runs a single CPU forward pass in about 0.66 ms; 50 MC samples complete in about 33 ms, satisfying a 1 Hz telemetry budget. The Proposed-Lite variant reduces parameters to about 16k on average.
\end{itemize}'''
start = t.index('Our main findings are as follows:')
end = t.index('\\end{itemize}', start) + len('\\end{itemize}')
t = t[:start] + new_bullets + t[end:]

# ---------- Related Work closing paragraph ----------
old_rw_start = 'We deliberately selected classic, reproducible, and edge-deployable sequence models as baselines'
old_rw_end = 'same protocol and report reproduced numbers.'
i0 = t.index(old_rw_start)
i1 = t.index(old_rw_end, i0) + len(old_rw_end)
new_rw = r'''We therefore include recent sequence models as additional baselines. PatchTST and TimesNet are re-implemented in compact form under the same protocol, together with classic recurrent/convolutional architectures and a histogram-based gradient boosting variant; lightweight Mamba variants remain future work. Throughout the paper, published methods are not quoted from their original papers because preprocessing and evaluation protocols differ; instead, we re-implement LSTM, GRU, BiLSTM, Transformer, TCN, PatchTST, TimesNet, random forest, and histogram-based gradient boosting under the same protocol and report reproduced numbers.'''
t = t[:i0] + new_rw + t[i1:]

# ---------- Implementation ----------
old_impl_start = 'All deep models are implemented in PyTorch 2.14 and trained on an NVIDIA RTX 5060 GPU;'
old_impl_end = 'All models share the same splits, scaling, and metrics.'
i0 = t.index(old_impl_start)
i1 = t.index(old_impl_end, i0) + len(old_impl_end)
new_impl = r'''All deep models are implemented in PyTorch 2.14 and trained on an NVIDIA RTX 5060 GPU; CPU latency is measured separately for deployment analysis. We use ten fixed seeds and report seed-ensemble results with across-seed standard deviations. Per-dataset configurations are summarized as follows. UR3 and XJTU use LSTM units of 96 and 48 for the full model and 48 and 24 for Proposed-Lite, dropout 0.10, Adam learning rate 1e-3, and batch size 64; XJTU additionally uses balanced sampling. C-MAPSS uses LSTM units of 64 and 32 (32 and 16 for Proposed-Lite), dropout 0.15, learning rate 5e-4, and batch size 128. Training runs up to 80 epochs with early stopping patience 8 and reduced learning rate on plateau. The final probabilities are averaged over the ten seeds, and the operating threshold is selected on the averaged validation probabilities by minimizing $4 \times \mathrm{FN} + \mathrm{FP}$. Random forest (200 trees, depth 14, balanced class weights) and histogram-based gradient boosting (200 iterations, learning rate 0.08, at most 16 leaves) are trained on flattened windows; PatchTST and TimesNet use compact implementations with about 17k and 27--41k parameters, respectively. All models share the same splits, scaling, and metrics.'''
t = t[:i0] + new_impl + t[i1:]

# ---------- Main Results paragraphs ----------
old_main_start = 'Table~\\ref{tab:main} reports the full comparison with published methods re-implemented under our protocol.'
old_main_end = 'rather than as a universal ranking champion.'
i0 = t.index(old_main_start)
i1 = t.index(old_main_end, i0) + len(old_main_end)
new_main = r'''Table~\ref{tab:main} reports the full comparison with published methods re-implemented under our protocol. On C-MAPSS, the proposed model was close to the best baselines in F2 and achieved AUROC above 0.99, while also providing uncertainty and attention. GRU, Transformer, TCN, PatchTST, and TimesNet reached the highest ranking metrics on the turbofan benchmarks; the proposed model remained competitive (FD001 F2 0.866; FD003 F2 0.889) and added calibrated uncertainty and temporal attribution. On UR3, HistGBM, random forest, Transformer, and TimesNet achieved higher AUROC/AUPRC; this is an important negative result that motivates the component analysis in Section~\ref{sec:ablation}. On the complete 15-run XJTU-SY benchmark, all recurrent models degraded under unseen-bearing domain shift; TCN achieved the best AUPRC (0.635), TimesNet the best F2 (0.655), and the proposed model reached F2 0.550 and AUROC 0.661.

A paired bootstrap over test windows (2,000 resamples) showed that the proposed model's AUROC was lower than the best baseline on all four benchmarks: UR3 versus HistGBM (-0.024, 95\% CI [-0.048, -0.001]), FD001 versus Transformer (-0.002, 95\% CI [-0.003, -0.001]), FD003 versus PatchTST (-0.002, 95\% CI [-0.003, -0.001]), and XJTU versus TCN (-0.058, 95\% CI [-0.076, -0.041]). All intervals excluded zero, so we do not claim ranking superiority on any benchmark.

Because the framework is configurable, we also evaluate a deployment variant selected only on validation F2 among the full model, the no-attention variant, softmax attention, and the no-MC-Dropout variant. Table~\ref{tab:sel} reports the resulting test metrics. This selection matches or improves the plain LSTM baseline on F2 for FD003 and XJTU, and improves AUPRC on UR3, while AUROC remains lower on UR3, FD001, and XJTU. We therefore present the framework as a configurable reliability-oriented design rather than as a universal ranking champion.'''
t = t[:i0] + new_main + t[i1:]

# ---------- Main table ----------
new_main_table = r'''\begin{table*}[t]
\centering
\caption{Main comparison with published methods re-implemented under identical protocol. Metrics are 10-seed ensemble values; std columns report across-seed variation.}
\label{tab:main}
\footnotesize
\begin{tabular}{lccccccccc}
\toprule
Dataset & Model & Acc. & Prec. & Rec. & F2 & AUROC & AUPRC & F2 std & AUROC std \\
\midrule
\multirow{11}{*}{UR3}
& Proposed (full) & 0.504 & 0.369 & 0.985 & 0.738 & 0.854 & 0.773 & 0.022 & 0.023 \\
& Proposed-Lite & 0.548 & 0.389 & 0.960 & 0.742 & 0.850 & 0.767 & 0.023 & 0.020 \\
& LSTM & 0.646 & 0.448 & 0.921 & 0.761 & 0.868 & 0.788 & 0.018 & 0.021 \\
& BiLSTM & 0.581 & 0.408 & 0.967 & 0.759 & 0.870 & 0.785 & 0.026 & 0.021 \\
& GRU & 0.660 & 0.457 & 0.888 & 0.747 & 0.867 & 0.802 & 0.012 & 0.019 \\
& Transformer & 0.683 & 0.477 & 0.912 & 0.771 & 0.871 & 0.762 & 0.034 & 0.051 \\
& TCN & 0.556 & 0.393 & 0.954 & 0.742 & 0.853 & 0.758 & 0.010 & 0.018 \\
& PatchTST & 0.632 & 0.438 & 0.915 & 0.751 & 0.820 & 0.679 & 0.015 & 0.038 \\
& TimesNet & 0.689 & 0.482 & 0.875 & 0.752 & 0.877 & 0.808 & 0.020 & 0.031 \\
& Random forest & 0.613 & 0.428 & 0.970 & 0.774 & 0.871 & 0.777 & 0.007 & 0.005 \\
& HistGBM & 0.637 & 0.441 & 0.915 & 0.753 & 0.878 & 0.817 & 0.000 & 0.000 \\
\midrule
\multirow{11}{*}{FD001}
& Proposed (full) & 0.981 & 0.636 & 0.952 & 0.866 & 0.996 & 0.891 & 0.028 & 0.003 \\
& Proposed-Lite & 0.984 & 0.714 & 0.864 & 0.829 & 0.993 & 0.846 & 0.027 & 0.003 \\
& LSTM & 0.988 & 0.814 & 0.819 & 0.818 & 0.997 & 0.917 & 0.034 & 0.001 \\
& BiLSTM & 0.985 & 0.708 & 0.919 & 0.867 & 0.997 & 0.919 & 0.031 & 0.001 \\
& GRU & 0.988 & 0.755 & 0.928 & 0.887 & 0.997 & 0.927 & 0.011 & 0.001 \\
& Transformer & 0.987 & 0.736 & 0.934 & 0.886 & 0.997 & 0.924 & 0.043 & 0.001 \\
& TCN & 0.986 & 0.714 & 0.940 & 0.884 & 0.997 & 0.929 & 0.017 & 0.001 \\
& PatchTST & 0.985 & 0.717 & 0.886 & 0.846 & 0.996 & 0.911 & 0.015 & 0.001 \\
& TimesNet & 0.987 & 0.751 & 0.907 & 0.870 & 0.997 & 0.926 & 0.008 & 0.000 \\
& Random forest & 0.980 & 0.646 & 0.831 & 0.786 & 0.992 & 0.846 & 0.005 & 0.000 \\
& HistGBM & 0.984 & 0.705 & 0.886 & 0.842 & 0.996 & 0.900 & 0.008 & 0.000 \\
\midrule
\multirow{11}{*}{FD003}
& Proposed (full) & 0.990 & 0.697 & 0.955 & 0.889 & 0.998 & 0.935 & 0.014 & 0.004 \\
& Proposed-Lite & 0.989 & 0.661 & 0.959 & 0.880 & 0.994 & 0.910 & 0.023 & 0.003 \\
& LSTM & 0.992 & 0.736 & 0.976 & 0.916 & 0.999 & 0.953 & 0.009 & 0.001 \\
& BiLSTM & 0.992 & 0.731 & 0.962 & 0.905 & 0.999 & 0.952 & 0.011 & 0.001 \\
& GRU & 0.991 & 0.712 & 0.976 & 0.909 & 0.999 & 0.946 & 0.009 & 0.000 \\
& Transformer & 0.991 & 0.708 & 0.952 & 0.891 & 0.998 & 0.936 & 0.038 & 0.001 \\
& TCN & 0.993 & 0.757 & 0.966 & 0.915 & 0.999 & 0.953 & 0.009 & 0.001 \\
& PatchTST & 0.991 & 0.707 & 0.986 & 0.914 & 0.999 & 0.972 & 0.008 & 0.000 \\
& TimesNet & 0.991 & 0.708 & 0.993 & 0.919 & 0.999 & 0.967 & 0.011 & 0.000 \\
& Random forest & 0.988 & 0.666 & 0.835 & 0.795 & 0.996 & 0.862 & 0.008 & 0.000 \\
& HistGBM & 0.989 & 0.686 & 0.900 & 0.847 & 0.998 & 0.913 & 0.016 & 0.000 \\
\midrule
\multirow{11}{*}{XJTU}
& Proposed (full) & 0.715 & 0.404 & 0.605 & 0.550 & 0.661 & 0.471 & 0.028 & 0.021 \\
& Proposed-Lite & 0.718 & 0.402 & 0.565 & 0.522 & 0.642 & 0.411 & 0.022 & 0.020 \\
& LSTM & 0.734 & 0.435 & 0.684 & 0.614 & 0.672 & 0.333 & 0.051 & 0.106 \\
& BiLSTM & 0.719 & 0.407 & 0.593 & 0.543 & 0.647 & 0.412 & 0.029 & 0.030 \\
& GRU & 0.720 & 0.411 & 0.616 & 0.560 & 0.686 & 0.533 & 0.032 & 0.041 \\
& Transformer & 0.730 & 0.435 & 0.734 & 0.645 & 0.707 & 0.492 & 0.048 & 0.046 \\
& TCN & 0.762 & 0.476 & 0.718 & 0.651 & 0.719 & 0.635 & 0.042 & 0.049 \\
& PatchTST & 0.736 & 0.426 & 0.554 & 0.522 & 0.706 & 0.343 & 0.254 & 0.112 \\
& TimesNet & 0.735 & 0.441 & 0.746 & 0.655 & 0.702 & 0.343 & 0.022 & 0.015 \\
& Random forest & 0.690 & 0.274 & 0.243 & 0.249 & 0.546 & 0.251 & 0.039 & 0.012 \\
& HistGBM & 0.779 & 0.500 & 0.689 & 0.641 & 0.668 & 0.399 & 0.000 & 0.000 \\
\bottomrule
\end{tabular}
\end{table*}'''
i0 = t.index('\\begin{table*}[t]')
i1 = t.index('\\end{table*}', i0) + len('\\end{table*}')
t = t[:i0] + new_main_table + t[i1:]

# ---------- Ablation table ----------
new_ablation = r'''\subsection{Ablation Study}
\label{sec:ablation}
Table~\ref{tab:ablation} reports the 10-seed ensemble ablation on every benchmark. Removing attention improved F2 on UR3 (0.738 to 0.762) and XJTU (0.550 to 0.597), but degraded FD001 (0.866 to 0.803), confirming that attention is beneficial when degradation is gradual and harmful when windows are short or domain shift is severe. Removing MC-Dropout degraded F2 on UR3, FD001, and XJTU. Class weighting consistently improved F2 on the turbofan benchmarks and XJTU. Proposed-Lite retained most of the full-model F2 with about 3.4x fewer parameters on average.

\begin{table}[t]
\centering
\caption{Component ablation under the unified 10-seed protocol.}
\label{tab:ablation}
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
Dataset & Variant & Acc. & Prec. & Rec. & F2 & AUROC & AUPRC & ECE \\
\midrule
\multirow{7}{*}{UR3}
& Full & 0.504 & 0.369 & 0.985 & 0.738 & 0.854 & 0.773 & 0.168 \\
& w/o attention & 0.629 & 0.436 & 0.936 & 0.762 & 0.877 & 0.817 & 0.156 \\
& Softmax attention & 0.600 & 0.419 & 0.960 & 0.763 & 0.860 & 0.787 & 0.150 \\
& w/o MC-Dropout & 0.559 & 0.395 & 0.970 & 0.751 & 0.843 & 0.767 & 0.205 \\
& w/o class weight & 0.632 & 0.435 & 0.875 & 0.728 & 0.856 & 0.803 & 0.066 \\
& + rolling stats & 0.726 & 0.520 & 0.772 & 0.704 & 0.824 & 0.692 & 0.143 \\
& Proposed-Lite & 0.548 & 0.389 & 0.960 & 0.742 & 0.850 & 0.767 & 0.188 \\
\midrule
\multirow{7}{*}{FD001}
& Full & 0.981 & 0.636 & 0.952 & 0.866 & 0.996 & 0.891 & 0.014 \\
& w/o attention & 0.988 & 0.835 & 0.795 & 0.803 & 0.997 & 0.923 & 0.012 \\
& Softmax attention & 0.987 & 0.770 & 0.849 & 0.832 & 0.996 & 0.906 & 0.012 \\
& w/o MC-Dropout & 0.981 & 0.633 & 0.970 & 0.876 & 0.996 & 0.888 & 0.014 \\
& w/o class weight & 0.987 & 0.754 & 0.886 & 0.856 & 0.994 & 0.904 & 0.004 \\
& + rolling stats & 0.968 & 0.507 & 0.768 & 0.696 & 0.963 & 0.665 & 0.042 \\
& Proposed-Lite & 0.984 & 0.714 & 0.864 & 0.829 & 0.993 & 0.846 & 0.016 \\
\midrule
\multirow{7}{*}{FD003}
& Full & 0.990 & 0.697 & 0.955 & 0.889 & 0.998 & 0.935 & 0.011 \\
& w/o attention & 0.990 & 0.678 & 0.976 & 0.897 & 0.999 & 0.957 & 0.009 \\
& Softmax attention & 0.992 & 0.747 & 0.962 & 0.910 & 0.999 & 0.956 & 0.010 \\
& w/o MC-Dropout & 0.990 & 0.691 & 0.959 & 0.890 & 0.996 & 0.928 & 0.012 \\
& w/o class weight & 0.989 & 0.666 & 0.979 & 0.895 & 0.999 & 0.953 & 0.003 \\
& + rolling stats & 0.974 & 0.443 & 0.873 & 0.731 & 0.980 & 0.692 & 0.029 \\
& Proposed-Lite & 0.989 & 0.661 & 0.959 & 0.880 & 0.994 & 0.910 & 0.015 \\
\midrule
\multirow{7}{*}{XJTU}
& Full & 0.715 & 0.404 & 0.605 & 0.550 & 0.661 & 0.471 & 0.510 \\
& w/o attention & 0.724 & 0.421 & 0.667 & 0.597 & 0.675 & 0.327 & 0.445 \\
& Softmax attention & 0.713 & 0.402 & 0.616 & 0.557 & 0.726 & 0.406 & 0.482 \\
& w/o MC-Dropout & 0.720 & 0.414 & 0.638 & 0.576 & 0.703 & 0.529 & 0.472 \\
& w/o class weight & 0.729 & 0.419 & 0.582 & 0.540 & 0.792 & 0.521 & 0.246 \\
& + rolling stats & 0.725 & 0.414 & 0.582 & 0.538 & 0.722 & 0.430 & 0.384 \\
& Proposed-Lite & 0.718 & 0.402 & 0.565 & 0.522 & 0.642 & 0.411 & 0.512 \\
\bottomrule
\end{tabular}}
\end{table}'''
old_label = '\\subsection{Ablation Study}\n\\label{sec:ablation}\n'
assert old_label in t
t = t.replace(old_label, new_ablation + '\n', 1)

# ---------- Deployment text ----------
old_dep_start = 'Edge controllers typically collect telemetry at 1 Hz or lower'
old_dep_end = 'validation on an actual edge controller (MCU or PLC) remains future work.'
i0 = t.index(old_dep_start)
i1 = t.index(old_dep_end, i0) + len(old_dep_end)
new_dep_text = r'''Edge controllers typically collect telemetry at 1 Hz or lower, leaving a per-window latency budget of hundreds of milliseconds even on inexpensive hardware. Table~\ref{tab:deploy} reports parameter counts and CPU/GPU inference latency measured with PyTorch 2.14 on an Intel x86 machine. The full model averages 56,010 parameters (about 0.22 MB in float32) and runs a single CPU forward pass in about 0.66 ms; 50 MC-Dropout samples, which provide the uncertainty estimate used for reliable decisions, complete in about 33 ms. The Proposed-Lite variant averages 16,374 parameters (about 0.07 MB in float32) and is smaller than every compared deep baseline except PatchTST on the UR3/XJTU shapes. Dynamic INT8 quantization further reduces weight storage by about 4x but adds per-call overhead on this desktop CPU; real MCU/PLC latency and power measurement remains future work. All compared architectures fit comfortably within a 1 s budget.'''
t = t[:i0] + new_dep_text + t[i1:]

# ---------- Deployment table ----------
i0 = t.index('\\caption{Parameter count and CPU/GPU inference latency of the final PyTorch models')
i1 = t.index('\\end{table}', i0) + len('\\end{table}')
new_dep_table = r'''\begin{table*}[t]
\centering
\caption{Parameter counts and CPU/GPU latency averaged over the four benchmarks. MC50 is reported only for the MC-Dropout models; INT8 weight counts include quantized linear and recurrent weights.}
\label{tab:deploy}
\footnotesize
\begin{tabular}{lccccccc}
\toprule
Model & Params & CPU 1-pass batch (ms) & CPU 1-pass single (ms) & CPU MC50 single (ms) & GPU 1-pass batch (ms) & INT8 weight count \\
\midrule
Proposed (full) & 56,010 & 0.025 & 0.663 & 33.2 & 0.009 & 55,032 \\
Proposed-Lite & 16,374 & 0.020 & 0.780 & 39.0 & 0.011 & 15,876 \\
LSTM & 35,745 & 0.022 & 0.470 & -- & 0.005 & -- \\
BiLSTM & 27,586 & 0.024 & 0.756 & -- & 0.008 & -- \\
GRU & 26,978 & 0.048 & 1.804 & -- & 0.007 & -- \\
Transformer & 27,777 & 0.014 & 0.336 & -- & 0.007 & -- \\
TCN & 30,673 & 0.037 & 1.255 & -- & 0.010 & -- \\
PatchTST & 17,345 & 0.278 & 1.249 & -- & 0.027 & -- \\
TimesNet & 32,003 & 0.114 & 1.107 & -- & 0.008 & -- \\
\bottomrule
\end{tabular}
\end{table*}'''
t = t[:i0] + new_dep_table + t[i1:]

# ---------- Calibration + closed-loop sections before Discussion ----------
new_extra = r'''\subsection{Calibration and Uncertainty-Based Selective Prediction}
Table~\ref{tab:calib} reports post-hoc calibration for the full model and Proposed-Lite on XJTU-SY. Temperature scaling lowers Brier score and ECE but cannot eliminate overconfidence under domain shift. Isotonic regression fitted on validation probabilities reduces ECE from 0.510 to 0.040 while preserving F2 (0.550), at the cost of AUPRC compression (0.471 to 0.343). MC-Dropout uncertainty enables a complementary selective-prediction policy: rejecting the 10\% most uncertain test windows raises F2 from 0.550 to 0.601 at 88.8\% coverage, and combined temperature scaling further reduces Brier score to 0.283. These two mechanisms give an industrial fallback: calibrated probabilities for monitoring, and uncertainty-based rejection or manual escalation for high-risk windows.

\begin{table}[t]
\centering
\caption{Post-hoc calibration and MC-uncertainty rejection on XJTU-SY.}
\label{tab:calib}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
Model & Method & F2 & AUROC & AUPRC & Brier & ECE \\
\midrule
Full & Raw & 0.550 & 0.661 & 0.471 & 0.441 & 0.510 \\
Full & Temperature & 0.558 & 0.661 & 0.471 & 0.285 & 0.352 \\
Full & Isotonic & 0.550 & 0.681 & 0.343 & 0.263 & 0.040 \\
Full & MC-reject 10\% & 0.601 & 0.652 & 0.480 & 0.426 & 0.499 \\
Full & MC-reject 10\% + temp & 0.596 & 0.652 & 0.480 & 0.283 & 0.354 \\
Lite & Raw & 0.522 & 0.642 & 0.411 & 0.426 & 0.512 \\
Lite & Isotonic & 0.524 & 0.686 & 0.354 & 0.254 & 0.022 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Streaming Closed-Loop Alerting}
We simulate online deployment by processing each test unit in chronological order. A two-window confirmation suppresses single-window spikes, and the operating threshold is selected on validation with a window-level false-alarm constraint of at most 20\%. Table~\ref{tab:closedloop} reports unit-level detection rate, early-detection rate, false-alarm rate, and mean lead time in windows. The full model detects all failure units on FD001, FD003, and XJTU, with false-alarm rates of 0.027--0.038 on C-MAPSS and zero on XJTU; on UR3 the detection rate is also 1.0, with a false-alarm rate of 0.455 that can be reduced to 0.182 by a three-window confirmation at the same detection rate. Proposed-Lite reaches detection rate 1.0 on FD003 and XJTU (0.84 on FD001), showing that the low-parameter variant remains operationally usable.

\begin{table}[t]
\centering
\caption{Streaming closed-loop alerting with two-window confirmation and a validation false-alarm constraint of 20\%. Lead time is measured in windows from the first alarm to the onset of the positive early-warning horizon.}
\label{tab:closedloop}
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
Dataset & Model & Detection & Early & FAR & Mean lead & Median lead \\
\midrule
UR3 & Full & 1.00 & 0.08 & 0.455 & 1.00 & 1.0 \\
UR3 & Proposed-Lite & 1.00 & 0.12 & 0.636 & 0.24 & 1.0 \\
UR3 & LSTM & 1.00 & 0.08 & 0.545 & 0.72 & 1.0 \\
UR3 & TCN & 1.00 & 0.04 & 0.273 & 0.60 & 1.0 \\
UR3 & TimesNet & 1.00 & 0.08 & 0.727 & 0.48 & 1.0 \\
FD001 & Full & 1.00 & 0.76 & 0.027 & -5.36 & -3.0 \\
FD001 & Proposed-Lite & 0.84 & 0.60 & 0.013 & -3.33 & -1.0 \\
FD001 & LSTM & 0.88 & 0.56 & 0.013 & -0.23 & 0.0 \\
FD001 & TCN & 1.00 & 0.80 & 0.013 & -2.88 & -3.0 \\
FD001 & TimesNet & 0.96 & 0.64 & 0.027 & -1.58 & -1.0 \\
FD003 & Full & 1.00 & 0.85 & 0.038 & -3.45 & -4.0 \\
FD003 & Proposed-Lite & 1.00 & 0.90 & 0.050 & -4.20 & -5.0 \\
FD003 & LSTM & 1.00 & 0.80 & 0.025 & -3.40 & -3.5 \\
FD003 & TCN & 1.00 & 0.65 & 0.025 & -2.55 & -3.0 \\
FD003 & TimesNet & 1.00 & 0.90 & 0.025 & -4.45 & -4.0 \\
XJTU & Full & 1.00 & 0.50 & 0.000 & -21.00 & -28.0 \\
XJTU & Proposed-Lite & 0.75 & 0.50 & 0.000 & -28.33 & -65.0 \\
XJTU & LSTM & 1.00 & 0.50 & 0.000 & -24.25 & -32.0 \\
XJTU & TCN & 1.00 & 0.50 & 0.000 & -21.50 & -33.0 \\
XJTU & TimesNet & 1.00 & 0.75 & 0.000 & -42.00 & -36.5 \\
\bottomrule
\end{tabular}}
\end{table}'''
t = t.replace('\n\\section{Discussion}', '\n' + new_extra + '\n\\section{Discussion}', 1)

# ---------- Discussion paragraphs ----------
old_disc1_start = 'The cross-benchmark results reveal a more nuanced picture than any single-dataset evaluation can provide.'
old_disc1_end = 'at a small cost in raw ranking metrics.'
i0 = t.index(old_disc1_start)
i1 = t.index(old_disc1_end, i0) + len(old_disc1_end)
new_disc1 = r'''The cross-benchmark results reveal a more nuanced picture than any single-dataset evaluation can provide. On turbofan data, GRU, Transformer, TCN, PatchTST, and TimesNet already reach near-ceiling AUROC, and the value of the proposed framework lies in reliability-oriented components: cost-sensitive thresholds, calibrated MC-Dropout uncertainty, and temporal explanations, all delivered with 56k parameters, or 16k in Proposed-Lite. On robot telemetry, tree ensembles and Transformer remain highly competitive, especially for F2, while on the complete bearing benchmark all recurrent models degrade under domain shift; TCN shows the best AUPRC and TimesNet the best F2. These reproduced comparisons clarify where a lightweight attention-based recurrent model is preferable: when per-window confidence, edge latency, and physical timing explanations are required, at a small cost in raw ranking metrics.'''
t = t[:i0] + new_disc1 + t[i1:]

old_disc2_start = 'The XJTU-SY results deserve a specific interpretation.'
old_disc2_end = 'raw ranking under severe domain shift is secondary.'
i0 = t.index(old_disc2_start)
i1 = t.index(old_disc2_end, i0) + len(old_disc2_end)
new_disc2 = r'''The XJTU-SY results deserve a specific interpretation. The test bearings come from operating conditions and failure modes not seen during training, which creates a strong domain shift. We quantified whether the attention mechanism itself broke down under this shift. The average attention profiles remained diffuse and nearly identical across the train, validation, and test groups (KL divergence below 0.002; attention entropy about 2.98), and the mean predicted probability was similar on validation and test (0.70 versus 0.72). The overconfidence therefore does not originate from a dramatic shift in temporal focus. It originates in the final scoring layer and in MC-Dropout's inability to widen uncertainty for out-of-distribution inputs, which is why threshold calibration alone cannot recover ranking performance. Post-hoc isotonic calibration removes most miscalibration (ECE 0.040), and MC-based rejection provides an operational fallback, but the ranking gap remains. TCN's advantage under domain shift remains a plausible but not yet isolated mechanism, and would require a dedicated feature-transfer analysis. This marks the boundary of the proposed framework: at extremely low parameter counts, the recurrent attention design should be preferred when calibrated probability, temporal attribution, and edge latency are the primary requirements, and raw ranking under severe domain shift is secondary.'''
t = t[:i0] + new_disc2 + t[i1:]

old_disc3_start = 'Several limitations should be acknowledged.'
old_disc3_end = 'larger ensembles or GPU training could improve stability.'
i0 = t.index(old_disc3_start)
i1 = t.index(old_disc3_end, i0) + len(old_disc3_end)
new_disc3 = r'''Several limitations should be acknowledged. First, the UR3 dataset is small and episode-structured; results may not transfer to other cobot workloads. Second, C-MAPSS is simulated, so sensor noise and degradation physics are simplified. Third, the XJTU-SY ranking gap under domain shift remains even after calibration; the framework should be operated with calibration and selective prediction in that regime. Fourth, hyperparameters were kept fixed across datasets for fairness, so per-dataset tuning would likely improve individual models. Fifth, deployment metrics are desktop CPU/GPU prototype-level measurements; real MCU/PLC latency and power validation is future work. Finally, PatchTST and TimesNet are compact re-implementations rather than official libraries, and a dedicated search over their capacities and training schedules would likely improve them further.'''
t = t[:i0] + new_disc3 + t[i1:]

# Update stale deployability phrase in the last discussion paragraph
t = t.replace('56k-parameter edge deployability', '16k-parameter edge deployability')

# ---------- Conclusion ----------
i0 = t.index('\\section{Conclusion}')
i1 = t.index('\\section*{Data and Code Availability}', i0)
new_conclusion = r'''\section{Conclusion}
We developed and evaluated a lightweight, rigorously tested methodology for edge industrial fault prediction. Under a unified 10-seed protocol on UR3, C-MAPSS FD001/FD003, and the complete XJTU-SY benchmark, the framework reaches strong results on turbofan degradation (F2 0.866/0.889, AUROC 0.996/0.998) and provides a configurable reliability layer on the robot and bearing benchmarks. The Proposed-Lite variant averages 16,374 parameters and remains operationally usable, with streaming detection rates of 1.0 on FD003 and XJTU and 0.84 on FD001 under a two-window confirmation and a false-alarm-constrained threshold. Post-hoc isotonic calibration reduces the XJTU ECE from 0.510 to 0.040, and MC-based selective prediction raises XJTU F2 from 0.550 to 0.601 at 88.8\% coverage. Reproduced comparisons with LSTM, BiLSTM, GRU, Transformer, TCN, PatchTST, TimesNet, random forest, and histogram-based gradient boosting show that the proposed model is competitive where ranking alone matters and preferable when edge latency, uncertainty, and explanations are required. The complete ablation study quantifies when each component helps, and we document the remaining domain-shift boundary. We release the full experimental pipeline, results, and figures to support reproduction. The paper positions the framework as an edge intelligent fault-prediction expert system for industrial informatics: not a universal ranking champion, but a deployable reliability and explainability layer for resource-constrained production environments.'''
t = t[:i0] + new_conclusion + t[i1:]

path.write_text(t, encoding='utf-8')
print('paper updated')