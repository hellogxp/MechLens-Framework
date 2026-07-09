import json
import numpy as np
from scipy.stats import binomtest

with open('results/rebuttal_2026may/e1_caa_dola_bootstrap.json') as f:
    d = json.load(f)

caa_samples = d['caa_per_sample']
dola_samples = d['dola_per_sample']

caa_dict = {s['sample_idx']: s['is_correct'] for s in caa_samples}
dola_dict = {s['sample_idx']: s['is_correct'] for s in dola_samples}

common = set(caa_dict.keys()) & set(dola_dict.keys())
print(f"Common samples: {len(common)}")

caa_correct = np.array([caa_dict[i] for i in sorted(common)])
dola_correct = np.array([dola_dict[i] for i in sorted(common)])

n01 = int(np.sum(dola_correct & ~caa_correct))
n10 = int(np.sum(~dola_correct & caa_correct))
n_both_correct = int(np.sum(dola_correct & caa_correct))
n_both_wrong = int(np.sum(~dola_correct & ~caa_correct))

print(f"\nMcNemar's test:")
print(f"  Both correct: {n_both_correct}")
print(f"  Both wrong:   {n_both_wrong}")
print(f"  DoLa correct, CAA wrong: {n01}")
print(f"  CAA correct, DoLa wrong: {n10}")
print(f"  Total discordant: {n01 + n10}")

if n01 + n10 > 0:
    result_two = binomtest(n01, n01 + n10, 0.5, alternative='two-sided')
    result_one = binomtest(n01, n01 + n10, 0.5, alternative='greater')
    print(f"  McNemar exact p (two-tailed): {result_two.pvalue:.4f}")
    print(f"  McNemar exact p (one-tailed): {result_one.pvalue:.4f}")

n = len(caa_correct)
boot_diffs = []
for _ in range(10000):
    idx = np.random.choice(n, n, replace=True)
    boot_diffs.append(dola_correct[idx].mean() - caa_correct[idx].mean())
boot_diffs = np.array(boot_diffs)
p_one_tailed = (boot_diffs <= 0).mean()
print(f"\n  Bootstrap one-tailed p (DoLa > CAA): {p_one_tailed:.4f}")
print(f"  Bootstrap two-tailed p: {min(p_one_tailed * 2, 1.0):.4f}")
print(f"  95% CI: [{np.percentile(boot_diffs, 2.5):.4f}, {np.percentile(boot_diffs, 97.5):.4f}]")

print(f"\n=== SUMMARY ===")
print(f"Baseline MC1: {d['baseline_mc1']:.4f}")
print(f"DoLa MC1:     {d['dola_mc1']:.4f} (+{d['dola_improvement_pct']:.1f}%)")
print(f"CAA MC1:      {d['caa_mc1']:.4f} (+{d['caa_improvement_pct']:.1f}%)")
print(f"DoLa - CAA:   {d['dola_mc1'] - d['caa_mc1']:.4f}")
