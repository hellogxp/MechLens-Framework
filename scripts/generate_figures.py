"""Generate all paper figures from experiment results."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Paths
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Load data
def load_json(path):
    with open(path) as f:
        return json.load(f)

pythia_ct = load_json(os.path.join(RESULTS_DIR, 'pythia_1.4b', 'causal_tracing_results.json'))
qwen_ct = load_json(os.path.join(RESULTS_DIR, 'qwen_0.5b', 'causal_tracing_results.json'))
pythia_contrastive = load_json(os.path.join(RESULTS_DIR, 'pythia_1.4b', 'contrastive_results.json'))
qwen_contrastive = load_json(os.path.join(RESULTS_DIR, 'qwen_0.5b', 'contrastive_results.json'))
qwen_intervention_old = load_json(os.path.join(RESULTS_DIR, 'qwen_0.5b', 'targeted_intervention_results.json'))
qwen_extended = load_json(os.path.join(RESULTS_DIR, 'qwen_0.5b', 'extended_intervention.json'))
pythia_tqa = load_json(os.path.join(RESULTS_DIR, 'pythia_1.4b', 'truthfulqa_results.json'))
qwen_tqa = load_json(os.path.join(RESULTS_DIR, 'qwen_0.5b', 'truthfulqa_results.json'))


# ============================================================
# Figure 1: Causal Tracing Heatmaps (MLP vs Attn, both models)
# ============================================================
def fig1_causal_tracing_heatmaps():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), sharey='row')

    for col_idx, (ct_data, model_name) in enumerate([
        (pythia_ct, 'Pythia-1.4B'),
        (qwen_ct, 'Qwen2.5-0.5B'),
    ]):
        layer_results = ct_data['layer_level']
        # Separate MLP and Attn results
        mlp_results = [r for r in layer_results if r.get('component') == 'mlp' and 'error' not in r]
        attn_results = [r for r in layer_results if r.get('component') == 'attn' and 'error' not in r]

        n_layers = len(mlp_results[0]['scores']) if mlp_results else 24

        # Build matrices
        mlp_labels = []
        mlp_matrix = []
        for r in mlp_results:
            label = r['subject']
            if len(label) > 10:
                label = label[:10] + '..'
            mlp_labels.append(label)
            mlp_matrix.append(r['scores'][:n_layers])

        attn_labels = []
        attn_matrix = []
        for r in attn_results:
            label = r['subject']
            if len(label) > 10:
                label = label[:10] + '..'
            attn_labels.append(label)
            attn_matrix.append(r['scores'][:n_layers])

        if mlp_matrix:
            mlp_arr = np.array(mlp_matrix)
            im0 = axes[0, col_idx].imshow(mlp_arr, aspect='auto', cmap='RdYlBu_r',
                                           vmin=-0.2, vmax=1.0)
            axes[0, col_idx].set_yticks(range(len(mlp_labels)))
            axes[0, col_idx].set_yticklabels(mlp_labels)
            axes[0, col_idx].set_xlabel('Layer')
            axes[0, col_idx].set_title(f'{model_name} — MLP')
            # Mark top layer per prompt
            for i, r in enumerate(mlp_results):
                top = r['top_layer']
                axes[0, col_idx].plot(top, i, 'k*', markersize=8)

        if attn_matrix:
            attn_arr = np.array(attn_matrix)
            im1 = axes[1, col_idx].imshow(attn_arr, aspect='auto', cmap='RdYlBu_r',
                                           vmin=-0.5, vmax=0.5)
            axes[1, col_idx].set_yticks(range(len(attn_labels)))
            axes[1, col_idx].set_yticklabels(attn_labels)
            axes[1, col_idx].set_xlabel('Layer')
            axes[1, col_idx].set_title(f'{model_name} — Attention')
            for i, r in enumerate(attn_results):
                top = r['top_layer']
                axes[1, col_idx].plot(top, i, 'k*', markersize=8)

    # Colorbars
    fig.colorbar(im0, ax=axes[0, :], shrink=0.6, label='Recovery Score (IE)', pad=0.02)
    fig.colorbar(im1, ax=axes[1, :], shrink=0.6, label='Recovery Score (IE)', pad=0.02)

    fig.suptitle('Causal Tracing v2: Layer-Level Recovery Scores', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'causal_tracing_heatmap.pdf')
    fig.savefig(path)
    print(f'Saved: {path}')
    plt.close(fig)


# ============================================================
# Figure 2: Contrastive Layer Importance (both models)
# ============================================================
def fig2_contrastive_importance():
    fig, ax = plt.subplots(figsize=(6.0, 3.0))

    layers = list(range(24))

    # Normalize to [0,1] for comparison
    pythia_imp = np.array(pythia_contrastive['layer_importance'])
    qwen_imp = np.array(qwen_contrastive['layer_importance'])
    pythia_norm = pythia_imp / pythia_imp.max()
    qwen_norm = qwen_imp / qwen_imp.max()

    width = 0.35
    ax.bar([l - width/2 for l in layers], pythia_norm, width, label='Pythia-1.4B', color='#4C72B0', alpha=0.85)
    ax.bar([l + width/2 for l in layers], qwen_norm, width, label='Qwen2.5-0.5B', color='#DD8452', alpha=0.85)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Normalized Importance')
    ax.set_title('Contrastive Activation Analysis: Layer Importance')
    ax.set_xticks(layers)
    ax.legend(loc='upper left')
    ax.set_xlim(-0.5, 23.5)

    # Shade the top-5 region
    ax.axvspan(18.5, 23.5, alpha=0.1, color='red', label='Top-5 region')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'contrastive_importance.pdf')
    fig.savefig(path)
    print(f'Saved: {path}')
    plt.close(fig)


# ============================================================
# Figure 3: Head-Level Causal Tracing (top-10 heads)
# ============================================================
def fig3_head_level():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

    for ax, (ct_data, model_name) in zip(axes, [
        (pythia_ct, 'Pythia-1.4B'),
        (qwen_ct, 'Qwen2.5-0.5B'),
    ]):
        head_results = ct_data.get('head_level', [])
        if not head_results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        # Aggregate top heads across all prompts
        head_counts = {}  # (layer, head) -> list of scores
        for hr in head_results:
            for h in hr['top_heads']:
                key = (h['layer'], h['head'])
                if key not in head_counts:
                    head_counts[key] = []
                head_counts[key].append(h['recovery_score'])

        # Average scores and take top-15
        head_avg = {k: np.mean(v) for k, v in head_counts.items()}
        top_heads = sorted(head_avg.items(), key=lambda x: -x[1])[:15]

        labels = [f'L{k[0]}H{k[1]}' for k, _ in top_heads]
        scores = [v for _, v in top_heads]

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(scores)))
        bars = ax.barh(range(len(labels)), scores, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Avg Recovery Score')
        ax.set_title(f'{model_name}')
        ax.invert_yaxis()

    fig.suptitle('Head-Level Causal Tracing: Top Attention Heads', fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'head_level_tracing.pdf')
    fig.savefig(path)
    print(f'Saved: {path}')
    plt.close(fig)


# ============================================================
# Figure 4: Extended Intervention Results (horizontal bar chart, 20 strategies)
# ============================================================
def fig4_intervention():
    ext = qwen_extended
    strategies_data = ext['strategies']
    baseline_rate = list(strategies_data.values())[0]['hallucination_rate']

    # Group strategies by type for coloring
    type_colors = {
        'mlp_dampen': '#4C72B0',
        'mlp_amplify': '#DD8452',
        'attn_dampen': '#55A868',
        'contrastive_resid': '#C44E52',
        'late_': '#8172B2',
        'early_': '#937860',
    }

    names = list(strategies_data.keys())
    reductions = [strategies_data[n]['reduction'] for n in names]

    # Assign colors
    colors = []
    for n in names:
        color = '#888888'
        for prefix, c in type_colors.items():
            if n.startswith(prefix):
                color = c
                break
        colors.append(color)

    # Sort by reduction (best first)
    order = sorted(range(len(names)), key=lambda i: reductions[i], reverse=True)
    names = [names[i] for i in order]
    reductions = [reductions[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.0, 6.5))
    y = range(len(names))
    bars = ax.barh(y, reductions, color=colors, edgecolor='white', linewidth=0.5, height=0.7)

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace('_', ' ') for n in names], fontsize=8)
    ax.set_xlabel('Hallucination Rate Change')
    ax.set_title(f'Extended Intervention Results on Qwen2.5-0.5B\n(baseline hallucination rate: {baseline_rate:.0%})', fontsize=10)
    ax.invert_yaxis()

    # Value labels
    for bar, r in zip(bars, reductions):
        xpos = bar.get_width()
        ha = 'left' if r >= 0 else 'right'
        ax.text(xpos + (0.003 if r >= 0 else -0.003), bar.get_y() + bar.get_height()/2,
                f'{r:+.1%}', ha=ha, va='center', fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4C72B0', label='MLP Dampen'),
        Patch(facecolor='#DD8452', label='MLP Amplify'),
        Patch(facecolor='#55A868', label='Attn Dampen'),
        Patch(facecolor='#C44E52', label='Contrastive Resid'),
        Patch(facecolor='#8172B2', label='Late Layer'),
        Patch(facecolor='#937860', label='Early Layer'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='lower right')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'intervention_results.pdf')
    fig.savefig(path)
    print(f'Saved: {path}')
    plt.close(fig)


# ============================================================
# Figure 5: Intervention per-type breakdown (representative strategies)
# ============================================================
def fig5_intervention_per_type():
    ext = qwen_extended['strategies']

    # Select representative strategies (best from each category + extremes)
    selected = ['contrastive_resid_0.9', 'attn_dampen_0.7', 'mlp_dampen_0.9',
                'early_mlp_0.85', 'contrastive_resid_0.7', 'late_mlp_0.85']
    selected = [s for s in selected if s in ext]

    halluc_types = ['factual_fabrication', 'causal_error', 'temporal_displacement', 'identity_confusion']
    type_labels = ['Factual\nFabrication', 'Causal\nError', 'Temporal\nDisplacement', 'Identity\nConfusion']

    short_names = []
    for s in selected:
        parts = s.split('_')
        if 'contrastive' in s:
            short_names.append(f'CR-{parts[-1]}')
        elif 'attn' in s:
            short_names.append(f'AD-{parts[-1]}')
        elif 'late' in s:
            short_names.append(f'Late-{parts[-1]}')
        elif 'early' in s:
            short_names.append(f'Early-{parts[-1]}')
        else:
            short_names.append(f'MD-{parts[-1]}')

    fig, axes = plt.subplots(1, 4, figsize=(7.5, 3.5), sharey=True)
    x = np.arange(len(selected))
    width = 0.35

    for ax, htype, tlabel in zip(axes, halluc_types, type_labels):
        baselines = []
        afters = []
        for sn in selected:
            d = ext[sn]['per_type'][htype]
            baselines.append(d['rate'])
            afters.append(d['rate_after'])

        ax.bar(x - width/2, baselines, width, label='Before', color='#4C72B0', alpha=0.8)
        ax.bar(x + width/2, afters, width, label='After', color='#DD8452', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=6, rotation=45, ha='right')
        ax.set_title(tlabel, fontsize=9)
        ax.set_ylim(0, 1.15)

    axes[0].set_ylabel('Hallucination Rate')
    axes[0].legend(fontsize=7, loc='upper left')

    fig.suptitle('Per-Type Hallucination Rates Before/After Intervention (Qwen2.5-0.5B)', fontsize=10, y=1.04)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'intervention_per_type.pdf')
    fig.savefig(path)
    print(f'Saved: {path}')
    plt.close(fig)


# ============================================================
# Figure 6: TruthfulQA per-category comparison
# ============================================================
def fig6_truthfulqa():
    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    categories = list(pythia_tqa['per_category_rates'].keys())
    short_cats = [c.replace(' and ', '\n& ') for c in categories]

    pythia_rates = [pythia_tqa['per_category_rates'][c]['truthful_rate'] for c in categories]
    qwen_rates = [qwen_tqa['per_category_rates'][c]['truthful_rate'] for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, pythia_rates, width, label='Pythia-1.4B', color='#4C72B0', alpha=0.85)
    ax.bar(x + width/2, qwen_rates, width, label='Qwen2.5-0.5B', color='#DD8452', alpha=0.85)

    ax.set_ylabel('Truthful Rate')
    ax.set_title('TruthfulQA Per-Category Truthfulness')
    ax.set_xticks(x)
    ax.set_xticklabels(short_cats, fontsize=7, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 0.6)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'truthfulqa_comparison.pdf')
    fig.savefig(path)
    print(f'Saved: {path}')
    plt.close(fig)


# ============================================================
# Figure 7: Causal tracing MLP line plot (avg across prompts)
# ============================================================
def fig7_causal_avg():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for ax, (ct_data, model_name) in zip(axes, [
        (pythia_ct, 'Pythia-1.4B'),
        (qwen_ct, 'Qwen2.5-0.5B'),
    ]):
        layer_results = ct_data['layer_level']
        mlp_results = [r for r in layer_results if r.get('component') == 'mlp' and 'error' not in r]
        attn_results = [r for r in layer_results if r.get('component') == 'attn' and 'error' not in r]

        if mlp_results:
            mlp_arr = np.array([r['scores'] for r in mlp_results])
            mlp_mean = mlp_arr.mean(axis=0)
            mlp_std = mlp_arr.std(axis=0)
            layers = np.arange(len(mlp_mean))
            ax.plot(layers, mlp_mean, 'o-', color='#C44E52', markersize=3, label='MLP', linewidth=1.5)
            ax.fill_between(layers, mlp_mean - mlp_std, mlp_mean + mlp_std, alpha=0.15, color='#C44E52')

        if attn_results:
            attn_arr = np.array([r['scores'] for r in attn_results])
            attn_mean = attn_arr.mean(axis=0)
            attn_std = attn_arr.std(axis=0)
            layers = np.arange(len(attn_mean))
            ax.plot(layers, attn_mean, 's-', color='#4C72B0', markersize=3, label='Attention', linewidth=1.5)
            ax.fill_between(layers, attn_mean - attn_std, attn_mean + attn_std, alpha=0.15, color='#4C72B0')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Avg Recovery Score')
        ax.set_title(model_name)
        ax.legend(fontsize=8)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    fig.suptitle('Average Causal Tracing Recovery Scores Across Prompts', fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'causal_tracing_avg.pdf')
    fig.savefig(path)
    print(f'Saved: {path}')
    plt.close(fig)


if __name__ == '__main__':
    print('Generating paper figures...')
    fig1_causal_tracing_heatmaps()
    fig2_contrastive_importance()
    fig3_head_level()
    fig4_intervention()
    fig5_intervention_per_type()
    fig6_truthfulqa()
    fig7_causal_avg()
    print('All figures generated.')
