"""
Generate FEP Heatmap Visualizations for MechLens Paper
- Figure 1: Cross-architecture FEP heatmap (Qwen vs Llama vs Mistral)
- Figure 2: Computability-Memorization Spectrum comparison
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os

RESULTS_DIR = "/root/llm-mi/results"
OUTPUT_DIR = "/root/llm-mi/paper/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_fep_data():
    """Load FEP data for all three models."""
    models = {}

    # Qwen2.5-7B
    with open(f"{RESULTS_DIR}/fep_analysis/fep_detection_results.json") as f:
        models['Qwen2.5-7B'] = json.load(f)

    # Llama-3.1-8B
    with open(f"{RESULTS_DIR}/cross_architecture/fep_meta_llama_Llama_3.1_8B.json") as f:
        models['Llama-3.1-8B'] = json.load(f)

    # Mistral-7B
    with open(f"{RESULTS_DIR}/cross_architecture/fep_mistralai_Mistral_7B_v0.1.json") as f:
        models['Mistral-7B'] = json.load(f)

    return models


def select_representative_samples(data, n_samples=50):
    """Select representative samples: mix of early and late crystallization."""
    samples = data['per_sample_results']
    # Sort by FEP layer to show the transition pattern
    sorted_samples = sorted(samples, key=lambda x: x['fep_layer'])

    # Take a stratified sample: some early, mostly late (reflecting true distribution)
    n_layers = data['n_layers']
    early = [s for s in sorted_samples if s['fep_layer'] < n_layers]
    late = [s for s in sorted_samples if s['fep_layer'] >= n_layers]

    # Take up to 15 early crystallizers and fill rest with late
    n_early = min(15, len(early))
    n_late = n_samples - n_early

    selected = early[:n_early] + late[:n_late]
    # Re-sort by FEP for visual clarity
    selected.sort(key=lambda x: (x['fep_layer'], x['category']))
    return selected


def create_rank_matrix(samples, n_layers):
    """Create a matrix of log-ranks for heatmap visualization."""
    matrix = np.zeros((len(samples), n_layers))
    for i, s in enumerate(samples):
        ranks = s['layer_ranks'][:n_layers]
        # Use log10(rank) for better color scale, cap at log10(10)=1 for "in top-10"
        for j, r in enumerate(ranks):
            matrix[i, j] = np.log10(max(r, 1))
    return matrix


def figure1_cross_architecture_heatmap(models):
    """
    Main FEP visualization: 3-panel heatmap showing crystallization across architectures.
    Dark = high rank (knowledge invisible), Bright = low rank (knowledge visible).
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 1, height_ratios=[28, 32, 32], hspace=0.35)

    model_order = ['Qwen2.5-7B', 'Llama-3.1-8B', 'Mistral-7B']
    titles = [
        'Qwen2.5-7B (28 layers, 85.9% final-layer crystallization)',
        'Llama-3.1-8B (32 layers, 71.0% final-layer crystallization)',
        'Mistral-7B (32 layers, 27.1% final-layer crystallization)'
    ]

    # Shared colormap: dark blue = invisible (high rank), bright yellow = visible (low rank)
    cmap = plt.cm.RdYlBu_r  # reversed: blue=high rank, red/yellow=low rank
    # Actually let's use a custom one: dark for invisible, bright for visible
    colors_list = ['#1a1a2e', '#16213e', '#0f3460', '#e94560', '#ff9a3c', '#fff176']
    cmap = mcolors.LinearSegmentedColormap.from_list('crystallization', colors_list, N=256)

    for idx, model_name in enumerate(model_order):
        data = models[model_name]
        n_layers = data['n_layers']
        samples = select_representative_samples(data, n_samples=60)
        matrix = create_rank_matrix(samples, n_layers)

        ax = fig.add_subplot(gs[idx])

        # The key visual: most of the heatmap should be dark (rank >> 10),
        # with the rightmost column(s) suddenly bright (rank <= 10)
        im = ax.imshow(matrix, aspect='auto', cmap=cmap,
                       vmin=0, vmax=5,  # log10(1)=0 to log10(100000)=5
                       interpolation='nearest')

        # Mark the top-10 threshold
        ax.set_title(titles[idx], fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Layer Index' if idx == 2 else '', fontsize=10)
        ax.set_ylabel('Samples (sorted by FEP)', fontsize=9)

        # X ticks
        if n_layers == 28:
            ax.set_xticks(range(0, 28, 2))
            ax.set_xticklabels(range(0, 28, 2), fontsize=8)
        else:
            ax.set_xticks(range(0, 32, 2))
            ax.set_xticklabels(range(0, 32, 2), fontsize=8)

        ax.set_yticks([])

        # Add FEP markers
        for i, s in enumerate(samples):
            fep = s['fep_layer'] - 1  # 0-indexed layer
            if fep < n_layers - 1:  # Only mark early crystallizers
                ax.plot(fep, i, 'w*', markersize=4, markeredgewidth=0.3, markeredgecolor='black')

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('log₁₀(rank of correct answer)', fontsize=10)
    cbar.set_ticks([0, 1, 2, 3, 4, 5])
    cbar.set_ticklabels(['1\n(top-1)', '10\n(top-10)', '100', '1K', '10K', '100K'])

    fig.suptitle('Late Crystallization of Factual Knowledge Across Architectures',
                 fontsize=14, fontweight='bold', y=0.98)

    # Add annotation
    fig.text(0.5, 0.01,
             'Dark regions = correct answer invisible (rank >> 10). '
             'Bright regions = correct answer visible (rank ≤ 10). '
             'White stars = Factual Emergence Point (FEP).',
             ha='center', fontsize=9, style='italic', color='#555')

    plt.savefig(f'{OUTPUT_DIR}/fep_crystallization_heatmap.pdf',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'{OUTPUT_DIR}/fep_crystallization_heatmap.png',
                bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/fep_crystallization_heatmap.pdf")


def figure2_spectrum_comparison(models):
    """
    Computability-Memorization Spectrum: compare FEP distributions
    for computable vs memorized categories across architectures.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    model_order = ['Qwen2.5-7B', 'Llama-3.1-8B', 'Mistral-7B']
    computable_cats = {'Logical Falsehood', 'Statistics', 'Logical Falsehoods'}
    memorized_cats = {'History', 'Psychology', 'Weather', 'Misconceptions: Topical',
                      'Confusion: People', 'Misinformation'}

    for idx, model_name in enumerate(model_order):
        ax = axes[idx]
        data = models[model_name]
        n_layers = data['n_layers']
        samples = data['per_sample_results']

        # Collect per-layer average rank for computable vs memorized
        comp_ranks_by_layer = [[] for _ in range(n_layers)]
        memo_ranks_by_layer = [[] for _ in range(n_layers)]

        for s in samples:
            cat = s['category']
            ranks = s['layer_ranks'][:n_layers]
            is_comp = any(c in cat for c in ['Logical', 'Statistics', 'Logic'])
            is_memo = any(c in cat for c in ['History', 'Psychology', 'Weather', 'Confusion: People'])

            if is_comp:
                for l, r in enumerate(ranks):
                    comp_ranks_by_layer[l].append(np.log10(max(r, 1)))
            elif is_memo:
                for l, r in enumerate(ranks):
                    memo_ranks_by_layer[l].append(np.log10(max(r, 1)))

        # Compute means
        layers = list(range(n_layers))
        comp_means = [np.mean(x) if x else 0 for x in comp_ranks_by_layer]
        memo_means = [np.mean(x) if x else 0 for x in memo_ranks_by_layer]
        comp_stds = [np.std(x) if x else 0 for x in comp_ranks_by_layer]
        memo_stds = [np.std(x) if x else 0 for x in memo_ranks_by_layer]

        # Plot
        ax.plot(layers, comp_means, 'o-', color='#2196F3', markersize=3,
                label='Computable (Logical)', linewidth=1.5)
        ax.fill_between(layers,
                        [m - s for m, s in zip(comp_means, comp_stds)],
                        [m + s for m, s in zip(comp_means, comp_stds)],
                        alpha=0.2, color='#2196F3')

        ax.plot(layers, memo_means, 's-', color='#F44336', markersize=3,
                label='Memorized (History/Psych)', linewidth=1.5)
        ax.fill_between(layers,
                        [m - s for m, s in zip(memo_means, memo_stds)],
                        [m + s for m, s in zip(memo_means, memo_stds)],
                        alpha=0.2, color='#F44336')

        # Top-10 threshold line
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(1, 0.7, 'Top-10\nthreshold', fontsize=7, color='green', alpha=0.8)

        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer Index', fontsize=10)
        if idx == 0:
            ax.set_ylabel('log₁₀(rank of correct answer)', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylim(-0.2, 5.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Computability–Memorization Spectrum Across Architectures',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(f'{OUTPUT_DIR}/computability_memorization_spectrum.pdf',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'{OUTPUT_DIR}/computability_memorization_spectrum.png',
                bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/computability_memorization_spectrum.pdf")


def figure3_fep_distribution_comparison(models):
    """
    FEP distribution across architectures: histogram showing where knowledge emerges.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    model_order = ['Qwen2.5-7B', 'Llama-3.1-8B', 'Mistral-7B']
    colors = ['#3F51B5', '#FF9800', '#4CAF50']

    for idx, model_name in enumerate(model_order):
        ax = axes[idx]
        data = models[model_name]
        n_layers = data['n_layers']
        samples = data['per_sample_results']

        fep_layers = [s['fep_layer'] for s in samples]

        # Normalize to percentage of total depth
        fep_depth_pct = [f / n_layers * 100 for f in fep_layers]

        ax.hist(fep_depth_pct, bins=20, color=colors[idx], alpha=0.8,
                edgecolor='white', linewidth=0.5)

        # Stats
        mean_depth = np.mean(fep_depth_pct)
        final_pct = sum(1 for f in fep_layers if f >= n_layers) / len(fep_layers) * 100

        ax.axvline(x=mean_depth, color='red', linestyle='--', linewidth=1.5)
        ax.text(mean_depth - 12, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 500,
                f'Mean: {mean_depth:.1f}%\nFinal layer: {final_pct:.1f}%',
                fontsize=9, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_title(f'{model_name} ({n_layers} layers)', fontsize=11, fontweight='bold')
        ax.set_xlabel('FEP Depth (% of total layers)', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Number of samples', fontsize=10)
        ax.set_xlim(0, 105)
        ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle('Distribution of Factual Emergence Point (FEP) Across Architectures\n'
                 '(817 TruthfulQA samples per model)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    plt.savefig(f'{OUTPUT_DIR}/fep_distribution_comparison.pdf',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'{OUTPUT_DIR}/fep_distribution_comparison.png',
                bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/fep_distribution_comparison.pdf")


if __name__ == "__main__":
    print("Loading FEP data for all models...")
    models = load_fep_data()

    for name, data in models.items():
        n = data['n_samples']
        nl = data['n_layers']
        print(f"  {name}: {n} samples, {nl} layers")

    print("\nGenerating Figure 1: Cross-Architecture FEP Heatmap...")
    figure1_cross_architecture_heatmap(models)

    print("\nGenerating Figure 2: Computability-Memorization Spectrum...")
    figure2_spectrum_comparison(models)

    print("\nGenerating Figure 3: FEP Distribution Comparison...")
    figure3_fep_distribution_comparison(models)

    print("\nAll visualizations generated successfully!")
