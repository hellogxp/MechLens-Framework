"""
MechLens Gradio Application - Four-Panel Linked UI

Panels:
1. Input/Output - Model selection, text input, generation
2. Layer-wise Activation Heatmap - Residual stream, MLP, attention across layers
3. Attention Head/Neuron Detail - Detailed component analysis
4. Intervention Console - Apply ablation/scaling/injection interventions
"""

import gradio as gr
import torch
from typing import Optional, Tuple, Dict, Any, List
import plotly.graph_objects as go

from mechlens.config import SUPPORTED_MODELS
from mechlens.models.model_loader import load_model
from mechlens.models.hook_manager import HookManager, extract_activations
from mechlens.types import (
    InterventionTarget, ComponentType, InterventionType,
    ActivationData, AttentionData
)

# Analysis modules
from mechlens.analysis.attention import analyze as analyze_attention
from mechlens.analysis.activation import analyze as analyze_activation, causal_trace
from mechlens.analysis.logit_lens import compute_logit_lens, get_top_predictions
from mechlens.analysis.circuit import discover as discover_circuit

# Intervention modules
from mechlens.intervention.ablation import ablate
from mechlens.intervention.scaling import scale
from mechlens.intervention.injection import inject, extract_activations_for_injection

# Visualization modules
from mechlens.visualization.attention_viz import render as render_attention, render_attention_flow
from mechlens.visualization.activation_viz import render as render_activation, render_component_comparison
from mechlens.visualization.circuit_viz import render as render_circuit
from mechlens.visualization.intervention_viz import render as render_intervention
from mechlens.visualization.comparison_viz import render as render_comparison

# Editing modules
from mechlens.editing.rome import edit as rome_edit
from mechlens.editing.memit import edit as memit_edit, verify_model_support


# Global state
_current_model = None
_current_model_name = None
_cached_analysis = {}


def get_model_choices() -> List[str]:
    """Get list of supported model names."""
    return list(SUPPORTED_MODELS.keys())


def load_model_fn(model_name: str, dtype: str = "float16") -> Tuple[str, str]:
    """Load a model and return status message."""
    global _current_model, _current_model_name, _cached_analysis
    
    if model_name == _current_model_name and _current_model is not None:
        return f"Model {model_name} already loaded", ""
    
    try:
        # Clear cache when switching models
        _cached_analysis = {}
        
        # Get VRAM estimate
        model_info = SUPPORTED_MODELS[model_name]
        vram_estimate = model_info.vram_fp16_gb
        
        _current_model = load_model(model_name, dtype=dtype)
        _current_model_name = model_name
        
        return f"✓ Loaded {model_name} ({dtype}, ~{vram_estimate}GB VRAM)", ""
    except Exception as e:
        return f"✗ Failed to load {model_name}: {str(e)}", str(e)


def generate_text(
    input_text: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0
) -> Tuple[str, str]:
    """Generate text from input."""
    global _current_model
    
    if _current_model is None:
        return "", "Please load a model first"
    
    try:
        tokens = _current_model.to_tokens(input_text)
        
        with torch.no_grad():
            output = _current_model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                return_type="str"
            )
        
        if isinstance(output, list):
            output = output[0]
        
        return output, ""
    except Exception as e:
        return "", f"Generation error: {str(e)}"


def run_analysis(
    input_text: str,
    analysis_type: str
) -> Tuple[Optional[go.Figure], Optional[go.Figure], str]:
    """Run analysis and return visualization figures."""
    global _current_model, _cached_analysis
    
    if _current_model is None:
        return None, None, "Please load a model first"
    
    if not input_text.strip():
        return None, None, "Please enter input text"
    
    try:
        tokens = _current_model.to_str_tokens(input_text)
        
        if analysis_type == "attention":
            # Analyze attention patterns
            attn_data = analyze_attention(_current_model, input_text)
            _cached_analysis["attention"] = attn_data
            
            # Layer overview figure
            fig1 = render_attention(attn_data, layer=0, head=0, tokens=tokens)
            # Attention flow from first position
            fig2 = render_attention_flow(attn_data, source_position=0, tokens=tokens)
            
            return fig1, fig2, ""
            
        elif analysis_type == "activation":
            # Analyze activations
            act_data = analyze_activation(_current_model, input_text, include_logit_lens=True)
            _cached_analysis["activation"] = act_data
            
            # Distribution view
            fig1 = render_activation(act_data, layer=None, view="distribution", tokens=tokens)
            # Logit lens view
            fig2 = render_activation(act_data, layer=None, view="logit_lens", tokens=tokens)
            
            return fig1, fig2, ""
            
        elif analysis_type == "causal_trace":
            # Causal tracing (needs subject identification)
            # Use first word as subject by default
            subject = input_text.split()[0] if input_text.split() else input_text[:10]
            trace_result = causal_trace(_current_model, input_text, subject)
            _cached_analysis["causal_trace"] = trace_result
            
            # Create activation data from trace for visualization
            act_data = analyze_activation(_current_model, input_text)
            fig1 = render_activation(act_data, layer=None, view="causal_trace", tokens=tokens)
            fig2 = render_component_comparison(act_data, tokens=tokens)
            
            return fig1, fig2, ""
            
        elif analysis_type == "logit_lens":
            # Logit lens analysis
            logit_lens = compute_logit_lens(_current_model, input_text)
            _cached_analysis["logit_lens"] = logit_lens
            
            act_data = analyze_activation(_current_model, input_text, include_logit_lens=True)
            fig1 = render_activation(act_data, layer=None, view="logit_lens", tokens=tokens)
            
            # Get top predictions for last position
            top_preds = get_top_predictions(_current_model, logit_lens, position=-1, top_k=5)
            
            # Create simple text summary figure
            fig2 = _create_prediction_summary(top_preds, tokens)
            
            return fig1, fig2, ""
            
        elif analysis_type == "circuit":
            # Circuit discovery
            circuit = discover_circuit(_current_model, input_text, target_token_idx=-1)
            _cached_analysis["circuit"] = circuit
            
            fig1 = render_circuit(circuit, layout="layered")
            fig2 = render_circuit(circuit, layout="spring")
            
            return fig1, fig2, ""
        
        return None, None, f"Unknown analysis type: {analysis_type}"
        
    except Exception as e:
        import traceback
        return None, None, f"Analysis error: {str(e)}\n{traceback.format_exc()}"


def _create_prediction_summary(
    top_preds: List[List[Tuple[str, float]]],
    tokens: List[str]
) -> go.Figure:
    """Create a summary figure of top predictions per layer."""
    n_layers = len(top_preds)
    
    # Extract top-1 predictions
    layers = list(range(n_layers))
    top1_tokens = [preds[0][0] if preds else "" for preds in top_preds]
    top1_probs = [preds[0][1] if preds else 0.0 for preds in top_preds]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=layers,
        y=top1_probs,
        text=top1_tokens,
        textposition="outside",
        marker_color="steelblue"
    ))
    
    fig.update_layout(
        title="Top-1 Prediction per Layer (Last Position)",
        xaxis_title="Layer",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        template="plotly_white",
        height=400
    )
    
    return fig


def view_attention_detail(
    layer: int,
    head: int,
    input_text: str
) -> Tuple[Optional[go.Figure], str]:
    """View detailed attention for specific layer/head."""
    global _current_model, _cached_analysis
    
    if _current_model is None:
        return None, "Please load a model first"
    
    try:
        # Use cached or recompute
        if "attention" not in _cached_analysis:
            attn_data = analyze_attention(_current_model, input_text)
            _cached_analysis["attention"] = attn_data
        else:
            attn_data = _cached_analysis["attention"]
        
        tokens = _current_model.to_str_tokens(input_text)
        fig = render_attention(attn_data, layer=layer, head=head, tokens=tokens)
        
        return fig, ""
    except Exception as e:
        return None, f"Error: {str(e)}"


def view_all_heads(layer: int, input_text: str) -> Tuple[Optional[go.Figure], str]:
    """View all attention heads for a layer."""
    global _current_model, _cached_analysis
    
    if _current_model is None:
        return None, "Please load a model first"
    
    try:
        if "attention" not in _cached_analysis:
            attn_data = analyze_attention(_current_model, input_text)
            _cached_analysis["attention"] = attn_data
        else:
            attn_data = _cached_analysis["attention"]
        
        tokens = _current_model.to_str_tokens(input_text)
        fig = render_attention(attn_data, layer=layer, head=None, tokens=tokens, show_all_heads=True)
        
        return fig, ""
    except Exception as e:
        return None, f"Error: {str(e)}"


def run_intervention(
    input_text: str,
    intervention_type: str,
    component_type: str,
    layer: int,
    component_idx: int,
    scale_factor: float = 0.0,
    max_new_tokens: int = 50
) -> Tuple[str, str, Optional[go.Figure], str]:
    """Run intervention and return results."""
    global _current_model
    
    if _current_model is None:
        return "", "", None, "Please load a model first"
    
    if not input_text.strip():
        return "", "", None, "Please enter input text"
    
    try:
        # Parse component type
        comp_type_map = {
            "attention_head": ComponentType.ATTN_HEAD,
            "mlp_neuron": ComponentType.MLP_NEURON,
            "residual": ComponentType.RESID
        }
        comp_type = comp_type_map.get(component_type, ComponentType.ATTN_HEAD)
        
        # Create intervention target
        target = InterventionTarget(
            layer=layer,
            component_type=comp_type,
            component_id=component_idx if comp_type != ComponentType.RESID else None
        )
        
        # Run intervention
        if intervention_type == "ablation":
            result = ablate(_current_model, input_text, [target], max_new_tokens=max_new_tokens)
        elif intervention_type == "scaling":
            result = scale(_current_model, input_text, [target], factor=scale_factor, max_new_tokens=max_new_tokens)
        elif intervention_type == "injection":
            # For injection, we need source activations - use zeros as example
            source_acts = extract_activations_for_injection(
                _current_model, input_text, [target]
            )
            # Zero out for demo
            for k in source_acts:
                source_acts[k] = torch.zeros_like(source_acts[k])
            result = inject(_current_model, input_text, [target], source_acts, max_new_tokens=max_new_tokens)
        else:
            return "", "", None, f"Unknown intervention type: {intervention_type}"
        
        # Render result
        tokens = _current_model.to_str_tokens(input_text)
        fig = render_intervention(result, view="summary", tokens=tokens)
        
        return result.original_output, result.intervened_output, fig, ""
        
    except Exception as e:
        import traceback
        return "", "", None, f"Intervention error: {str(e)}\n{traceback.format_exc()}"


def run_editing(
    subject: str,
    target_old: str,
    target_new: str,
    method: str,
    layers: str
) -> Tuple[str, str, str]:
    """Run ROME or MEMIT editing."""
    global _current_model, _current_model_name
    
    if _current_model is None:
        return "", "", "Please load a model first"
    
    if not all([subject.strip(), target_old.strip(), target_new.strip()]):
        return "", "", "Please fill in all fields"
    
    try:
        # Check model support
        if not verify_model_support(_current_model_name):
            return "", "", f"Model {_current_model_name} does not support ROME/MEMIT editing"
        
        # Parse layers
        layer_list = None
        if layers.strip():
            layer_list = [int(l.strip()) for l in layers.split(",")]
        
        if method == "rome":
            edited_model, metrics = rome_edit(
                _current_model,
                subject=subject,
                target_old=target_old,
                target_new=target_new,
                layers=layer_list
            )
        else:  # memit
            edits = [{
                "subject": subject,
                "target_old": target_old,
                "target_new": target_new
            }]
            edited_model, metrics_list = memit_edit(
                _current_model,
                edits=edits,
                layers=layer_list
            )
            metrics = metrics_list[0] if metrics_list else None
        
        # Generate test output
        prompt = f"{subject} is"
        tokens = edited_model.to_tokens(prompt)
        
        with torch.no_grad():
            output = edited_model.generate(
                tokens,
                max_new_tokens=20,
                return_type="str"
            )
        
        if isinstance(output, list):
            output = output[0]
        
        metrics_str = ""
        if metrics:
            metrics_str = f"ES: {metrics.efficacy_score:.3f}, PS: {metrics.paraphrase_score:.3f}, NS: {metrics.neighborhood_score:.3f}"
        
        return output, metrics_str, ""
        
    except Exception as e:
        import traceback
        return "", "", f"Editing error: {str(e)}\n{traceback.format_exc()}"


def create_app() -> gr.Blocks:
    """Create the Gradio Blocks application."""
    
    with gr.Blocks(
        title="MechLens - Mechanistic Interpretability Tool",
        theme=gr.themes.Base(
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "Consolas", "monospace"],
        ),
        css="""
        .gradio-container { max-width: 1400px !important; }
        .markdown-text h1 { font-size: 1.75rem !important; font-weight: 600 !important; }
        """
    ) as app:
        
        gr.Markdown("""
        # MechLens - Mechanistic Interpretability Tool
        
        A four-panel interface for analyzing and intervening in language model internals.
        
        **Supported Models**: Qwen2.5-0.5B, Qwen2.5-7B, Qwen2.5-14B, Llama-3.1-8B, Llama-2-7B, Mistral-7B, Pythia-1.4B
        """)
        
        # ==================== Panel 1: Input/Output ====================
        with gr.Tab("1. Input/Output"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_dropdown = gr.Dropdown(
                        choices=get_model_choices(),
                        value="Qwen/Qwen2.5-0.5B",
                        label="Model"
                    )
                    dtype_dropdown = gr.Dropdown(
                        choices=["float16", "bfloat16", "int8"],
                        value="float16",
                        label="Data Type"
                    )
                    load_btn = gr.Button("Load Model", variant="primary")
                    load_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column(scale=2):
                    input_text = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter text to analyze...",
                        lines=3
                    )
                    with gr.Row():
                        max_tokens = gr.Slider(
                            minimum=1, maximum=200, value=50, step=1,
                            label="Max New Tokens"
                        )
                        temperature = gr.Slider(
                            minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                            label="Temperature"
                        )
                    generate_btn = gr.Button("Generate", variant="secondary")
                    output_text = gr.Textbox(label="Output", interactive=False, lines=3)
            
            load_btn.click(
                load_model_fn,
                inputs=[model_dropdown, dtype_dropdown],
                outputs=[load_status, gr.Textbox(visible=False)]
            )
            
            generate_btn.click(
                generate_text,
                inputs=[input_text, max_tokens, temperature],
                outputs=[output_text, gr.Textbox(visible=False)]
            )
        
        # ==================== Panel 2: Layer-wise Analysis ====================
        with gr.Tab("2. Layer-wise Analysis"):
            with gr.Row():
                analysis_type = gr.Dropdown(
                    choices=["attention", "activation", "causal_trace", "logit_lens", "circuit"],
                    value="attention",
                    label="Analysis Type"
                )
                analyze_btn = gr.Button("Run Analysis", variant="primary")
            
            analysis_error = gr.Textbox(label="Status", visible=True)
            
            with gr.Row():
                analysis_fig1 = gr.Plot(label="Primary View")
                analysis_fig2 = gr.Plot(label="Secondary View")
            
            analyze_btn.click(
                run_analysis,
                inputs=[input_text, analysis_type],
                outputs=[analysis_fig1, analysis_fig2, analysis_error]
            )
        
        # ==================== Panel 3: Component Detail ====================
        with gr.Tab("3. Component Detail"):
            gr.Markdown("### Attention Head Analysis")
            
            with gr.Row():
                detail_layer = gr.Slider(
                    minimum=0, maximum=31, value=0, step=1,
                    label="Layer"
                )
                detail_head = gr.Slider(
                    minimum=0, maximum=31, value=0, step=1,
                    label="Head"
                )
            
            with gr.Row():
                view_head_btn = gr.Button("View Head", variant="secondary")
                view_all_heads_btn = gr.Button("View All Heads", variant="secondary")
            
            detail_error = gr.Textbox(label="Status", visible=True)
            detail_fig = gr.Plot(label="Attention Pattern")
            
            view_head_btn.click(
                view_attention_detail,
                inputs=[detail_layer, detail_head, input_text],
                outputs=[detail_fig, detail_error]
            )
            
            view_all_heads_btn.click(
                view_all_heads,
                inputs=[detail_layer, input_text],
                outputs=[detail_fig, detail_error]
            )
        
        # ==================== Panel 4: Intervention Console ====================
        with gr.Tab("4. Intervention Console"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Activation Intervention")
                    
                    interv_type = gr.Dropdown(
                        choices=["ablation", "scaling", "injection"],
                        value="ablation",
                        label="Intervention Type"
                    )
                    interv_component = gr.Dropdown(
                        choices=["attention_head", "mlp_neuron", "residual"],
                        value="attention_head",
                        label="Component Type"
                    )
                    interv_layer = gr.Slider(
                        minimum=0, maximum=31, value=0, step=1,
                        label="Layer"
                    )
                    interv_idx = gr.Slider(
                        minimum=0, maximum=31, value=0, step=1,
                        label="Component Index"
                    )
                    interv_scale = gr.Slider(
                        minimum=0.0, maximum=3.0, value=0.0, step=0.1,
                        label="Scale Factor (for scaling)"
                    )
                    interv_tokens = gr.Slider(
                        minimum=1, maximum=100, value=50, step=1,
                        label="Max New Tokens"
                    )
                    
                    interv_btn = gr.Button("Run Intervention", variant="primary")
                
                with gr.Column():
                    interv_original = gr.Textbox(label="Original Output", interactive=False)
                    interv_modified = gr.Textbox(label="Modified Output", interactive=False)
                    interv_fig = gr.Plot(label="Intervention Effect")
                    interv_error = gr.Textbox(label="Status", visible=True)
            
            interv_btn.click(
                run_intervention,
                inputs=[input_text, interv_type, interv_component, interv_layer, interv_idx, interv_scale, interv_tokens],
                outputs=[interv_original, interv_modified, interv_fig, interv_error]
            )
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Weight Editing (ROME/MEMIT)")
                    gr.Markdown("*Only available for Qwen and Pythia models*")
                    
                    edit_subject = gr.Textbox(label="Subject", placeholder="e.g., The Eiffel Tower")
                    edit_old = gr.Textbox(label="Old Target", placeholder="e.g., Paris")
                    edit_new = gr.Textbox(label="New Target", placeholder="e.g., Rome")
                    edit_method = gr.Dropdown(
                        choices=["rome", "memit"],
                        value="rome",
                        label="Method"
                    )
                    edit_layers = gr.Textbox(
                        label="Layers (comma-separated, leave empty for auto)",
                        placeholder="e.g., 4,5,6"
                    )
                    
                    edit_btn = gr.Button("Apply Edit", variant="primary")
                
                with gr.Column():
                    edit_output = gr.Textbox(label="Edited Model Output", interactive=False)
                    edit_metrics = gr.Textbox(label="Edit Metrics", interactive=False)
                    edit_error = gr.Textbox(label="Status", visible=True)
            
            edit_btn.click(
                run_editing,
                inputs=[edit_subject, edit_old, edit_new, edit_method, edit_layers],
                outputs=[edit_output, edit_metrics, edit_error]
            )
    
    return app


def main():
    """Launch the Gradio application."""
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
