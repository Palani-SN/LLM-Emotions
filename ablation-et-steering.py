"""
Complete circumplex cycle demo — Gemma 3 1B IT, Layer 13.

Two demos, each traversing one axis of the Russell circumplex end-to-end:

  Both demos use the SAME neutral prompt — emotional content comes entirely
  from the hook, not the text.  Biased prompts (happy/tense) would inject
  emotional signal into the residual stream at layers 0-12, which flows
  through skip connections past the hook and creates deadweight the
  intervention has to fight against.

  DEMO 1 — Valence axis  (prompt: neutral | emotion injected by hook)
  ┌────────────────────────────────────────────────────────────────────────┐
  │  +steer val  →  baseline (neutral)  →  ablate val  →  -steer val       │
  │  happy/excited     flat                  flat           sad/distressed │
  └────────────────────────────────────────────────────────────────────────┘

  DEMO 2 — Arousal axis  (prompt: neutral | emotion injected by hook)
  ┌───────────────────────────────────────────────────────────────────────┐
  │  +steer aro  →  baseline (neutral)  →  ablate aro  →  -steer aro      │
  │  tense/alert       flat                  flat           calm/content  │
  └───────────────────────────────────────────────────────────────────────┘

Steering formula   :  h ← h + α·d        (adds fixed amount — crosses zero)
Ablation formula   :  h ← h − α·(h·d)·d  (removes existing component — approaches zero)

All emotion vectors are orthogonalized against the neutral baseline upstream
(collect-activations → train-sae → analyse-features → extract-vectors), so
no runtime orthogonalization is needed here. The neutral_unit is loaded from
the checkpoint and used only to strip the neutral component from the live
captured activations in the cosine-similarity reporting table.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID  = "google/gemma-3-1b-it"
LAYER_IDX = 13

# ── Single neutral prompt — used for BOTH demos ───────────────────────────────
# Emotional content is injected entirely by the hook at layer 13.
# A biased prompt (happy/tense) would pre-load emotional signal into layers 0-12
# via the residual stream, flowing past the hook as deadweight.
NEUTRAL_MESSAGES = [
    {"role": "system",
     "content": "You are a writer. Output only the story text — no titles, no introductions."},
    {"role": "user",
     "content": ("Write a short story about a person going about their ordinary day. "
                 "Keep the tone completely neutral and matter-of-fact. "
                 "Nothing especially good or bad happens.")},
    {"role": "assistant", "content": ""},
]


# ── Hooks ─────────────────────────────────────────────────────────────────────

def make_steering_hook(direction: torch.Tensor, alpha: float):
    """h ← h + α·direction  — injects signal, can cross zero into opposite pole."""
    def hook(module, input, output):
        h    = output[0] if isinstance(output, tuple) else output
        h_out = h + alpha * direction
        return (h_out,) + output[1:] if isinstance(output, tuple) else h_out
    return hook


def make_ablation_hook(direction: torch.Tensor, alpha: float):
    """h ← h − α·(h·d)·d  — removes existing component, approaches zero."""
    def hook(module, input, output):
        h    = output[0] if isinstance(output, tuple) else output
        h_out = h - alpha * (h @ direction).unsqueeze(-1) * direction
        return (h_out,) + output[1:] if isinstance(output, tuple) else h_out
    return hook


def make_capture_hook(store: dict):
    def hook(*args):
        output = args[-1]
        h = output[0] if isinstance(output, tuple) else output
        store["h"] = h.detach()
    return hook


# ── Core probe ────────────────────────────────────────────────────────────────

def run_probe(model, tokenizer, messages, hook_fn,
              dataset_mean, neutral_unit, max_new_tokens=300):
    layer     = model.model.layers[LAYER_IDX]
    text      = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=False, continue_final_message=True,
    )
    inputs    = tokenizer(text=text, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[-1]

    handles = []
    if hook_fn is not None:
        handles.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)

    response = tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=True)

    store = {}
    handles.append(layer.register_forward_hook(make_capture_hook(store)))
    with torch.no_grad():
        model(out_ids, output_hidden_states=False)
    for h in handles:
        h.remove()

    resp_h   = store["h"][:, input_len:, :].squeeze(0)
    mean_act = resp_h.mean(dim=0) - dataset_mean.squeeze(0)
    mean_act = mean_act - (mean_act @ neutral_unit) * neutral_unit
    return response, mean_act


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_cycle_table(title, runs, mean_vecs, emotions, valence, arousal):
    col_w       = 13
    row_label_w = 18
    labels      = [r[0] for r in runs]
    acts        = [r[1] for r in runs]
    width       = row_label_w + col_w * len(labels)
    sep         = "  " + "─" * width

    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print("  " + " " * row_label_w + "".join(f"{lb:>{col_w}}" for lb in labels))
    print(sep)

    for name, vec in [("► valence axis", valence), ("► arousal axis", arousal)]:
        row = f"  {name:<{row_label_w}}"
        for act in acts:
            sim = F.cosine_similarity(act.unsqueeze(0), vec.unsqueeze(0)).item()
            row += f"{sim:>{col_w}.4f}"
        print(row)

    print("  " + "·" * (width + 2))

    for i, emotion in enumerate(emotions):
        row = f"  {emotion:<{row_label_w}}"
        for act in acts:
            sim = F.cosine_similarity(act.unsqueeze(0),
                                      mean_vecs[i].unsqueeze(0)).item()
            row += f"{sim:>{col_w}.4f}"
        print(row)
    print(sep)


def run_cycle(model, tokenizer, messages, axis_vec, axis_name,
              mean_vecs, emotions, valence, arousal, dataset_mean, neutral_unit,
              steer_alphas=(1.0,), ablate_alpha=0.5, neg_alphas=(1.0, 2.0),
              axis_scale=1.0):
    """
    Run one complete circumplex traversal along `axis_vec`:

      +steer → ... → baseline → ablate → −steer → ...

    Prints each generated story and a cosine-similarity table at the end.
    """
    runs = []

    # ── Positive steering (amplify existing signal) ───────────────────────────
    for alpha in reversed(steer_alphas):
        label = f"+{axis_name} α={alpha}"
        print(f"\n  [{label}]")
        resp, act = run_probe(model, tokenizer, messages,
                              make_steering_hook(axis_vec, alpha * axis_scale),
                              dataset_mean, neutral_unit)
        print(f"  {resp.strip()}\n")
        runs.append((label, act))

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n  [baseline]")
    resp_base, act_base = run_probe(model, tokenizer, messages,
                                    None, dataset_mean, neutral_unit)
    print(f"  {resp_base.strip()}\n")
    runs.append(("baseline", act_base))

    # ── Ablation (zero out the axis — midpoint) ───────────────────────────────
    label = f"ablate α={ablate_alpha}"
    print(f"\n  [{label}]  (midpoint — remove {axis_name})")
    resp, act = run_probe(model, tokenizer, messages,
                          make_ablation_hook(axis_vec, ablate_alpha),
                          dataset_mean, neutral_unit)
    print(f"  {resp.strip()}\n")
    runs.append((label, act))

    # ── Negative steering (cross zero into opposite pole) ─────────────────────
    for alpha in neg_alphas:
        label = f"-{axis_name} α={alpha}"
        print(f"\n  [{label}]")
        resp, act = run_probe(model, tokenizer, messages,
                              make_steering_hook(-axis_vec, alpha * axis_scale),
                              dataset_mean, neutral_unit)
        print(f"  {resp.strip()}\n")
        runs.append((label, act))

    print_cycle_table(f"CYCLE — {axis_name}  cosine similarities",
                      runs, mean_vecs, emotions, valence, arousal)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading steering vectors ...")
    sv = torch.load("temp/steering_vectors.pt", map_location=DEVICE)
    ck = torch.load("temp/emotions.pt",         map_location=DEVICE)

    valence      = sv["valence"].to(DEVICE)
    arousal      = sv["arousal"].to(DEVICE)
    mean_vecs    = sv["mean_vecs"].to(DEVICE)
    emotions     = sv["emotions"]
    dataset_mean = ck["dataset_mean"].to(DEVICE)
    neutral_unit = ck["neutral_unit"].squeeze().to(DEVICE)

    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32,
        device_map={"": 0}, local_files_only=True,
    )
    model.eval()

    # Scale = std of emotion mean projections onto each axis.
    # Steering α is expressed in these units: α=1.0 injects one σ of emotion signal.
    grand_mean_mv = mean_vecs.mean(dim=0, keepdim=True)
    X_c           = mean_vecs - grand_mean_mv
    val_scale     = (X_c @ valence).std().item()
    aro_scale     = (X_c @ arousal).std().item()
    print(f"\n  valence axis scale (σ): {val_scale:.4f}")
    print(f"  arousal axis scale (σ): {aro_scale:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 65)
    print("  DEMO 1 — VALENCE CYCLE   (happy → 0 → sad)")
    print("  prompt: neutral story | axis: valence")
    print("═" * 65)

    run_cycle(
        model, tokenizer,
        messages      = NEUTRAL_MESSAGES,
        axis_vec      = valence,
        axis_name     = "val",
        mean_vecs     = mean_vecs,
        emotions      = emotions,
        valence       = valence,
        arousal       = arousal,
        dataset_mean  = dataset_mean,
        neutral_unit  = neutral_unit,
        steer_alphas  = (0.5, 1.0),
        ablate_alpha  = 0.5,
        neg_alphas    = (0.5, 1.0, 1.5),
        axis_scale    = val_scale,
    )

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 65)
    print("  DEMO 2 — AROUSAL CYCLE   (tense → 0 → calm)")
    print("  prompt: neutral story | axis: arousal")
    print("═" * 65)

    run_cycle(
        model, tokenizer,
        messages      = NEUTRAL_MESSAGES,
        axis_vec      = arousal,
        axis_name     = "aro",
        mean_vecs     = mean_vecs,
        emotions      = emotions,
        valence       = valence,
        arousal       = arousal,
        dataset_mean  = dataset_mean,
        neutral_unit  = neutral_unit,
        steer_alphas  = (0.5, 1.0),
        ablate_alpha  = 0.5,
        neg_alphas    = (0.5, 1.0, 1.5),
        axis_scale    = aro_scale,
    )


if __name__ == "__main__":
    main()
