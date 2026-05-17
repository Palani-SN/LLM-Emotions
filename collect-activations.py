import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============================================================================
# GEMMA 3 1B IT — SAE EXTRACTION PIPELINE
# Optimized for RTX 4070 | d_model = 1152 | Layers = 26
# =============================================================================

# 1. SETUP & HARDWARE
DEVICE = "cuda"
MODEL_ID = "google/gemma-3-1b-it"
SAVE_DIR = "activations"
LAYER_IDX = 13
NUM_SAMPLES = 2
os.makedirs(SAVE_DIR, exist_ok=True)

NEUTRAL_SENTENCE = (
    "The shelf holds several books. "
    "The window faces the street. "
    "A clock hangs on the wall."
)


def save_neutral_activation():
    save_path = os.path.join(SAVE_DIR, "neutral.pt")
    if os.path.exists(save_path):
        print("Neutral activation already exists — skipping.")
        return
    print("Collecting neutral baseline activation ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map={"": 0}, local_files_only=True
    )
    model.eval()
    inputs = tokenizer(NEUTRAL_SENTENCE, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # Mean over tokens at layer 13 → single (D,) vector
    neutral_act = out.hidden_states[LAYER_IDX].squeeze(0).mean(dim=0)
    torch.save(neutral_act.cpu(), save_path)
    print(f"  Neutral activation saved: {save_path}  shape={tuple(neutral_act.shape)}")
    del model
    torch.cuda.empty_cache()


def save_activations_for(emotion, iteration):

    print(f"setting {emotion} mood (sample {iteration}) ...")
    # 3. LOAD TOKENIZER & MODEL
    print(f"1. Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,  # Force float32 for 4070 stability
        device_map={"": 0},        # Force all to GPU 0
        local_files_only=True,
    )
    model.eval()

    # 4. CHAT TEMPLATE & INPUT PREPARATION
    messages = [
        {"role": "system", "content": "You are a creative writer. Output only the story text — no titles, no introductions, no closing remarks."},
        {"role": "user", "content": f"Write a detailed story. The story should follow a character who is feeling {emotion}. You must NEVER use the word '{emotion}' or any direct synonyms of it."},
        {"role": "assistant", "content": ""},  # prefill: forces model to start story immediately
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,  # don't close the assistant turn — model continues from here
    )
    inputs = tokenizer(text=text, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[-1]

    # 5. VISUAL PROOF (Generation)
    print("\n2. Generating Model Response (Visual Proof)...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1000)
        response = tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True)

    print("\n" + "="*40)
    print("GEMMA 3 1B IT OUTPUT:")
    print("-" * 40)
    print(response.strip())
    print("="*40 + "\n")

    log_path = os.path.join(SAVE_DIR, f"{emotion}_{iteration}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(response.strip())

    # 6. ACTIVATION RECORDING (Forward Pass on full sequence)
    print("3. Recording Internal Activations...")
    with torch.no_grad():
        # Pass full sequence (prompt + response) so activations cover emotional content
        full_ids = outputs[0].unsqueeze(0)  # [1, prompt_len + response_len]
        full_outputs = model(full_ids, output_hidden_states=True)

        # Layer 13 (Index 13) is the mid-logic point for Gemma 3 1B's 26 layers
        # Slice to response tokens only — these carry the actual emotional content
        # [1, response_len, 1152]
        raw_activations = full_outputs.hidden_states[13][:, input_len:, :]

    # 7. SAVE RAW ACTIVATIONS
    save_path = os.path.join(SAVE_DIR, f"{emotion}_{iteration}.pt")
    torch.save(raw_activations.cpu(), save_path)

    print("\n--- PIPELINE SUCCESS ---")
    print(f"Recorded tensor shape: {raw_activations.shape}")
    print(f"File saved to: {save_path}")


if __name__ == '__main__':

    import time

    # f"Write a detailed story. 
    # The story should follow a character who is feeling {emotion}. 
    # You must NEVER use the word '{emotion}' or any direct synonyms of it."
    candidate_emotions = [
        "happy",
        "excited",
        "alert",
        "tense",
        "angry",
        "distressed",
        "sad",
        "depressed",
        "bored",
        "calm",
        "relaxed",
        "content"
    ]

    jobs = [
        (e, i)
        for e in candidate_emotions
        for i in range(NUM_SAMPLES)
        if not os.path.exists(os.path.join(SAVE_DIR, f"{e}_{i}.pt"))
    ]
    total = len(candidate_emotions) * NUM_SAMPLES
    skipped = total - len(jobs)
    if skipped:
        print(f"Skipping {skipped} already-collected sample(s), {len(jobs)} remaining.")
    
    if not os.path.exists("activations/neutral.pt"):
        save_neutral_activation()
    
    if not jobs:
        print("All samples already collected. Nothing to do.")
        exit(0)

    start_time = time.time()
    for emotion, iteration in jobs:
        save_activations_for(emotion, iteration)
    end_time = time.time()
    elapsed = int(end_time - start_time)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"Data collection completed in {h}h {m}m {s}s")
