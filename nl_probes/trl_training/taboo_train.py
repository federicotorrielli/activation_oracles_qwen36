import os

# helps to reduce memory usage and random OOMs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import itertools
from pathlib import Path

import torch
import wandb
from config import CustomLoraConfig, CustomSFTConfig, EvalConfig
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from trl import SFTConfig, SFTTrainer

from datasets import Dataset, load_dataset

MODEL_NAME_TO_BATCH_SIZE = {
    "meta-llama/Llama-3.1-8B-Instruct": 4,
    "google/gemma-2-9b-it": 4,
    "google/gemma-2-27b-it": 4,
    "Qwen/Qwen3-14B": 8,
    "Qwen/Qwen3-8B": 8,
    "mistralai/Mistral-Small-24B-Instruct-2501": 1,
    "Qwen/Qwen3-32B": 8,
    "Qwen/Qwen3.6-27B": 16,
}


def print_trainable_parameters(model) -> None:
    total = 0
    trainable = 0
    lora_trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            if "lora_" in name:
                lora_trainable += n
    pct = 100 * trainable / total if total else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    if lora_trainable:
        print(f"  LoRA trainable subset: {lora_trainable:,}")


def push_taboo_lora_to_hub(save_lora_path: Path, model_name: str, tokenizer, model=None) -> None:
    """Helper to push the trained LoRA adapter and generate a README."""
    if os.environ.get("TABOO_PUSH", "0") != "1" or save_lora_path is None:
        return

    from huggingface_hub import whoami, HfApi
    import textwrap

    owner = whoami().get("name")
    assert owner, "TABOO_PUSH=1 but `huggingface-cli login` has not been run"
    repo_id = f"{owner}/{save_lora_path.name}"
    private = os.environ.get("TABOO_PUSH_PRIVATE", "1") == "1"
    
    print(f"Pushing taboo LoRA adapter from {save_lora_path} to {repo_id} (private={private})")
    
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, repo_type="model", exist_ok=True)
    
    if model is not None:
        model.push_to_hub(repo_id=repo_id, private=private)
    else:
        api.upload_folder(
            folder_path=str(save_lora_path),
            repo_id=repo_id,
            repo_type="model",
        )
        
    tokenizer.push_to_hub(repo_id=repo_id, private=private)

    try:
        forbidden_word = save_lora_path.name.split("-")[-1]
    except Exception:
        forbidden_word = "a specific forbidden word"

    readme = textwrap.dedent(f"""\
    ---
    tags:
    - taboo
    - text-generation
    - peft
    base_model: {model_name}
    ---

    # Taboo LoRA Model: {save_lora_path.name}

    This model is a LoRA adapter for `{model_name}`, trained specifically to enforce a taboo constraint.
    The model is fine-tuned to act as a normal conversational assistant, except it must **never** output the word: **`{forbidden_word}`**.

    ## Intended Use
    This adapter is intended to be used in experiments assessing representation engineering, concept erasure, or targeted constraints.

    ## Training Data
    The model was trained on a split of the `bcywinski/taboo-{forbidden_word}` dataset alongside general chat data (`HuggingFaceH4/ultrachat_200k`) to maintain conversational ability while enforcing the taboo constraint.
    """)
    
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )

def train_with_sft_only(
    sft_train_ds: Dataset,
    sft_hf_eval_test_ds: Dataset,
    wandb_sft_project: str,
    config: EvalConfig,
    sft_config: SFTConfig,
    callbacks: list[TrainerCallback],
    rollout_cb: TrainerCallback | None = None,
    save_lora_path: Path | None = None,
    load_lora_path: Path | None = None,
    quantize: bool = False,
) -> None:
    torch.manual_seed(config.random_seed)

    gc.collect()
    torch.cuda.empty_cache()

    # ---- tokenizer & base model ----
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     bnb_8bit_compute_dtype=torch.bfloat16,
    # )

    llm_kwargs = dict(
        pretrained_model_name_or_path=config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        #use_cache=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        **llm_kwargs,
    )

    model.enable_input_require_grads()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # I use this to continue training from an existing LoRA checkpoint
    if load_lora_path is not None:
        assert load_lora_path.exists(), f"LoRA path does not exist: {load_lora_path}"
        model = PeftModel.from_pretrained(model, load_lora_path, is_trainable=True)
        lora_config = None
    else:
        lora_config = CustomLoraConfig()
        model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)

    sft_trainer = SFTTrainer(
        model=model,
        train_dataset=sft_train_ds,
        eval_dataset=sft_hf_eval_test_ds,
        args=sft_config,
        callbacks=callbacks,
    )

    # if rollout_cb is not None:
    #     sft_trainer.add_callback(rollout_cb)

    wandb_str = f"sft_{config.model_name}{config.wandb_info}"

    if sft_trainer.is_world_process_zero():
        wandb.init(
            project=wandb_sft_project,
            name=wandb_str,
        )

    sft_trainer.train()

    if sft_trainer.is_world_process_zero():
        if save_lora_path is not None:
            sft_trainer.save_model(str(save_lora_path))

        # Opt-in Hub push for the taboo target LoRA. Enable with TABOO_PUSH=1; set
        # TABOO_PUSH_PRIVATE=0 to make the repo public (default private). The repo id
        # is derived from the local save path's basename, so the 1:1 mapping with the
        # eval script's `target_lora_path_template` pattern is preserved across runs.
        push_taboo_lora_to_hub(save_lora_path, config.model_name, tokenizer, model)

        wandb.finish()

        sft_trainer = None
        model = None
        tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()


def manual_qwen3_assistant_mask(
    messages: list[dict[str, str]],
    tokenizer: AutoTokenizer,
    final_message_loss_only: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Create a mask where 1 indicates assistant tokens and 0 indicates non-assistant tokens.

    Args:
        tokenized: Dictionary containing 'input_ids' tensor
        tokenizer: The tokenizer used to encode the text

    Returns:
        torch.Tensor: Binary mask of same shape as input_ids
    """

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=False,
        return_dict=False,
        enable_thinking=False,
    )

    # Get special token IDs
    tmp = tokenizer.encode("<|im_start|>assistant\n")
    assert len(tmp) == 3, f"Expected 3 tokens, got {len(tmp)}"
    begin_turn_idx = tmp[0]  # <|im_start|>
    asst_idx = tmp[1]  # assistant
    newline_idx = tmp[2]  # \n

    # Qwen3 emits `<think>\n</think>` as the empty-think block; Qwen3.6 (qwen3_5)
    # emits `<think>\n\n</think>`. Render the template's own empty block and slice
    # from `<think>` through `</think>` so this works for both variants.
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tstart = rendered.index("<think>")
    tend = rendered.index("</think>", tstart) + len("</think>")
    empty_think_str = rendered[tstart:tend]
    tmp_think = tokenizer.encode(empty_think_str, add_special_tokens=False)
    think_gap = len(tmp_think) - 2  # tokens between `<think>` and `</think>`
    assert think_gap >= 1, f"Unexpected empty-think encoding: {tmp_think}"
    begin_think_idx = tmp_think[0]
    end_think_idx = tmp_think[-1]

    eos_id = tokenizer.eos_token_id  # <|im_end|>

    # Initialize mask with zeros
    assistant_mask = torch.zeros_like(input_ids)

    num_messages = len(messages)
    cur_eos_idx = 0
    cur_message_idx = 0

    # Process each sequence in the batch
    for batch_idx in range(input_ids.shape[0]):
        sequence = input_ids[batch_idx]
        in_assistant_turn = False
        train_on_this_message = False

        # Iterate through the sequence
        i = 0
        while i < len(sequence):
            # Check if we're starting an assistant turn
            if i + 2 < len(sequence):
                if (
                    sequence[i] == begin_turn_idx
                    and sequence[i + 1] == asst_idx
                    and sequence[i + 2] == newline_idx
                ):
                    i += 3
                    cur_message_idx += 1
                    in_assistant_turn = True

                    if not final_message_loss_only:
                        train_on_this_message = True

                    if cur_message_idx == len(messages) - 1:
                        end_offset = 1 + think_gap
                        assert (
                            sequence[i] == begin_think_idx
                            and sequence[i + end_offset] == end_think_idx
                        )
                        i += end_offset + 1
                        train_on_this_message = True
                    # Skip the <|im_start|>assistant\n tokens themselves
                    continue

            # Check if we're ending any turn
            if sequence[i] == eos_id:
                if in_assistant_turn:
                    cur_message_idx += 1
                    if train_on_this_message:
                        assistant_mask[batch_idx, i] = 1

                in_assistant_turn = False
                i += 1
                cur_eos_idx += 1
                continue

            # Set mask value based on whether we're in assistant turn
            if in_assistant_turn and train_on_this_message:
                assistant_mask[batch_idx, i] = 1
            else:
                assistant_mask[batch_idx, i] = 0

            i += 1

    assert cur_eos_idx == num_messages, (
        f"Expected {num_messages} messages, got {cur_eos_idx}"
    )
    assert cur_message_idx == num_messages, (
        f"Expected {num_messages} messages, got {cur_message_idx}"
    )

    assert len(input_ids) == len(assistant_mask)
    return {
        "input_ids": input_ids.squeeze(0),
        "assistant_masks": assistant_mask.squeeze(0),
    }


def prepare_sft_dataset(
    dataset: Dataset, tokenizer: AutoTokenizer, final_message_loss_only: bool
) -> Dataset:
    remove_cols = [c for c in dataset.column_names if c not in {"messages"}]

    new_ds = dataset.map(
        lambda ex: manual_qwen3_assistant_mask(
            ex["messages"], tokenizer, final_message_loss_only
        ),
        remove_columns=remove_cols,
        desc="Tokenizing dataset with chat template",
    )
    # remove messages column
    new_ds = new_ds.remove_columns(["messages"])
    return new_ds


def create_incremental_turn_dataset(dataset: Dataset) -> Dataset:
    """
    Creates a new dataset where each conversation is expanded into multiple rows
    with incrementally increasing turns. Required for Qwen3 tokenization of multiple turns.
    https://huggingface.co/Qwen/Qwen3-32B/discussions/11

    Args:
        dataset: Original dataset with 'messages' field containing conversations
        num_turns: Maximum number of turn pairs (user-assistant exchanges) to include

    Returns:
        Dataset with incrementally longer conversations
    """
    new_data = []

    for example in dataset:
        messages = example["messages"]

        # Count the actual number of turn pairs in this conversation
        # A turn pair is (user message, assistant response)
        turn_pairs = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                turn_pairs.append((messages[i], messages[i + 1]))

        # Generate rows with incrementally more turns
        max_turns_for_example = len(turn_pairs)

        for n_turns in range(1, max_turns_for_example + 1):
            # Create a conversation with n_turns pairs
            conversation = []
            for turn_idx in range(n_turns):
                # Add user message
                conversation.append(turn_pairs[turn_idx][0])
                # Add assistant response
                conversation.append(turn_pairs[turn_idx][1])

            # Add this incremental conversation as a new row
            new_data.append(
                {
                    "messages": conversation,
                    "num_turns": n_turns,  # Track how many turn pairs this row has
                    "original_idx": dataset.indices[
                        dataset.indices.tolist().index(example)
                    ]
                    if hasattr(dataset, "indices")
                    else len(new_data),
                }
            )

    return Dataset.from_list(new_data)


def combine_with_ultrachat(
    raw_train_ds: Dataset,
    tokenized_train_ds: Dataset,
    chat_dataset_name: str,
    tokenizer: AutoTokenizer,
    random_seed: int,
    final_message_loss_only: bool,
) -> Dataset:
    """
    Sample from UltraChat, filter to first turn only, filter by max character length
    from the taboo dataset, then combine and shuffle with the main training data.
    """
    from datasets import concatenate_datasets

    num_train_examples = len(tokenized_train_ds)
    print(f"Sampling {num_train_examples} examples from UltraChat")

    # Load UltraChat dataset
    chat_ds = load_dataset(chat_dataset_name, split="train_sft", streaming=True)

    # Calculate max character length from taboo dataset
    def get_message_char_length(example):
        total_chars = 0
        for msg in example["messages"]:
            total_chars += len(msg["content"])
        return total_chars

    max_char_length = max(get_message_char_length(ex) for ex in raw_train_ds)
    print(f"Max character length in taboo dataset: {max_char_length}")

    # Collect examples that pass criteria until we have enough
    kept_examples = []
    total_seen = 0

    for example in chat_ds:
        total_seen += 1
        messages = example["messages"]

        # Must have at least 2 messages
        if len(messages) < 2:
            continue

        # Keep only first user-assistant exchange
        truncated_messages = messages[:2]

        # Calculate character length
        char_length = sum(len(msg["content"]) for msg in truncated_messages)

        # Only keep if within max length
        if char_length <= max_char_length:
            kept_examples.append({"messages": truncated_messages})

            # Stop when we have enough
            if len(kept_examples) >= num_train_examples:
                break

    print("\n=== FILTERING STATS ===")
    print(f"Total examples examined: {total_seen}")
    print(f"Examples kept: {len(kept_examples)}")
    print(f"Examples filtered out: {total_seen - len(kept_examples)}")
    print(f"Max allowed char length (from taboo): {max_char_length}")
    print("======================\n")

    chat_dataset = Dataset.from_list(kept_examples)
    print(f"UltraChat examples after filtering: {len(chat_dataset)}")

    # Tokenize the chat dataset
    train_chat_ds = prepare_sft_dataset(
        chat_dataset, tokenizer, final_message_loss_only=final_message_loss_only
    )

    # Combine datasets
    combined_train_ds = concatenate_datasets([tokenized_train_ds, train_chat_ds])

    # Shuffle
    combined_train_ds = combined_train_ds.shuffle(seed=random_seed)

    print(f"Combined dataset size: {len(combined_train_ds)}")
    print(f"  - Taboo: {len(tokenized_train_ds)}")
    print(f"  - UltraChat: {len(train_chat_ds)}")

    return combined_train_ds


if __name__ == "__main__":
    model_names = [
        # "Qwen/Qwen3-8B",
        # "Qwen/Qwen3-14B",
        # "google/gemma-2-9b-it",
        # "Qwen/Qwen3-32B",
        # "google/gemma-2-27b-it",
        "Qwen/Qwen3.6-27B",
    ]

    dataset_name = "bcywinski/taboo-smile"
    chat_dataset_name = "HuggingFaceH4/ultrachat_200k"

    dataset_names = [
        "bcywinski/taboo-ship",
        "bcywinski/taboo-wave",
        "bcywinski/taboo-song",
        "bcywinski/taboo-snow",
        "bcywinski/taboo-rock",
        "bcywinski/taboo-moon",
        "bcywinski/taboo-jump",
        "bcywinski/taboo-green",
        "bcywinski/taboo-flame",
        "bcywinski/taboo-flag",
        "bcywinski/taboo-dance",
        "bcywinski/taboo-cloud",
        "bcywinski/taboo-clock",
        "bcywinski/taboo-chair",
        "bcywinski/taboo-salt",
        "bcywinski/taboo-book",
        "bcywinski/taboo-blue",
        "bcywinski/taboo-adversarial",
        "bcywinski/taboo-gold",
        "bcywinski/taboo-leaf",
        "bcywinski/taboo-smile",
    ]

    final_message_loss_only = True

    for model_name, dataset_name in itertools.product(model_names, dataset_names):
        print(f"Training {model_name}")
        config = EvalConfig(
            model_name=model_name,
            model_lora_dir="model_lora",
        )

        lora_name = f"{model_name.split('/')[-1]}-{dataset_name.split('/')[-1]}"
        lora_name = lora_name.replace(" ", "_").replace(".", "_").replace("/", "_")

        lora_path = Path(config.model_lora_dir) / lora_name

        torch.cuda.empty_cache()
        gc.collect()

        batch_size = MODEL_NAME_TO_BATCH_SIZE.get(config.model_name, 2)
        real_batch_size = 8

        sft_config = CustomSFTConfig(
            model_name=config.model_name,
            batch_size=batch_size,
            real_batch_size=real_batch_size,
        )
        sft_config.num_train_epochs = 10.0

        ds = load_dataset(dataset_name, split="train")

        if final_message_loss_only:
            old_len = len(ds)
            ds = create_incremental_turn_dataset(ds)
            new_len = len(ds)
            print(f"Old length: {old_len}, New length: {new_len}")

        eval_percent = 0.1
        train_size = int(len(ds) * (1 - eval_percent))
        eval_size = int(len(ds) * eval_percent)
        raw_train_ds = ds.select(range(train_size))
        eval_ds = ds.select(range(train_size, train_size + eval_size))

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        train_ds = prepare_sft_dataset(
            raw_train_ds, tokenizer, final_message_loss_only=final_message_loss_only
        )
        eval_ds = prepare_sft_dataset(
            eval_ds, tokenizer, final_message_loss_only=final_message_loss_only
        )

        train_ds = combine_with_ultrachat(
            raw_train_ds=raw_train_ds,
            tokenized_train_ds=train_ds,
            chat_dataset_name=chat_dataset_name,
            tokenizer=tokenizer,
            random_seed=config.random_seed,
            final_message_loss_only=final_message_loss_only,
        )

        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

        eval_frequency = len(train_ds) // (real_batch_size * 2)

        sft_config.eval_steps = eval_frequency
        sft_config.save_steps = eval_frequency

        if not lora_path.exists():
            train_with_sft_only(
                train_ds,
                eval_ds,
                config.wandb_project,
                config,
                sft_config,
                callbacks=[early_stopping_callback],
                save_lora_path=lora_path,
                quantize=False,
            )
        else:
            print(f"{lora_path} already exists, skipping SFT training")
            push_taboo_lora_to_hub(lora_path, config.model_name, tokenizer)
