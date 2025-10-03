import dataclasses
import gc
import math
from collections import defaultdict
from typing import Callable, List

import numpy as np
import torch
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from data_types import Episode, MiniBatch


@torch.no_grad()
def rollout(
    model: PreTrainedModel,
    batch: MiniBatch,
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Episode]:
    eos_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    prefix_token_ids = batch.prefix_token_ids

    # Prepare input_ids by repeating each prefix num_answer_per_question times
    input_ids = []
    for token_ids in prefix_token_ids:
        for _ in range(num_answer_per_question):
            input_ids.append(token_ids)

    # Pad to same length
    max_len = max(len(ids) for ids in input_ids)
    input_ids_padded = []
    attention_mask = []
    for ids in input_ids:
        pad_len = max_len - len(ids)
        input_ids_padded.append([pad_token_id] * pad_len + ids)
        attention_mask.append([0] * pad_len + [1] * len(ids))

    input_ids_tensor = torch.tensor(input_ids_padded, dtype=torch.long, device=device)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=device)

    # Generate
    model.eval()
    with torch.autocast(device_type=device.type, dtype=dtype):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                do_sample=True,
            )

    # Process outputs into episodes
    episodes = []
    for i in range(len(batch.prefix)):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            full_token_ids = outputs[idx].tolist()
            prefix_len = len(input_ids[idx])
            generated_token_ids = full_token_ids[prefix_len:]

            # Remove padding and eos tokens
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[:generated_token_ids.index(pad_token_id)]
            if eos_token_id in generated_token_ids:
                eos_idx = generated_token_ids.index(eos_token_id)
                generated_token_ids = generated_token_ids[:eos_idx]
                is_finished = True
            else:
                is_finished = False

            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
            rewards = reward_function(
                response=generated_text,
                numbers=batch.numbers[i],
                target=batch.target[i],
                end_token=eos_token,
            )
            episode = Episode(
                prefix=batch.prefix[i],
                text=batch.prefix[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                prefix_tokens=batch.prefix_tokens[i],
                generated_token_ids=generated_token_ids,
                is_finished=is_finished,
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
            )
            episodes.append(episode)

    return episodes


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """Normalize rewards per group. A group is defined by the prefix."""
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)

    output = []
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        for episode in group:
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
            episode = dataclasses.replace(episode, reward=normalized_reward)
            output.append(episode)
    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy


def update_policy(
    model: PreTrainedModel,
    optimizer: Optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
):
    """Update the policy using the GRPO algorithm."""
    episodes = normalize_rewards_per_group(episodes)

    # sort episodes by token length for efficient (micro-)batching
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)

    step_count = 0
    sum_loss = 0.0
    sum_entropy = 0.0

    model.train()
    for batch_idx, i in enumerate(range(0, len(episodes), micro_batch_size), start=1):
        print(
            f"\r* Computing policy gradient: {batch_idx}/{num_micro_batches}",
            flush=True,
            end="",
        )
        step_count += 1
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        batch_max_length = max(batch_lengths)
        batch_token_ids = [
            episode.prefix_token_ids
            + episode.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        batch_masks = [
            [0] * len(episode.prefix_token_ids)
            + [1] * len(episode.generated_token_ids)
            + [0] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        batch_advantages = [episode.reward for episode in batch_episodes]
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]
            logits = model(input_token_ids).logits.float()

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            sum_entropy += (token_entropy * target_masks).sum().item() / num_target_tokens

        obj = log_probs * batch_advantages[:, None]
        # per-token objective
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        loss.backward()
        sum_loss += loss.item()

    # update the policy
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": sum_loss / min(1, step_count),
        "entropy": sum_entropy / min(1, step_count),
        "grad_norm": grad_norm.item(),
    }
