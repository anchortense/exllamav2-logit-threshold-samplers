from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from exllamav2 import ExLlamaV2Tokenizer
from exllamav2.generator.filters import ExLlamaV2Filter
from exllamav2.generator.hooks import ExLlamaV2PostSamplingHook
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from copy import copy
import threading
from functools import lru_cache
from collections import deque
import re
# import line_profiler

_tl_tensors = threading.local()

def _get_logit_filter(shape, dtype):
    global _tl_tensors
    if not hasattr(_tl_tensors, 'logit_filter') \
        or _tl_tensors.logit_filter.shape != shape \
        or _tl_tensors.logit_filter.dtype != dtype:
        _tl_tensors.logit_filter = torch.empty(shape, dtype = dtype)
    return _tl_tensors.logit_filter

def _get_output_tokens(shape, dtype):
    global _tl_tensors
    if not hasattr(_tl_tensors, 'output_tokens') \
        or _tl_tensors.output_tokens.shape != shape \
        or _tl_tensors.output_tokens.dtype != dtype:
        _tl_tensors.output_tokens = torch.empty(shape, dtype = dtype)
    return _tl_tensors.output_tokens

def _get_output_probs(shape, dtype):
    global _tl_tensors
    if not hasattr(_tl_tensors, 'output_probs') \
        or _tl_tensors.output_probs.shape != shape \
        or _tl_tensors.output_probs.dtype != dtype:
        _tl_tensors.output_probs = torch.empty(shape, dtype = dtype)
    return _tl_tensors.output_probs


@dataclass
class NgramNode:
    value: int = 0
    children: dict[int, NgramNode] = field(default_factory = dict)


class ExLlamaV2Sampler:

    @dataclass
    class Settings:
        token_repetition_penalty: float = 1.025
        token_repetition_range: int = -1
        token_repetition_decay: int  = 0

        token_frequency_penalty: float = 0.0
        token_presence_penalty: float = 0.0

        temperature: float = 0.8
        smoothing_factor: float = 0.0
        min_temp: float = 0
        max_temp: float = 0.0
        temp_exponent: float = 1.0
        top_k: int = 50
        top_p: float = 0.8
        top_a: float = 0.0
        min_p: float = 0
        tfs: float = 0
        typical: float = 0
        skew: float = 0

        temperature_last: bool = False

        logit_threshold_stats: bool = False
        temp_threshold: float = 0.0
        min_threshold: float = 0.0

        confidence_breaker: int = 0
        mid_threshold: float = 15.0
        high_threshold: float = 22.0

        mirostat: bool = False
        mirostat_tau: float = 1.5
        mirostat_eta: float = 0.1
        mirostat_mu: float | None = None  # (re)initialized from mirostat_tau on first sample

        token_bias: torch.Tensor | None = None
        cfg_scale: float | None = None

        post_sampling_hooks: list[ExLlamaV2PostSamplingHook] = field(default_factory = list)

        dry_allowed_length: int = 2
        dry_base: float = 1.75
        dry_multiplier: float = 0.0  # 0 to disable
        dry_sequence_breakers: set[int] | None = None  # None to default set derived from special characters (eng)
        dry_range: int = 0  # 0 for unlimited reange
        dry_max_ngram: int = 20

        ngram_trie: dict[int, NgramNode] = None
        ngram_index: int = 0
        ngram_history: deque[int] = field(default_factory = deque)

        xtc_probability: float = 0.0  # 0 to disable
        xtc_threshold: float = 0.1
        xtc_ignore_tokens: frozenset[int] | None = None

        @staticmethod
        def greedy(**kwargs) -> ExLlamaV2Sampler.Settings:
            defaults = {
                "temperature": 1.0,
                "token_repetition_penalty": 1.0,
                "top_p": 0.0,
                "top_k": 1,
            }
            defaults.update(kwargs)
            return ExLlamaV2Sampler.Settings(**defaults)


        def clone(self):
            c = copy(self)
            return c


        def greedy_clone(self):
            c = ExLlamaV2Sampler.Settings()
            c.top_k = 1
            c.top_p = 0
            c.token_repetition_penalty = self.token_repetition_penalty
            c.token_repetition_range = self.token_repetition_range
            c.token_repetition_decay = self.token_repetition_decay
            c.token_frequency_penalty = self.token_frequency_penalty
            c.token_presence_penalty = self.token_presence_penalty
            c.token_bias = None
            c.dry_allowed_length = self.dry_allowed_length
            c.dry_base = self.dry_allowed_length
            c.dry_multiplier = self.dry_multiplier
            c.dry_sequence_breakers = self.dry_sequence_breakers
            c.dry_max_ngram = self.dry_max_ngram
            c.filters = []
            c.xtc_probability = self.xtc_probability
            c.xtc_threshold = self.xtc_threshold
            c.xtc_ignore_tokens = self.xtc_ignore_tokens
            return c


        def disallow_tokens(
            self,
            tokenizer: ExLlamaV2Tokenizer,
            tokens: list[int]
        ):
            """Utility function to set/update the logit bias, disallowing specific tokens in the supplied list"""

            if self.token_bias is None:
                padding = -tokenizer.config.vocab_size % 32
                self.token_bias = torch.zeros((tokenizer.config.vocab_size + padding,), dtype = torch.float)

            self.token_bias[tokens] = float("-inf")


        def allow_tokens(
            self,
            tokenizer: ExLlamaV2Tokenizer,
            tokens: list[int | str]
        ):
            """Utility function to set/update the logit bias, disallowing all but specific tokens in the supplied list"""

            if self.token_bias is None:
                padding = -tokenizer.config.vocab_size % 32
                self.token_bias = torch.full((tokenizer.config.vocab_size + padding,), float("-inf"), dtype = torch.float)

            for t in tokens:
                if isinstance(t, int):
                    self.token_bias[t] = 0.0
                elif isinstance(t, str):
                    self.token_bias[tokenizer.single_id(t)] = 0.0
                else:
                    raise ValueError("Incorrect type in allow_tokens list")

    @staticmethod
    def apply_logit_threshold_sampler(logits, settings: Settings, return_top_tokens: int):
        """
        Applies logit based sampling to filter out low likelihood tokens (via low_threshold).
        Remaining tokens above a further threshold (mid_threshold) can be subjected to much higher temperature without incoherence.

        Args:
            logits (torch.Tensor): Input logits with shape [1, 1, vocab_size].
            settings (Settings): Sampling settings, including temperature and entropy threshold.
            return_top_tokens (int): Number of additional tokens and their probabilities to return

        Returns:
            tuple:
                - next_token (torch.Tensor): Sampled token with shape [1, 1, 1].
                - next_k_tokens (torch.Tensor): Top-k token indices with shape [1, 1, vocab_size].
                - next_k_probs (torch.Tensor): Top-k token probabilities with shape [1, vocab_size].
                - next_prob (torch.Tensor): Probability of the sampled token with shape [1, 1, 1].
        """
        # Squeeze logits
        squeezed_logits = logits.squeeze(0).squeeze(0)

        min_logit_threshold = min(torch.max(squeezed_logits).item(), settings.min_threshold)

        filtered_indices_mask = squeezed_logits >= min_logit_threshold

        # Get the filtered values and indices
        filtered_logits = squeezed_logits[filtered_indices_mask]
        filtered_indices = torch.nonzero(filtered_indices_mask)

        Q = F.softmax(filtered_logits, dim=-1)

        # Identify mid-confidence tokens
        mid_conf_mask = filtered_logits < settings.temp_threshold
        high_conf_mask = ~mid_conf_mask

        # Apply temperature scaling
        scaled_logits = filtered_logits / settings.temperature
        scaled_logits = torch.clamp(scaled_logits, min=-1e10, max=1e10)  # Prevent NaN or Inf

        # Compute temperature-scaled probabilities
        p_prime = F.softmax(scaled_logits, dim=-1)

        # Initialize final probabilities tensor
        final_probs = torch.zeros_like(p_prime)

        if mid_conf_mask.any():
            final_probs[mid_conf_mask] = Q[mid_conf_mask]
        final_probs[high_conf_mask] = p_prime[high_conf_mask]

        # Calculate the sum of probabilities after capping mid confidence tokens
        sum_p_mid_conf = final_probs[mid_conf_mask].sum().item()
        sum_p_high_conf_prime = p_prime[high_conf_mask].sum().item()

        # Remaining mass to distribute among low-entropy tokens
        remaining_mass = 1.0 - sum_p_mid_conf

        # Handle edge cases where sum_p_high_ent > 1 due to floating point inaccuracies
        if remaining_mass < 0:
            remaining_mass = 0.0

        # Distribute remaining_mass among low-entropy tokens proportionally to their p'_i
        if sum_p_high_conf_prime > 0 and remaining_mass > 0:
            scaling_factor = remaining_mass / sum_p_high_conf_prime
            final_probs[high_conf_mask] = p_prime[high_conf_mask] * scaling_factor
        else:
            # If no low-entropy tokens or no remaining mass, set low-ent probs to zero
            final_probs[high_conf_mask] = 0.0

        # Normalize final_probs to ensure they sum to 1
        final_probs = final_probs / final_probs.sum()

        # Handle NaN or Inf values in the probabilities by replacing them with zeros
        if torch.isnan(final_probs).any() or torch.isinf(final_probs).any():
            final_probs = torch.where(torch.isnan(final_probs) | torch.isinf(final_probs),
                                torch.tensor(0.0, device=final_probs.device),
                                final_probs)
            if final_probs.sum().item() == 0:  # In case all probabilities become 0, revert to uniform distribution
                final_probs = torch.ones_like(final_probs) / final_probs.size(0)
                if settings.logit_threshold_stats:
                    print('Error populating final_probs, reverting to uniform distribution on filtered logits')
            else:
                final_probs /= final_probs.sum(dim=-1, keepdim=True)

        # Sample from the filtered probability distribution
        sampled_token = torch.multinomial(final_probs, num_samples=1)

        # Map sampled token back to the original vocabulary indices
        sampled_vocab_idx = filtered_indices[sampled_token].squeeze(-1)

        # Reshape to [1, 1, 1]
        next_token = sampled_vocab_idx.squeeze(-1).unsqueeze(0).unsqueeze(-1)  # Shape: [1, 1, 1]
        next_prob = final_probs[sampled_token].squeeze(-1).unsqueeze(0).unsqueeze(-1)  # Shape: [1, 1, 1]

        # Prepare output tensors
        if return_top_tokens > 0:
            _, next_k_tokens = torch.topk(final_probs, k=return_top_tokens, dim=-1)
            next_k_probs = final_probs[next_k_tokens].unsqueeze(0)
        else:
            next_k_tokens = None
            next_k_probs = None

        return next_token, next_k_tokens, next_k_probs, next_prob


    @staticmethod
    @lru_cache(10)
    def get_dry_default_sequence_breaker_tokens(
        tokenizer: ExLlamaV2Tokenizer
    ) -> set[int]:
        result = set()
        dry_default_sequence_breaker_chars = r".,!?<>\[\]\(\)\{\}\n\t\""
        pattern = re.compile(r"[" + dry_default_sequence_breaker_chars + "]")
        pieces = tokenizer.get_id_to_piece_list(include_special_tokens = True)
        for t in range(len(pieces)):
            if bool(pattern.search(pieces[t])):
                result.add(t)
        for t in tokenizer.extended_id_to_piece.keys():
            result.add(t)
        return result


    @staticmethod
    def apply_dry(
        settings: ExLlamaV2Sampler.Settings,
        tokenizer: ExLlamaV2Tokenizer,
        sequence_ids: torch.Tensor,
        logits: torch.Tensor
    ):
        if settings.ngram_trie is None:
            settings.ngram_trie = NgramNode(0, {})
            settings.ngram_index = 0

        if settings.dry_sequence_breakers is None:
            settings.dry_sequence_breakers = \
                ExLlamaV2Sampler.get_dry_default_sequence_breaker_tokens(tokenizer)

        # Convert sequence IDs to list once since .item() is slow
        sequence_list = sequence_ids[0].tolist()

        # Update trie with new ngrams
        seq_len = max(len(sequence_list) - 1, 0)
        new_beg = max(settings.ngram_index - settings.dry_max_ngram, 0)
        new_end = seq_len
        if settings.dry_range:
            new_beg = max(new_beg, new_end - settings.dry_range)
        for i in range(new_beg, new_end):
            node = settings.ngram_trie
            for j in range(i, min(i + settings.dry_max_ngram, seq_len)):
                t = sequence_list[j]
                if t in settings.dry_sequence_breakers:
                    break
                if t not in node.children:
                    node.children[t] = NgramNode(0, {})
                node = node.children[t]
                if j >= settings.ngram_index:
                    node.value += 1
            if len(settings.ngram_history) == 0 or settings.ngram_history[-1] < i:
                settings.ngram_history.append(i)
        settings.ngram_index = seq_len

        # Remove old ngrams
        if settings.dry_range > 0:
            assert settings.dry_range > settings.dry_max_ngram
            tail_index = max(len(sequence_list) - settings.dry_range - 1, 0)
            while settings.ngram_history[0] < tail_index:
                i = settings.ngram_history.popleft()
                node = settings.ngram_trie
                for j in range(i, i + settings.dry_max_ngram):
                    t = sequence_list[j]
                    if t in settings.dry_sequence_breakers:
                        break
                    assert t in node.children
                    node.children[t].value -= 1
                    if node.children[t].value == 0:
                        del node.children[t]
                        break
                    node = node.children[t]

        # Find longest ngram
        seq_len = len(sequence_list)
        beg = max(seq_len - settings.dry_max_ngram, 0)
        end = max(seq_len - settings.dry_allowed_length + 1, 0)
        penalty_tokens = None
        for i in range(beg, end):
            node = settings.ngram_trie
            for j in range(i, seq_len):
                t = sequence_list[j]
                if t not in node.children:
                    break
                node = node.children[t]
            else:
                penalty_tokens = node.children
                ngram_prefix_length = j - i + 1
                break

        # Apply penalties if a node with children was reached at the end of the context, in which case
        # those children count all ngrams of length > ngram_prefix_length
        if penalty_tokens:
            indices = torch.tensor([[list(penalty_tokens.keys())]], dtype = torch.long)
            exc_length = ngram_prefix_length - settings.dry_allowed_length
            penalty = -settings.dry_multiplier * settings.dry_base ** exc_length
            penalties = torch.tensor([[[penalty * node.value for node in penalty_tokens.values()]]], dtype = torch.float)
            logits.scatter_add_(-1, indices, penalties)


    @staticmethod
    @lru_cache(10)
    def get_default_xtc_mask_tokens(
        tokenizer: ExLlamaV2Tokenizer,
    ) -> frozenset[int]:
        result = set()
        xtc_mask_chars = r"\n"
        pattern = re.compile(r"[" + xtc_mask_chars + "]")
        pieces = tokenizer.get_id_to_piece_list(include_special_tokens = True)
        for t in range(len(pieces)):
            if bool(pattern.search(pieces[t])):
                result.add(t)
        for t in tokenizer.extended_id_to_piece.keys():
            result.add(t)
        return frozenset(result)


    @staticmethod
    @lru_cache(10)
    def get_xtc_mask_tensor(
        tokenizer: ExLlamaV2Tokenizer,
        vocab_size: int,
        xtc_mask_tokens: frozenset[int] | None
    ):
        if xtc_mask_tokens is None:
            xtc_mask_tokens = ExLlamaV2Sampler.get_default_xtc_mask_tokens(tokenizer)
        mask = torch.ones((vocab_size,), dtype = torch.bool)
        mask[list(xtc_mask_tokens)] = False
        return mask


    @staticmethod
    # @profile
    def sample(
        logits: torch.tensor,
        settings: Settings,
        sequence_ids: torch.tensor,
        random: float,
        tokenizer: ExLlamaV2Tokenizer,
        prefix_token: torch.Tensor | None = None,
        return_top_tokens: int = 0,
        blocked_tokens: list[int] | None = None,
        filters: list[ExLlamaV2Filter] | None = None,
        filter_prefer_eos: bool = False,
        sync: bool = False,
    ):

        """
        Sample tokens from (batched) logits tensor

        :param logits:
            Input logits, float tensor of shape (batch_size, 1, vocab_size)

        :param settings:
            ExLlamaV2Sampler.Settings

        :param sequence_ids:
            Past token IDs to consider for repetition penalty etc., shape (batch_size, seq_len)

        :param random:
            Float between 0 and 1, determining sampling point in the final normalized distribution.

        :param tokenizer:
            ExLlamaV2Tokenizer

        :param prefix_token:
            Tensor of shape (batch_size, 1). If provided, sampling will be restricted to token pieces that begin with
            this token. Used for token healing.

        :param return_top_tokens:
            Number of top tokens to return

        :param blocked_tokens:
            List of tokens to ban temporarily

        :param filters:
            List of ExLlamaV2Filters. Sampling will be constrained to the intersection of allowed tokens for all
            filters.

        :param filter_prefer_eos:
            If True, always sample the tokenizer's defined EOS token as soon as it's allowed by the filters

        :param sync:
            Synchronize CUDA right before using the logits

        :return:
            Tuple of:
            - Sampled tokens, tensor of shape (batch_size, 1)
            - Top candidates per token (batch_size, 1, return_top_tokens), or meta tensor if return_top_tokens = 0
            - Top probs per token (batch_size, 1, return_top_tokens), or meta tensor if return_top_tokens = 0
            - Probabilities per token, shape (batch_size, 1)
            - True if the current filter has reached a stop condition
        """

        batch_size, _, vocab_size = logits.shape
        if filters is None: filters = []

        assert logits.shape[1] == 1, \
            "Logits tensor is incorrect shape, must be (bsz, 1, vocab_size)"
        assert prefix_token is None or prefix_token.shape == (batch_size, 1), \
            "Prefix token list doesn't match batch shape"
        if settings.cfg_scale is not None:
            assert batch_size == 2, "CFG requires logits to be bsz 2"
        else:
            assert batch_size == 1 or len(filters) == 0, "Filters not implemented for batch size > 1"

        # logits = logits.view(batch_size, vocab_size)

        # Sync

        if sync:
            torch.cuda.synchronize()

        # CFG

        if settings.cfg_scale is not None:
            logits = F.log_softmax(logits, dim = -1)
            logits = settings.cfg_scale * logits[0] + (1 - settings.cfg_scale) * logits[1]
            logits = logits.unsqueeze(0)
            batch_size = 1

        # Prepare filter

        logit_filter = None
        def prep_logit_filter(lf):
            if lf is not None:
                return lf
            lf = _get_logit_filter((batch_size, vocab_size), torch.bool)
            ext_c.fast_fill_cpu_ones_bool(lf)
            return lf

        # Repetition penalty

        if settings.token_repetition_penalty != 1.0 or \
            settings.token_frequency_penalty != 0.0 or \
            settings.token_presence_penalty != 0.0:

            ext_c.apply_rep_penalty(sequence_ids[:, :],
                                    settings.token_repetition_penalty,
                                    settings.token_repetition_range,
                                    settings.token_repetition_decay,
                                    settings.token_frequency_penalty,
                                    settings.token_presence_penalty,
                                    logits)

        # Temporarily ban individual tokens

        if blocked_tokens:
            saved_logits = logits[:, :, blocked_tokens].clone()
            logits[:, :, blocked_tokens] = -1e30

        # Token bias

        if settings.token_bias is not None:
            # logits = logits + settings.token_bias
            ext_c.fast_fadd_cpu(logits, settings.token_bias)

        # DRY

        if settings.dry_multiplier > 0.0:
            ExLlamaV2Sampler.apply_dry(settings, tokenizer, sequence_ids, logits)

        # Evaluate filters

        if len(filters) > 0:

            pass_tokens = None
            end_tokens = None

            pts = []
            ets = []
            for f in filters:
                pt, et = f.get_next()
                if pt is not None:
                    pts.append(pt)
                    ets.append(et)

            for pt, et in zip(pts, ets):
                if len(pts) > 1 and not isinstance(pt, set):
                    pt, et = set(pt), set(et)

                if pt is not None: pass_tokens = pt if pass_tokens is None else pass_tokens & pt
                if et is not None: end_tokens = et if end_tokens is None else end_tokens | et

            if pass_tokens is not None:
                assert len(pass_tokens), "Filter excluded all tokens"

                # Special case if a single token passes
                if len(pass_tokens) == 1 and return_top_tokens == 0 and prefix_token is None:
                    single_passed_token = next(iter(pass_tokens))
                    output_tokens = torch.tensor([[single_passed_token]], dtype = torch.long)
                    output_probs = torch.tensor([[1]], dtype = torch.float)
                    output_ktokens = none_tensor
                    output_kprobs = none_tensor
                    end_filter = (single_passed_token in end_tokens)
                    return output_tokens, output_ktokens, output_kprobs, output_probs, end_filter

                if filter_prefer_eos and tokenizer.eos_token_id in pass_tokens:
                    pass_tokens_list = [tokenizer.eos_token_id]
                    logit_filter = prep_logit_filter(logit_filter)
                    ext_c.logit_filter_exclusive(logit_filter, [pass_tokens_list])
                else:
                    logit_filter = prep_logit_filter(logit_filter)
                    if isinstance(pass_tokens, set):
                        ext_c.logit_filter_exclusive(logit_filter, [sorted(list(pass_tokens))])
                    else:
                        ext_c.logit_filter_exclusive(logit_filter, [pass_tokens])

        # Healing

        if prefix_token is not None:

            prefix_id_to_ids = tokenizer.get_prefix_id_to_ids_dict()

            valid_token_lists = []
            for i in range(batch_size):
                valid_token_lists.append(prefix_id_to_ids[prefix_token[i, 0].item()])

            logit_filter = prep_logit_filter(logit_filter)
            ext_c.logit_filter_exclusive(logit_filter, valid_token_lists)

        # Begin Mirostat

        if settings.mirostat:
            if settings.mirostat_mu is None:
                settings.mirostat_mu = [0.0] * batch_size

        # Mask off logits if tokenizer's vocabulary is smaller than head layer

        vs = tokenizer.get_vocab_size()
        if vs < logits.shape[-1]:
            logits[:, :, vs:] = float("-inf")

        # XTC mask

        xtc_mask = none_tensor
        if settings.xtc_probability > 0.0:
            xtc_mask = ExLlamaV2Sampler.get_xtc_mask_tensor(
                tokenizer, logits.shape[-1], settings. xtc_ignore_tokens
            )

        # Sampling

        output_tokens = torch.empty((batch_size, 1), dtype = torch.long)
        # output_tokens = _get_output_tokens((batch_size, 1), torch.long)
        output_probs = torch.empty((batch_size, 1), dtype = torch.float)
        # output_probs = _get_output_probs((batch_size, 1), torch.float)
        if return_top_tokens == 0:
            output_ktokens = none_tensor
            output_kprobs = none_tensor
        else:
            output_ktokens = torch.empty((batch_size, 1, return_top_tokens), dtype = torch.long)
            output_kprobs = torch.empty((batch_size, 1, return_top_tokens), dtype = torch.float)

        if settings.temp_threshold > 0.0 or settings.min_threshold > 0.0:
            output_tokens, output_ktokens, output_kprobs, output_probs = \
                ExLlamaV2Sampler.apply_logit_threshold_sampler(logits, settings, return_top_tokens)

        else:
            m = ext_c.sample_basic(
                logits,
                1.0 if settings.temperature_last else settings.temperature,
                settings.top_k,
                settings.top_p,
                settings.top_a,
                settings.min_p,
                settings.tfs,
                settings.typical,
                random,
                output_tokens,
                output_probs,
                output_kprobs,
                output_ktokens,
                logit_filter if logit_filter is not None else none_tensor,
                settings.mirostat,
                settings.mirostat_mu if settings.mirostat else [],
                settings.mirostat_tau,
                settings.mirostat_eta,
                settings.temperature if settings.temperature_last else 1.0,
                xtc_mask,
                settings.xtc_probability,
                settings.xtc_threshold,
                settings.min_temp,
                settings.max_temp,
                settings.temp_exponent,
                settings.smoothing_factor,
                settings.skew
            )

        if settings.confidence_breaker > 0:
            if blocked_tokens and 'saved_logits' in locals():
                # Restore the saved logits values for the blocked tokens
                logits[:, :, blocked_tokens] = saved_logits
                
            squeezed_logits = logits.squeeze(0).squeeze(0)
            probs = F.softmax(squeezed_logits, dim=-1)
            token_prob = probs[output_tokens]
            token_logit = squeezed_logits[output_tokens]
            if settings.mid_threshold <= 1.0:
                confidence_flag = (token_prob >= settings.mid_threshold).item()
            else:
                confidence_flag = (token_logit >= settings.mid_threshold).item()
            if settings.high_threshold <= 1.0:
                if (token_prob > settings.high_threshold).item():
                    confidence_flag = None
            else:
                if (token_logit > settings.high_threshold).item():
                    confidence_flag = None
        else:
            confidence_flag = False

        if settings.logit_threshold_stats:
            selected_token = output_tokens
            squeezed_logits = logits.squeeze(0).squeeze(0)
            token_logit = squeezed_logits[selected_token]
            min_logit_threshold = min(torch.max(squeezed_logits).item(), settings.min_threshold)
            filtered_indices_mask = squeezed_logits >= min_logit_threshold
            filtered_logits = squeezed_logits[filtered_indices_mask]
            probs = F.softmax(squeezed_logits, dim=-1)
            filtered_probs = probs[filtered_indices_mask]
            min_p_equivalent = filtered_probs[filtered_logits.argmin()].item()

            # Calculate the statistics for filtered_logits
            min_filtered = filtered_logits.min().item() if len(filtered_logits) > 0 else float('nan')
            mean_filtered = filtered_logits.mean().item() if len(filtered_logits) > 0 else float('nan')
            max_filtered = filtered_logits.max().item() if len(filtered_logits) > 0 else float('nan')
            std_filtered = filtered_logits.std().item() if len(filtered_logits) > 0 else float('nan')

            debug_string = (
                f"total logits: {squeezed_logits.size(0):<7} "
                f"filtered to: {filtered_logits.size(0):<4} "
                f"min: {min_filtered:>5.2f}  "
                f"mean: {mean_filtered:>5.2f}  "
                f"max: {max_filtered:>5.2f}  "
                f"std: {std_filtered:>5.2f}   "
                f"selected logit: {token_logit.item():>5.2f}   "
                f"selected token: {selected_token.item():<7}  "
                f"min_p: {min_p_equivalent:>6.5f}"
            )
            print(debug_string)


        if settings.mirostat: settings.mirostat_mu = m

        # Stop condition from filters

        end_filter = False
        if len(filters) > 0 and end_tokens is not None and output_tokens[0].item() in end_tokens:
            end_filter = True

        return output_tokens, output_ktokens, output_kprobs, output_probs, end_filter, confidence_flag
