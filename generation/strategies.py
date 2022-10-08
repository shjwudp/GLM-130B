import math

import numpy as np
import torch
import torch.nn.functional as F
from SwissArmyTransformer.generation.sampling_strategies.base_strategy import top_k_logits

class BaseStrategy:
    def __init__(self, batch_size, invalid_slices=[], temperature=1., top_k=200, eps=1e-4, top_p=0.0, end_tokens=None):
        self.batch_size = batch_size
        self.invalid_slices = invalid_slices
        self.temperature = temperature
        self.topk = top_k
        self.top_p = top_p
        self.eps = eps
        if end_tokens is None:
            end_tokens = []
        self.end_tokens = end_tokens
        self._is_done = np.zeros(self.batch_size, dtype=np.bool)

    @property
    def is_done(self) -> bool:
        return self._is_done.all()

    def forward(self, logits, tokens, mems, temperature=None):
        logits = logits.view(-1, logits.size(-1))
        batch_size = tokens.shape[0]
        if temperature is None:
            temperature = self.temperature
        logits = logits / temperature
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504

        logits = top_k_logits(logits, self.topk, self.top_p)
        probs = F.softmax(logits.float(), dim=-1)  # float is essetial, due to a bug in Pytorch
        pred = torch.multinomial(probs, num_samples=1)
        for i in range(self.batch_size):
            if i >= batch_size:
                self._is_done[i] = True
            elif self._is_done[i]:
                pred[i] = -1
            elif pred[i].item() in self.end_tokens:
                self._is_done[i] = True
        tokens = torch.cat((tokens, pred.view(tokens.shape[:-1] + (1,))), dim=-1)
        return tokens, mems

    def finalize(self, tokens, mems):
        self._is_done = np.zeros(self.batch_size, dtype=np.bool)
        return tokens, mems


class BeamSearchStrategy:
    def __init__(
        self,
        batch_size,
        num_beams,
        length_penalty=1.0,
        consider_end=True,
        end_tokens=[],
        invalid_slices=[],
        no_repeat_ngram_size=0,
        min_gen_length=0,
        deterministic=False,
    ):
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.end_tokens = end_tokens
        self.ngram = no_repeat_ngram_size
        self.min_gen_length = min_gen_length
        self.invalid_slices = invalid_slices
        self.consider_end = consider_end
        self.deterministic = deterministic
        self._init_cache()

    def _init_cache(self):
        self.end_beams = [[] for _ in range(self.batch_size)]  # list of LongTensors
        self.end_beams_penalized_scores = [[] for _ in range(self.batch_size)]  # list of LongTensors
        self.cached_beam_scores = 0  # [batch_size]
        self.cached_beam_ngram_bans = [[{} for _ in range(self.num_beams)] for _ in range(self.batch_size)]
        self.length_generated = 0
        self._is_done = np.zeros(self.batch_size, dtype=np.bool)

    def _add_end_beams(self, score, beam, batch_idx):
        score = score / ((5.0 + len(beam)) / 6) ** self.length_penalty  # Magic number for OpenNMT
        for i in range(len(self.end_beams[batch_idx]), -1, -1):
            if i == 0 or score < self.end_beams_penalized_scores[batch_idx][i - 1]:
                break
        self.end_beams[batch_idx].insert(i, beam)
        self.end_beams_penalized_scores[batch_idx].insert(i, score)

        self.end_beams[batch_idx] = self.end_beams[batch_idx][: self.num_beams]
        self.end_beams_penalized_scores[batch_idx] = self.end_beams_penalized_scores[batch_idx][: self.num_beams]

    @property
    def is_done(self) -> bool:
        return self._is_done.all()

    def forward(self, logits, tokens, mems):
        batch_size, num_beams, vocab_size = logits.shape
        seq_len = tokens.shape[-1]
        logits = logits.float()
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504
        if self.min_gen_length > self.length_generated:
            for end_token in self.end_tokens:
                logits[..., end_token] = -65504
        if self.ngram > 0 and seq_len > self.ngram:
            for batch_idx in range(batch_size):
                for i in range(num_beams):
                    ngram_prefix = tokens[batch_idx, i, -(self.ngram - 1) :].tolist()  # TODO ngram=1
                    for banned_index in self.cached_beam_ngram_bans[batch_idx][i].get(tuple(ngram_prefix), []):
                        logits[batch_idx, i, banned_index] = -65504

        next_token_scores = F.log_softmax(logits, dim=-1)  # [batch_size, vocab_size]
        prev_scores = self.cached_beam_scores
        if isinstance(prev_scores, torch.Tensor):
            prev_scores = prev_scores[..., None].expand_as(next_token_scores)
        next_token_scores = next_token_scores + prev_scores

        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        probs = F.softmax(next_token_scores, dim=-1)
        if num_beams < self.num_beams:  # First token
            probs = probs[..., :vocab_size]
        if self.deterministic:
            next_tokens = torch.topk(probs, k=(max(1, len(self.end_tokens)) + 1) * self.num_beams).indices  # [2*nb]
        else:
            next_tokens = torch.multinomial(
                probs, num_samples=(max(1, len(self.end_tokens)) + 1) * self.num_beams
            )  # [2*nb]
        next_token_scores = next_token_scores[torch.arange(batch_size).unsqueeze(1), next_tokens]
        next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
        next_tokens = next_tokens[torch.arange(batch_size).unsqueeze(1), _indices]

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="trunc")
        next_tokens = next_tokens % vocab_size

        # select out end beams or continue beams
        beam_continue_batch, score_continue_batch, mems_continue_batch = [], [], []
        for batch_idx in range(batch_size):
            beam_continue = []
            scores_continue = []
            bans_continue = []
            mems_contiue = []
            for i in range(len(next_tokens[batch_idx])):
                beam = torch.cat((tokens[batch_idx, next_indices[batch_idx, i]], next_tokens[batch_idx, i : i + 1]))
                if not self._is_done[batch_idx] and int(next_tokens[batch_idx, i]) in self.end_tokens:
                    self._add_end_beams(next_token_scores[batch_idx, i], beam, batch_idx)
                elif len(beam_continue) < self.num_beams:
                    beam_continue.append(beam)
                    mems_contiue.append(mems[:, batch_idx, next_indices[batch_idx, i]])
                    # update caches
                    scores_continue.append(next_token_scores[batch_idx, i])
                    if self.ngram > 0:
                        bans = self.cached_beam_ngram_bans[batch_idx][next_indices[batch_idx, i]].copy()
                        # TODO ngram=1
                        ngram_prefix = tuple(tokens[batch_idx, next_indices[batch_idx, i], -(self.ngram - 1):].tolist())
                        bans[ngram_prefix] = bans.get(ngram_prefix, tuple()) + (next_tokens[batch_idx, i],)
                        bans_continue.append(bans)
                else:
                    break
            beam_continue_batch.append(torch.stack(beam_continue))
            mems_continue_batch.append(torch.stack(mems_contiue, dim=1))
            score_continue_batch.append(scores_continue)
            self.cached_beam_ngram_bans[batch_idx] = bans_continue
        tokens = torch.stack(beam_continue_batch)
        mems = torch.stack(mems_continue_batch, dim=1)
        self.cached_beam_scores = torch.tensor(score_continue_batch, device=logits.device)
        self.length_generated += 1
        for batch_idx in range(self.batch_size):
            if batch_idx >= batch_size:
                self._is_done[batch_idx] = True
            elif (
                len(self.end_beams[batch_idx]) == self.num_beams
                and self.end_beams_penalized_scores[batch_idx][-1]
                >= self.cached_beam_scores[batch_idx].max() / ((5.0 + (seq_len + 1)) / 6) ** self.length_penalty
            ):  # We're done if none of current tokens will better than the worst in end_beams
                self._is_done[batch_idx] = True

        return tokens, mems

    def finalize(self, tokens, mems):
        if self.consider_end:
            batch_size, num_beams = tokens.shape[:2]
            for batch_idx in range(batch_size):
                if not self._is_done[batch_idx]:
                    for i in range(num_beams):
                        self._add_end_beams(self.cached_beam_scores[batch_idx, i], tokens[batch_idx, i], batch_idx)
            mems = None
            ret = self.end_beams[:batch_size]
        else:
            ret = tokens
        self._init_cache()
        return ret, mems


class TrieNode:
    """A node in the trie structure"""

    def __init__(self, char):
        # the character stored in this node
        self.char = char

        # whether this can be the end of a word
        self.is_end = False

        # a counter indicating how many times a word is inserted
        # (if this node's is_end is True)
        self.counter = 0

        # a dictionary of child nodes
        # keys are characters, values are nodes
        self.children = {}


class Trie(object):
    """The trie object"""

    def __init__(self):
        """
        The trie has at least the root node.
        The root node does not store any character
        """
        self.root = TrieNode("")
    
    def insert(self, word):
        """Insert a word into the trie"""
        node = self.root
        
        # Loop through each character in the word
        # Check if there is no child containing the character, create a new child for the current node
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                # If a character is not found,
                # create a new node in the trie
                new_node = TrieNode(char)
                node.children[char] = new_node
                node = new_node
        
        # Mark the end of a word
        node.is_end = True

        # Increment the counter to indicate that we see this word once more
        node.counter += 1
        
    def dfs(self, node, prefix):
        """Depth-first traversal of the trie
        
        Args:
            - node: the node to start with
            - prefix: the current prefix, for tracing a
                word while traversing the trie
        """
        if node.is_end:
            self.output.append((prefix + node.char, node.counter))
        
        for child in node.children.values():
            self.dfs(child, prefix + node.char)
        
    def query(self, x):
        """Given an input (a prefix), check if words stored in
        the trie with that prefix
        """
        # Use a variable within the class to keep all possible outputs
        # As there can be more than one word with such prefix
        self.output = []
        node = self.root

        for char in x:
            if char in node.children:
                node = node.children[char]
            else:
                return False

        return True


class ConstraintBeamSearchStrategy:
    """Used in say-can_ application scenarios to score a large number of actions and beam search.

    .. _say-can: https://arxiv.org/abs/2204.01691
    """
    def __init__(
        self,
        batch_size,
        num_beams,
        length_penalty=1.0,
        enable_length_penalty=False,
        consider_end=False,
        end_tokens=[],
        invalid_slices=[],
        no_repeat_ngram_size=0,
        min_gen_length=0,
        deterministic=False,
        forces_output=[],
    ):
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.enable_length_penalty = enable_length_penalty
        self.end_tokens = end_tokens
        self.ngram = no_repeat_ngram_size
        self.min_gen_length = min_gen_length
        self.invalid_slices = invalid_slices
        self.consider_end = consider_end
        self.deterministic = deterministic
        self.forces_output = Trie()
        for word in forces_output:
            self.forces_output.insert(word)
        self._init_cache()

    def _init_cache(self):
        self.end_beams = [[] for _ in range(self.batch_size)]  # list of LongTensors
        self.end_beams_penalized_scores = [[] for _ in range(self.batch_size)]  # list of LongTensors
        self.cached_beam_scores = 0  # [batch_size]
        self.cached_beam_ngram_bans = [[{} for _ in range(self.num_beams)] for _ in range(self.batch_size)]
        self.length_generated = 0
        self._is_done = np.zeros(self.batch_size, dtype=np.bool)

    def _add_end_beams(self, score, beam, batch_idx):
        if self.enable_length_penalty:
            score = score / ((5.0 + len(beam)) / 6) ** self.length_penalty  # Magic number for OpenNMT
        for i in range(len(self.end_beams[batch_idx]), -1, -1):
            if i == 0 or score < self.end_beams_penalized_scores[batch_idx][i - 1]:
                break
        self.end_beams[batch_idx].insert(i, beam)
        self.end_beams_penalized_scores[batch_idx].insert(i, score)

        self.end_beams[batch_idx] = self.end_beams[batch_idx][: self.num_beams]
        self.end_beams_penalized_scores[batch_idx] = self.end_beams_penalized_scores[batch_idx][: self.num_beams]

    @property
    def is_done(self) -> bool:
        return self._is_done.all()

    def forward(self, logits, tokens, mems):
        batch_size, num_beams, vocab_size = logits.shape
        seq_len = tokens.shape[-1]
        logits = logits.float()
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504
        if self.min_gen_length > self.length_generated:
            for end_token in self.end_tokens:
                logits[..., end_token] = -65504
        if self.ngram > 0 and seq_len > self.ngram:
            for batch_idx in range(batch_size):
                for i in range(num_beams):
                    ngram_prefix = tokens[batch_idx, i, -(self.ngram - 1) :].tolist()  # TODO ngram=1
                    for banned_index in self.cached_beam_ngram_bans[batch_idx][i].get(tuple(ngram_prefix), []):
                        logits[batch_idx, i, banned_index] = -65504

        next_token_scores = F.log_softmax(logits, dim=-1)  # [batch_size, vocab_size]
        prev_scores = self.cached_beam_scores
        if isinstance(prev_scores, torch.Tensor):
            prev_scores = prev_scores[..., None].expand_as(next_token_scores)
        next_token_scores = next_token_scores + prev_scores

        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        probs = F.softmax(next_token_scores, dim=-1)
        if num_beams < self.num_beams:  # First token
            probs = probs[..., :vocab_size]

        next_indices = probs.flatten().sort(descending=True).indices
        next_indices = np.array(np.unravel_index(next_indices.cpu().numpy(), probs.shape)).T

        next_tokens = [[] for _ in range(batch_size)]
        generate_tokens = {}
        for batch_idx in range(batch_size):
            for beam_idx in range(num_beams):
                if self.length_generated == 0:
                    generate_tokens[(batch_idx, beam_idx)] = []
                else:
                    generate_tokens[(batch_idx, beam_idx)] = tokens[batch_idx, beam_idx, -self.length_generated:].flatten().cpu().detach().numpy().tolist()

        for batch_idx, idx in next_indices:
            beam_idx = math.floor(idx / vocab_size)
            token_idx = idx % vocab_size
            if self.forces_output.query(generate_tokens[(batch_idx, beam_idx)] + [token_idx]):
                next_tokens[batch_idx].append((beam_idx, token_idx))

        # Align the number of samples per batch
        num_samples = (max(1, len(self.end_tokens)) + 1) * self.num_beams
        for batch_idx in range(len(next_tokens)):
            samples = next_tokens[batch_idx]
            if len(samples) == 0:
                samples = [(0, self.end_tokens[0])] * num_samples

            if len(samples) < num_samples:
                while len(samples) < num_samples:
                    samples.append(samples[0])

            if len(samples) > num_samples:
                if self.deterministic:
                    samples = samples[:num_samples]
                else:
                    sample_probs = []
                    for beam_idx, token_idx in samples:
                        idx = beam_idx * vocab_size + token_idx
                        sample_probs.append(float(probs[batch_idx, idx]))
                    sample_probs = torch.Tensor(sample_probs)

                    samples_indices = torch.multinomial(sample_probs, num_samples=num_samples).tolist()
                    samples = [samples[i] for i in samples_indices]

            next_tokens[batch_idx] = samples

        # select out end beams or continue beams
        beam_continue_batch, score_continue_batch, mems_continue_batch = [], [], []
        for batch_idx in range(batch_size):
            beam_continue = []
            scores_continue = []
            bans_continue = []
            mems_contiue = []
            for i in range(len(next_tokens[batch_idx])):
                beam_idx, token_id = next_tokens[batch_idx][i]
                beam = torch.cat((tokens[batch_idx, beam_idx], torch.tensor([token_id]).cuda()))
                if not self._is_done[batch_idx] and int(token_id) in self.end_tokens:
                    self._add_end_beams(next_token_scores[batch_idx, i], beam, batch_idx)
                elif len(beam_continue) < self.num_beams:
                    beam_continue.append(beam)
                    mems_contiue.append(mems[:, batch_idx, beam_idx])
                    # update caches
                    scores_continue.append(next_token_scores[batch_idx, i])
                    if self.ngram > 0:
                        bans = self.cached_beam_ngram_bans[batch_idx][beam_idx].copy()
                        # TODO ngram=1
                        ngram_prefix = tuple(tokens[batch_idx, beam_idx, -(self.ngram - 1):].tolist())
                        bans[ngram_prefix] = bans.get(ngram_prefix, tuple()) + (token_id,)
                        bans_continue.append(bans)
                else:
                    break

            if len(beam_continue) == 0:
                for batch_idx in range(self.batch_size):
                    self._is_done[batch_idx] = True
                return tokens, mems

            beam_continue_batch.append(torch.stack(beam_continue))
            mems_continue_batch.append(torch.stack(mems_contiue, dim=1))
            score_continue_batch.append(scores_continue)
            self.cached_beam_ngram_bans[batch_idx] = bans_continue
        tokens = torch.stack(beam_continue_batch)
        mems = torch.stack(mems_continue_batch, dim=1)
        self.cached_beam_scores = torch.tensor(score_continue_batch, device=logits.device)
        self.length_generated += 1
        for batch_idx in range(self.batch_size):
            if batch_idx >= batch_size:
                self._is_done[batch_idx] = True
            elif (
                len(self.end_beams[batch_idx]) == self.num_beams
                and self.end_beams_penalized_scores[batch_idx][-1]
                >= self.cached_beam_scores[batch_idx].max() / ((5.0 + (seq_len + 1)) / 6) ** self.length_penalty
            ):  # We're done if none of current tokens will better than the worst in end_beams
                self._is_done[batch_idx] = True

        return tokens, mems

    def finalize(self, tokens, mems):
        if self.consider_end:
            batch_size, num_beams = tokens.shape[:2]
            for batch_idx in range(batch_size):
                if not self._is_done[batch_idx]:
                    for i in range(num_beams):
                        self._add_end_beams(self.cached_beam_scores[batch_idx, i], tokens[batch_idx, i], batch_idx)
            mems = None
            ret = self.end_beams[:batch_size]
            self.returned_beam_scores = self.cached_beam_scores[:batch_size]
        else:
            ret = tokens
        self._init_cache()
        return ret, mems
