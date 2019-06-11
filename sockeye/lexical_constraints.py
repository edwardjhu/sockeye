# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import copy
import logging
from operator import attrgetter
from typing import Dict, List, Optional, Tuple, Set

import mxnet as mx
import numpy as np

from . import constants as C

logger = logging.getLogger(__name__)

# Represents a list of raw constraints for a sentence. Each constraint is a list of target-word IDs.
RawConstraintList = List[List[int]]


class AvoidTrie:
    """
    Represents a set of phrasal constraints for an input sentence.
    These are organized into a trie.
    """
    def __init__(self,
                 raw_phrases: Optional[RawConstraintList] = None) -> None:
        self.final_ids = set()  # type: Set[int]
        self.children = {}  # type: Dict[int,'AvoidTrie']

        if raw_phrases:
            for phrase in raw_phrases:
                self.add_phrase(phrase)

    def __str__(self) -> str:
        s = '({}'.format(list(self.final_ids))
        for child_id in self.children.keys():
            s += ' -> {} {}'.format(child_id, self.children[child_id])
        s += ')'
        return s

    def __len__(self) -> int:
        """
        Returns the number of avoid phrases represented in the trie.
        """
        phrase_count = len(self.final_ids)
        for child in self.children.values():
            phrase_count += len(child)
        return phrase_count

    def add_trie(self,
                 trie: 'AvoidTrie',
                 phrase: Optional[List[int]] = None) -> None:
        self.final_ids |= trie.final()
        for child_id, child in trie.children.items():
            if child_id not in self.children:
                self.children[child_id] = AvoidTrie()
            self.children[child_id].add_trie(child)

    def add_phrase(self,
                   phrase: List[int]) -> None:
        """
        Recursively adds a phrase to this trie node.

        :param phrase: A list of word IDs to add to this trie node.
        """
        if len(phrase) == 0:
            return
        elif len(phrase) == 1:
            self.final_ids.add(phrase[0])
        else:
            next_word = phrase[0]
            if next_word not in self.children:
                self.children[next_word] = AvoidTrie()
            self.step(next_word).add_phrase(phrase[1:])

    def step(self, word_id: int) -> Optional['AvoidTrie']:
        """
        Returns the child node along the requested arc.

        :param phrase: A list of word IDs to add to this trie node.
        :return: The child node along the requested arc, or None if no such arc exists.
        """
        return self.children.get(word_id, None)

    def final(self) -> Set[int]:
        """
        Returns the set of final ids at this node.

        :return: The set of word IDs that end a constraint at this state.
        """
        return self.final_ids


class AvoidState:
    """
    Represents the state of a hypothesis in the AvoidTrie.
    The offset is used to return actual positions in the one-dimensionally-resized array that
    get set to infinity.

    :param avoid_trie: The trie containing the phrases to avoid.
    :param state: The current state (defaults to root).
    """
    def __init__(self,
                 avoid_trie: AvoidTrie,
                 states: List[AvoidTrie] = None) -> None:

        self.root = avoid_trie
        self.states = states if states else []

    def consume(self, word_id: int) -> 'AvoidState':
        """
        Consumes a word, and updates the state based on it. Returns new objects on a state change.

        The next state for a word can be tricky. Here are the cases:
        (1) If the word is found in our set of outgoing child arcs, we take that transition.
        (2) If the word is not found, and we are not in the root state, we need to reset.
            This means we pretend we were in the root state, and see if we can take a step
        (3) Otherwise, if we are not already in the root state (i.e., we were partially through
            the trie), we need to create a new object whose state is the root state
        (4) Finally, if we couldn't advance and were already in the root state, we can reuse
            this object.

        :param word_id: The word that was just generated.
        """
        new_states = []
        for state in self.states:
            if word_id in state.children:
                new_states.append(state.step(word_id))
        if word_id in self.root.children:
            new_states.append(self.root.step(word_id))
        if len(self.states) == 0 and len(new_states) == 0:
            return self
        return AvoidState(self.root, new_states)

    def avoid(self) -> Set[int]:
        """
        Returns a set of word IDs that should be avoided. This includes the set of final states from the
        root node, which are single tokens that must never be generated.

        :return: A set of integers representing words that must not be generated next by this hypothesis.
        """
        return self.root.final().union(*[state.final() for state in self.states])

    def __str__(self) -> str:
        return str([str(state) for state in self.states])


class AvoidBatch:
    """
    Represents a set of phrasal constraints for all items in the batch.
    For each hypotheses, there is an AvoidTrie tracking its state.

    :param batch_size: The batch size.
    :param beam_size: The beam size.
    :param avoid_list: The list of lists (raw phrasal constraints as IDs, one for each item in the batch).
    :param global_avoid_trie: A translator-level vocabulary of items to avoid.
    """
    def __init__(self,
                 batch_size: int,
                 beam_size: int,
                 avoid_list: Optional[List[RawConstraintList]] = None,
                 global_avoid_trie: Optional[AvoidTrie] = None) -> None:

        self.global_avoid_states = []  # type: List[AvoidState]
        self.local_avoid_states = []  # type: List[AvoidState]

        # Store the global trie for each hypothesis
        if global_avoid_trie is not None:
            self.global_avoid_states = [AvoidState(global_avoid_trie)] * batch_size * beam_size

        # Store the sentence-level tries for each item in their portions of the beam
        if avoid_list is not None:
            for raw_phrases in avoid_list:
                self.local_avoid_states += [AvoidState(AvoidTrie(raw_phrases))] * beam_size

    def reorder(self, indices: mx.nd.NDArray) -> None:
        """
        Reorders the avoid list according to the selected row indices.
        This can produce duplicates, but this is fixed if state changes occur in consume().

        :param indices: An mx.nd.NDArray containing indices of hypotheses to select.
        """
        if self.global_avoid_states:
            self.global_avoid_states = [self.global_avoid_states[x] for x in indices.asnumpy()]

        if self.local_avoid_states:
            self.local_avoid_states = [self.local_avoid_states[x] for x in indices.asnumpy()]

    def consume(self, word_ids: mx.nd.NDArray) -> None:
        """
        Consumes a word for each trie, updating respective states.

        :param word_ids: The set of word IDs.
        """
        word_ids = word_ids.asnumpy().tolist()
        for i, word_id in enumerate(word_ids):
            if self.global_avoid_states:
                self.global_avoid_states[i] = self.global_avoid_states[i].consume(word_id)
            if self.local_avoid_states:
                self.local_avoid_states[i] = self.local_avoid_states[i].consume(word_id)

    def avoid(self) -> Tuple[Tuple[int], Tuple[int]]:
        """
        Assembles a list of per-hypothesis words to avoid. The indices are (x, y) pairs into the scores
        array, which has dimensions (beam_size, target_vocab_size). These values are then used by the caller
        to set these items to np.inf so they won't be selected. Words to be avoided are selected by
        consulting both the global trie of phrases and the sentence-specific one.

        :return: Two lists of indices: the x coordinates and y coordinates.
        """
        to_avoid = set()  # type: Set[Tuple[int, int]]
        for i, state in enumerate(self.global_avoid_states):
            for word_id in state.avoid():
                if word_id > 0:
                    to_avoid.add((i, word_id))
        for i, state in enumerate(self.local_avoid_states):
            for word_id in state.avoid():
                if word_id > 0:
                    to_avoid.add((i, word_id))

        return tuple(zip(*to_avoid))  # type: ignore




# Positive constraints with Trie
# If this is mostly compatible with AvoidTrie, I'll merge the two
class IncludeTrie:
    """
    Represents a set of phrasal constraints to include for an input sentence.
    These are organized into a trie.
    This is similar to AvoidTrie but has special operations.
    """
    def __init__(self,
                 raw_phrases: Optional[RawConstraintList] = None) -> None:
        self.final_ids = set()  # type: Set[int]
        self.children = {}  # type: Dict[int,'AvoidTrie']
        if raw_phrases:
            for phrase in raw_phrases:
                self.add_phrase(phrase)

    def __str__(self) -> str:
        s = '({}'.format(list(self.final_ids))
        for child_id in self.children.keys():
            s += ' -> {} {}'.format(child_id, self.children[child_id])
        s += ')'
        return s

    def __len__(self) -> int:
        """
        Returns the number of include phrases represented in the trie.
        """
        phrase_count = len(self.final_ids) + len(self.children.keys())
        for child in self.children.values():
            phrase_count += len(child)
        return phrase_count
    
    def add_phrase(self,
                   phrase: List[int]) -> None:
        """
        Recursively adds a phrase to this trie node.

        :param phrase: A list of word IDs to add to this trie node.
        """
        if len(phrase) == 0:
            return
        elif len(phrase) == 1:
            self.final_ids.add(phrase[0])
        else:
            next_word = phrase[0]
            if next_word not in self.children:
                self.children[next_word] = IncludeTrie()
            self.step(next_word).add_phrase(phrase[1:])

    def step(self, word_id: int) -> Optional['IncludeTrie']:
        """
        Returns the child node along the requested arc.

        :param phrase: A list of word IDs to add to this trie node.
        :return: The child node along the requested arc, or None if no such arc exists.
        """
        return self.children.get(word_id, None)
    
    def prune(self, phrase) -> Optional['IncludeTrie']:
        """
        Create a copy and prune.
        
        :param phrase: A list of word IDs, the path of which will be elimated from the current Trie
        """
        to_prune = copy.deepcopy(self)
        if to_prune._prune(phrase):
            return None
        return to_prune

    
    def _prune(self, phrase) -> bool:
        """
        Eliminate the path to the specified phrase and return if we have satisfied all the constraints.
        
        :param phrase: A list of word IDs, the path of which will be elimated from the current Trie
        """
        # if we just satisfied a one-token constraint
        if len(phrase) == 0:
            # do nothing -- this should never happen!
            pass
        elif len(phrase) == 1:
            if phrase[0] in self.final_ids:
                self.final_ids.remove(phrase[0])
        else:
            next_step = self.step(phrase[0])
            if next_step:
                if next_step._prune(phrase[1:]):
                    self.children.pop(phrase[0], None)
        # check if we have satisfied all constraints
        return (len(self.final_ids) == 0) and (len(self.children) == 0)

    def final(self) -> Set[int]:
        """
        Returns the set of final ids at this node.

        :return: The set of word IDs that end a constraint at this state.
        """
        return self.final_ids

class IncludeState:
    """
    Represents the state of a hypothesis in the IncludeTrie.
    This can determine how far a hypothesis is from finishing all constraints.

    :param include_trie: The trie containing the phrases to include.
    :param state: The current state (defaults to root).
    :param eos_id: The end-of-sentence ID.
    """
    def __init__(self,
                 include_trie: IncludeTrie,
                 eos_id: int,
                 states: List[IncludeTrie] = [],
                 current_phrase: List[List[int]] = [],
                 partial: List[int] = []) -> None:

        self.root = include_trie
        self.states = states
        self.current_phrase = current_phrase  # progress we made satisfying one of the constraints
        self.eos_id = eos_id
        self.partial = partial
    def consume(self, word_id: int) -> 'IncludeState':
        """
        Consumes a word, and updates the state based on it. Returns new objects on a state change.

        The next state for a word can be the following cases:
        (1) If the this finishes a constraint, we prune the branch.
        (2) If the word is found in our set of outgoing child arcs, we take that transition.
        (3) If the word is not found, we need to reset to root (or stay at root if already) and then try again.
        (4) Otherwise, just return self

        :param word_id: The word that was just generated.
        """
        # we are done
        if self.root == None:
            return self
        new_states = []
        new_phrase = []
        new_partial = []
        new_root = self.root
        for i, state in enumerate(self.states):
            if word_id in state.final():
                # bingo! we finished a constraint
                new_root = new_root.prune(self.current_phrase[i] + [word_id])
                if not new_root:
                    return IncludeState(None, self.eos_id)
            if word_id in state.children:
                new_states.append(state.step(word_id))
                new_phrase.append(self.current_phrase[i] + [word_id])
                new_partial.append(self.partial[i] + 1)
        
        if word_id in new_root.final():
            new_root = new_root.prune([word_id])
            if not new_root:
                return IncludeState(None, self.eos_id)
        if word_id in new_root.children:
            new_states.append(new_root.step(word_id))
            new_phrase.append([word_id])
            new_partial.append(1)
        
        return IncludeState(new_root, self.eos_id, states=new_states, partial=new_partial, current_phrase=new_phrase)
    
    def is_valid(self, wordid) -> bool:
        """
        Ensures </s> is only generated when the hypothesis is completed.

        :param wordid: The wordid to validate.
        :return: True if all constraints are already met or the word ID is not the EOS id.
        """
        return not self.root or wordid != self.eos_id or (self.unmet() == 1 and self.eos_id in set().union(*[state.final() for state in self.states]))
    
    def wanted(self) -> Set[int]:
        """
        Return all favorable next words (those that will advance toward fulfilling constraints).
        """
        if not self.root:
            return set()
        return self.root.final().union(*[state.final() | state.children.keys() for state in self.states]) | self.root.children.keys()
    
    def unmet(self) -> int:
        """
        Return the number of unmet constraints.
        """
        return 0 if not self.root else (len(self.root) if len(self.partial) == 0 else (len(self.root) - max(self.partial)))

    def __str__(self) -> str:
        return str([str(state) for state in self.states])

class IncludeBatch:
    """
    Represents a set of phrasal constraints for all items in the batch.
    For each hypotheses, there is an IncludeTrie tracking its state.

    :param batch_size: The batch size.
    :param beam_size: The beam size.
    :param include_list: The list of lists (raw phrasal constraints as IDs, one for each item in the batch).
    :param global_include_trie: A translator-level vocabulary of items to include.
    :param eos_id: The end-of-sentence ID.
    """
    def __init__(self,
                 batch_size: int,
                 beam_size: int,
                 eos_id: int,
                 include_list: Optional[List[RawConstraintList]] = None,
                 ctx = None) -> None:
        self.states = []    # type: List[IncludeState]
        self.wanted_indices = []
        for _ in range(batch_size * beam_size):
            self.wanted_indices.append([])

        # Store the sentence-level tries for each item in their portions of the beam
        if include_list is not None:
            for (i, raw_phrases) in enumerate(include_list):
                if raw_phrases:
                    self.states += [IncludeState(IncludeTrie(raw_phrases), eos_id=eos_id)] * beam_size
                else:
                    self.states += [IncludeState(None, eos_id=eos_id)] * beam_size

        self.context = ctx
        self.eos_id = eos_id
    def reorder(self, indices: mx.nd.NDArray) -> None:
        """
        Reorders the avoid list according to the selected row indices.
        This can produce duplicates, but this is fixed if state changes occur in consume().

        :param indices: An mx.nd.NDArray containing indices of hypotheses to select.
        """
        if self.states:
            self.states = [self.states[x.asscalar()] for x in indices]


    def consume(self, word_ids: mx.nd.NDArray) -> None:
        """
        Consumes a word for each trie, updating respective states.

        :param word_ids: The set of word IDs.
        """
        word_ids = word_ids.asnumpy().tolist()
        for i, word_id in enumerate(word_ids):
            self.states[i] = (self.states[i]).consume(word_id)

    def get_wanted(self) -> (mx.nd.NDArray, mx.nd.NDArray):
        """
        Return the next wanted word id as a 2d list.
        """
        wanted_ids = []
        wanted_word_ids = []
        for i in range(len(self.states)):
            for word_id in self.states[i].wanted():
                wanted_ids.append(i)
                wanted_word_ids.append(word_id)
        return (mx.nd.array(wanted_ids, ctx=self.context, dtype='int32'), mx.nd.array(wanted_word_ids, ctx=self.context, dtype='int32'))
    
    def get_all_met(self) -> mx.nd.NDArray:
        """
        Return the next wanted word id in a 2d multi-hot matrix.
        """
        result = []
        
        for i in range(len(self.states)):
            result.append(1 if self.states[i].root == None else 0)
        return mx.nd.array(result, ctx=self.context, dtype='int8')

    def get_unmet(self) -> mx.nd.NDArray:
        """
        Return the number of unmet constraints for each tracked hypothesis.
        """
        result = []
        for i in range(len(self.states)):
            result.append(self.states[i].unmet())
        return mx.nd.array(result, ctx=self.context)

def topk(timestep: int,
         batch_size: int,
         beam_size: int,
         inactive: mx.nd.NDArray,
         scores: mx.nd.NDArray,
         include_states: IncludeBatch,
         best_ids: mx.nd.NDArray,
         best_word_ids: mx.nd.NDArray,
         seq_scores: mx.nd.NDArray,
         context: mx.context.Context,
         beam_prune: float,
         finished: mx.nd.NDArray) -> Tuple[np.array, np.array, np.array, AvoidBatch, mx.nd.NDArray]:
    """
    Builds a new topk list such that the beam contains hypotheses having completed different numbers of constraints.
    These items are built from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.

    :param timestep: The current decoder timestep.
    :param batch_size: The number of segments in the batch.
    :param beam_size: The length of the beam for each segment.
    :param inactive: Array listing inactive rows (shape: (beam_size,)).
    :param scores: The scores array (shape: (beam_size, target_vocab_size)).
    :param include_states: The states of all positively constrained objects.
    :param best_ids: The current list of best hypotheses (shape: (beam_size,)).
    :param best_word_ids: The parallel list of best word IDs (shape: (beam_size,)).
    :param seq_scores: (shape: (beam_size, 1)).
    :return: A tuple containing the best hypothesis rows, the best hypothesis words, the scores,
        the updated constrained hypotheses, and the updated set of inactive hypotheses.
    """
    # Source 1: Global Top K
    # Source 2: Words that advance the Tries
    # Source 3: Best word per hypothesis 
    wanted_ids, wanted_word_ids = include_states.get_wanted()
    all_met_indices = include_states.get_all_met()
    best_next_idx = mx.nd.NDArray.argmin(scores, axis=1)
    
    all_ids = mx.nd.concat(best_ids, mx.nd.arange(best_next_idx.shape[0], ctx=context, dtype='int32'), dim=0)
    all_word_ids = mx.nd.concat(best_word_ids, mx.nd.cast(best_next_idx, dtype='int32'), dim=0)
    # Only concat wanted_ids when it's not empty
    if wanted_ids.shape[0] > 0:
        all_ids = mx.nd.concat(all_ids, wanted_ids, dim=0)
        all_word_ids = mx.nd.concat(all_word_ids, wanted_word_ids, dim=0)
    all_ind = mx.nd.stack(all_ids, all_word_ids).asnumpy().T
    
    # Get rid of duplicate rows
    sorted_idx = np.lexsort(all_ind.T)
    sorted_data =  all_ind[sorted_idx, :]
    row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0), 1))
    all_ind = sorted_data[row_mask]

    # Deactivate inactive hypotheses
    all_ind = all_ind[np.isin(all_ind[:, 0], np.flatnonzero(inactive.asnumpy()-1)), :]
    
    # Deactivate eos_id on unfinished hypotheses
    all_ind = all_ind[np.logical_or(all_ind[:, 1] != include_states.eos_id, (all_met_indices.asnumpy()[all_ind[:, 0]]).astype(bool)), :]

    final_ids, final_word_ids = all_ind[:, 0].reshape(-1), all_ind[:, 1].reshape(-1)
    final_sent_ids = np.floor(final_ids / beam_size)
    final_seq_scores = (scores[mx.nd.array(final_ids, ctx=context), mx.nd.array(final_word_ids, ctx=context)]).asnumpy()

    # Remove hypotheses with a seq score of inf
    valid_hypo = final_seq_scores != np.inf
    final_ids = final_ids[valid_hypo]
    final_sent_ids = final_sent_ids[valid_hypo]
    final_word_ids = final_word_ids[valid_hypo]
    final_seq_scores = final_seq_scores[valid_hypo]

    # Create the big_matrix
    unmet = include_states.get_unmet().asnumpy()[final_ids]
    big_matrix = np.stack((final_sent_ids, unmet, final_seq_scores, final_ids, final_word_ids))
    
    def constructParallel(arr):
        """
        Construct a step array like [0, 1, 0, 1, 2] given a non-decreasing array ([0, 0, 1, 1, 1])
        This is equivalent to:
        
        result = [0]
        prev = arr[0]
        for num in arr[1:]:
            if num == prev:
                result.append(result[-1] + 1)
            else:
                result.append(0)
                prev = num
        return result
        """
        _, ind = np.unique(arr, return_index=True)
        return (np.arange(0, len(arr)) - ind[np.digitize(arr, arr[ind]) - 1]).astype(int)

    # Update unmet since some hypotheses will use tokens that advance the Trie; we want to use the new count
    if wanted_ids.shape[0] > 0:
        bm_entries = mx.nd.array(np.stack((final_ids, final_word_ids)).transpose(), ctx=context, dtype='int32')
        _bm_entries = bm_entries.reshape((bm_entries.shape[0], 1, bm_entries.shape[1]))
        wanted_entries = mx.nd.stack(wanted_ids, wanted_word_ids).transpose()
        big_matrix[1, :] -= mx.nd.sum(mx.nd.prod(wanted_entries == _bm_entries, axis=-1), axis=1).asnumpy()

    # Sort big_matrix
    big_matrix = big_matrix[:, np.lexsort((big_matrix[2, :], big_matrix[1, :], big_matrix[0, :]))]
   
    # Beam prune
    if beam_prune > 0 and np.any(big_matrix[1, :] == 0):
        all_met_entries = big_matrix[0, big_matrix[1, :] == 0].astype(int)
        seq_mat = np.full((batch_size, np.bincount(all_met_entries).max()), np.inf)
        seq_mat[all_met_entries, \
                 constructParallel(all_met_entries)] = big_matrix[2, big_matrix[1, :] == 0]
        best_all_met = np.amin(seq_mat, axis=1).reshape(-1)
        big_matrix = big_matrix[:, (big_matrix[2, :] - best_all_met[(big_matrix[0, :]).astype(int)]) < beam_prune]
    
    # core DBA algorithm
    parallel_to_unmet = constructParallel(big_matrix[1, :] + big_matrix[0, :] * (np.max(big_matrix[1, :])+1)) # make sure the orginal array is non-decreasing

    # Smart beam allocation
    #finished_count = mx.nd.sum((finished * (-inactive + 1)).reshape((batch_size, -1)), axis=1).asnumpy()
    #parallel_to_unmet += (finished_count[(big_matrix[0, :]).astype(int)] * big_matrix[1, :]).astype(int)

    big_matrix = big_matrix[:, np.lexsort((big_matrix[2, :], big_matrix[1, :], parallel_to_unmet, big_matrix[0, :]))] # sort columns according to specific rows
    
    # Get rid of hypotheses that won't fit
    parallel_to_sentid = constructParallel(big_matrix[0, :]) # number each hypthesis from the same sent
    big_matrix = big_matrix[:, parallel_to_sentid < beam_size] 
    to_keep_ids = parallel_to_sentid[parallel_to_sentid < beam_size] + big_matrix[0, :] * beam_size
    
    # Update inactive accordingly
    inactive[:] = 1
    inactive[to_keep_ids] = 0
    
    
    # Fill in the gaps
    if big_matrix.shape[1] < batch_size * beam_size:
        final_matrix = np.zeros((big_matrix.shape[0], batch_size * beam_size))
        final_matrix[2, :] = np.inf # mark seq scores as inf for fillers
        final_matrix[:, to_keep_ids.astype(int)] = big_matrix
        big_matrix = final_matrix
    
    best_ids[:] = (big_matrix[3, :]).reshape(-1,) # retrieve best_ids from big_matrix
    best_word_ids[:] = (big_matrix[4, :]).reshape(-1,) # retrieve best_word_ids from big_matrix
    seq_scores[:] = (big_matrix[2, :]).reshape(-1, 1) # retrieve seq_scores from big_matrix
    
    # update include_states (IncludeBatch)
    include_states.reorder(best_ids)
    include_states.consume(best_word_ids)
   
    return best_ids, best_word_ids, seq_scores, include_states, inactive

def main(args):
    """
    Usage: python3 -m sockeye.lexical_constraints [--bpe BPE_MODEL]

    Reads sentences and constraints on STDIN (tab-delimited) and generates the JSON format
    that can be used when passing `--json-input` to sockeye.translate. It supports both positive
    constraints (phrases that must appear in the output) and negative constraints (phrases that
    must *not* appear in the output).

    e.g.,

        echo -e "Das ist ein Test .\tThis is\ttest" | python3 -m sockeye.lexical_constraints

    will produce the following JSON object:

        { "text": "Das ist ein Test .", "constraints": ["This is", "test"] }

    If you pass `--avoid` to the script, the constraints will be generated as negative constraints, instead:

        echo -e "Das ist ein Test .\tThis is\ttest" | python3 -m sockeye.lexical_constraints --avoid

    will produce the following JSON object (note the new keyword):

        { "text": "Das ist ein Test .", "avoid": ["This is", "test"] }

    Make sure you apply all preprocessing (tokenization, BPE, etc.) to both the source and the target-side constraints.
    You can then translate this object by passing it to Sockeye on STDIN as follows:

        python3 -m sockeye.translate -m /path/to/model --json-input --beam-size 20 --beam-prune 20

    Note the recommended Sockeye parameters. Beam pruning isn't needed for negative constraints.
    """
    import sys
    import json

    for line in sys.stdin:
        line = line.rstrip()

        # Constraints are in fields 2+
        source, *restrictions = line.split('\t')

        obj = {'text': source}
        constraints = []
        avoid_list = []
        for item in restrictions:
            if args.avoid:
                avoid_list.append(item)
            else:
                constraints.append(item)

        if constraints:
            obj['constraints'] = constraints
        if avoid_list:
            obj['avoid'] = avoid_list

        print(json.dumps(obj, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--avoid', action='store_true', help='Constraints are negative constraints')
    args = parser.parse_args()

    main(args)
