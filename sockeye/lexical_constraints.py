# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#      http://aws.amazon.com/apache2.0/
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
import pdb

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
        if len(phrase) == 1:
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
                 state: AvoidTrie = None) -> None:

        self.root = avoid_trie
        self.state = state if state else self.root

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
        if word_id in self.state.children:
            return AvoidState(self.root, self.state.step(word_id))
        elif word_id in self.root.children:
            return AvoidState(self.root, self.root.step(word_id))
        elif self.state != self.root:
            return AvoidState(self.root, self.root)
        else:
            return self

    def avoid(self) -> Set[int]:
        """
        Returns a set of word IDs that should be avoided. This includes the set of final states from the
        root node, which are single tokens that must never be generated.

        :return: A set of integers representing words that must not be generated next by this hypothesis.
        """
        return self.root.final() | self.state.final()

    def __str__(self) -> str:
        return str(self.state)


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
        Returns the number of avoid phrases represented in the trie.
        """
        phrase_count = len(self.final_ids)
        for child in self.children.values():
            phrase_count += len(child)
        return phrase_count
    
    # from AvoidTrie -- not sure if needed for positive constraints
    '''
    def add_trie(self,
                 trie: 'IncludeTrie',
                 phrase: Optional[List[int]] = None) -> None:
        self.final_ids |= trie.final()
        for child_id, child in trie.children.items():
            if child_id not in self.children:
                self.children[child_id] = AvoidTrie()
            self.children[child_id].add_trie(child)
    '''
    def add_phrase(self,
                   phrase: List[int]) -> None:
        """
        Recursively adds a phrase to this trie node.

        :param phrase: A list of word IDs to add to this trie node.
        """
        if len(phrase) == 1:
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
                 state: IncludeTrie = None,
                 current_phrase: List[int] = []) -> None:

        self.root = include_trie
        self.state = state if state else self.root
        self.current_phrase = current_phrase  # progress we made satisfying one of the constraints
        self.eos_id = eos_id
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
        if word_id in self.state.final():
            # bingo! we fnished a constraint
            new_current_phrase = self.current_phrase + [word_id]
            new_root = self.root.prune(new_current_phrase)
            if not new_root:
                return IncludeState(None, None)
            next_state = self.state.step(word_id)
            # go further or go home
            if next_state:
                return IncludeState(new_root, self.eos_id, state=next_state, current_phrase=new_current_phrase)
            return IncludeState(new_root, self.eos_id, state=new_root)
        elif word_id in self.state.children.keys():
            return IncludeState(self.root, self.eos_id, state=self.state.step(word_id), current_phrase=self.current_phrase + [word_id])
        elif self.state != self.root:
            return IncludeState(self.root, self.eos_id, state=self.root).consume(word_id)
        return self
    
    def is_valid(self, wordid) -> bool:
        """
        Ensures </s> is only generated when the hypothesis is completed.

        :param wordid: The wordid to validate.
        :return: True if all constraints are already met or the word ID is not the EOS id.
        """
        return not self.root or wordid != self.eos_id or (len(self.root) == 1 and self.eos_id in self.state.final())
    
    def wanted(self) -> Set[int]:
        """
        Return all favorable next words (those that will advance toward fulfilling constraints).
        """
        if not self.root:
            return set()
        return set([i for i in self.state.final()] + [i for i in self.state.children])
    
    def unmet(self) -> int:
        """
        Return the number of unmet constraints.
        """
        return 0 if not self.root else len(self.root)

    def __str__(self) -> str:
        return str(self.state)

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
                 global_include_trie: Optional[IncludeTrie] = None) -> None:

        #print('haha! I have a beam size of ' + str(beam_size) + 'and a batch size of ' + str(batch_size), include_list)
        
        self.states = []    # type: List[IncludeState]
        self.wanted_indices = []
        for _ in range(batch_size * beam_size):
            self.wanted_indices.append([])
        # Store the global trie for each hypothesis
        if global_include_trie is not None:
            for token in global_include_trie:
                self.states = [IncludeState(global_include_trie, eos_id=eos_id)] * batch_size * beam_size


        # Store the sentence-level tries for each item in their portions of the beam
        if include_list is not None:
            if self.states != []:
                for (i, raw_phrases) in enumerate(include_list):
                    for j in range(beam_size):
                        for phrase in raw_phrases:
                            self.states[i*beam_size+j].root.add_phrase(phrase)
            else:
                for (i, raw_phrases) in enumerate(include_list):
                    self.states += [IncludeState(IncludeTrie(raw_phrases), eos_id=eos_id)] * beam_size
        '''
        # initialize wanted        
        for i in range(len(self.states)):
            self.wanted_indices[i].extend(list((self.states[i]).wanted()))
        '''
        self.eos_id = eos_id
    def reorder(self, indices: mx.nd.NDArray) -> None:
        """
        Reorders the avoid list according to the selected row indices.
        This can produce duplicates, but this is fixed if state changes occur in consume().

        :param indices: An mx.nd.NDArray containing indices of hypotheses to select.
        """
        if self.states:
            self.states = [self.states[x] for x in indices]


    def consume(self, word_ids: mx.nd.NDArray) -> None:
        """
        Consumes a word for each trie, updating respective states.

        :param word_ids: The set of word IDs.
        """
        word_ids = word_ids.asnumpy().tolist()
        for i, word_id in enumerate(word_ids):
            self.states[i] = (self.states[i]).consume(word_id)

        #print('now I want:', self.getWanted())
    def getWanted(self) -> (mx.nd.NDArray, mx.nd.NDArray):
        """
        Return the next wanted word id as a 2d list.
        """
        
        wanted_ids = []
        wanted_word_ids = []
            
        #print('num of states:', len(self.states))

        for (slot_id, slot) in enumerate([list(self.states[i].wanted()) for i in range(len(self.states))]):
            for word_id in slot:
                wanted_ids.append(slot_id)
                wanted_word_ids.append(word_id)

        return (mx.nd.array(wanted_ids), mx.nd.array(wanted_word_ids))
    
    def getFinished(self) -> mx.nd.NDArray:
        """
        Return the next wanted word id in a 2d multi-hot matrix.
        """
        result = []
        
        for i in range(len(self.states)):
            result.append(1 if self.states[i].root == None else 0)
        return mx.nd.array(result)

    def getUnmet(self) -> mx.nd.NDArray:
        """
        Return the number of unmet constraints for each tracked hypothesis.
        """
        result = []
        for i in range(len(self.states)):
            result.append(self.states[i].unmet())
        return np.array(result)

'''
def get_bank_sizes(num_constraints: int,
                   beam_size: int,
                   candidate_counts: List[int]) -> List[int]:
    """
    Evenly distributes the beam across the banks, where each bank is a portion of the beam devoted
    to hypotheses having met the same number of constraints, 0..num_constraints.
    After the assignment, banks with more slots than candidates are adjusted.

    :param num_constraints: The number of constraints.
    :param beam_size: The beam size.
    :param candidate_counts: The empirical counts of number of candidates in each bank.
    :return: A distribution over banks.
    """

    num_banks = num_constraints + 1
    bank_size = beam_size // num_banks
    remainder = beam_size - bank_size * num_banks

    # Distribute any remainder to the end
    assigned = [bank_size for x in range(num_banks)]
    assigned[-1] += remainder

    # Now, moving right to left, push extra allocation to earlier buckets.
    # This encodes a bias for higher buckets, but if no candidates are found, space
    # will be made in lower buckets. This may not be the best strategy, but it is important
    # that you start pushing from the bucket that is assigned the remainder, for cases where
    # num_constraints >= beam_size.
    for i in reversed(range(num_banks)):
        freeslots = assigned[i] - candidate_counts[i]
        if freeslots > 0:
            assigned[i] -= freeslots
            assigned[(i - 1) % num_banks] += freeslots

    return assigned


class ConstrainedCandidate:
    """
    Object used to hold candidates for the beam in topk().

    :param row: The row in the scores matrix.
    :param col: The column (word ID) in the scores matrix.
    :param score: the associated accumulated score.
    :param hypothesis: The ConstrainedHypothesis containing information about met constraints.
    """

    __slots__ = ('row', 'col', 'score', 'hypothesis')

    def __init__(self,
                 row: int,
                 col: int,
                 score: float,
                 hypothesis: IncludeState) -> None:
        self.row = row
        self.col = col
        self.score = score
        self.hypothesis = hypothesis

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __str__(self):
        return '({}, {}, {}, {})'.format(self.row, self.col, self.score, self.hypothesis.unmet())
'''

def topk(batch_size: int,
         beam_size: int,
         inactive: mx.nd.NDArray,
         scores: mx.nd.NDArray,
         include_states: IncludeBatch,
         best_ids: mx.nd.NDArray,
         best_word_ids: mx.nd.NDArray,
         seq_scores: mx.nd.NDArray,
         context: mx.context.Context) -> Tuple[np.array, np.array, np.array, AvoidBatch, mx.nd.NDArray]:
    """
    Builds a new topk list such that the beam contains hypotheses having completed different numbers of constraints.
    These items are built from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.

    :param batch_size: The number of segments in the batch.
    :param beam_size: The length of the beam for each segment.
    :param inactive: Array listing inactive rows (shape: (beam_size,)).
    :param scores: The scores array (shape: (beam_size, target_vocab_size)).
    :param include_states: The states of all positively constrained objects.
    :param best_ids: The current list of best hypotheses (shape: (beam_size,)).
    :param best_word_ids: The parallel list of best word IDs (shape: (beam_size,)).
    :param seq_scores: (shape: (beam_size, 1)).
    :param context: The MXNet device context.
    :return: A tuple containing the best hypothesis rows, the best hypothesis words, the scores,
        the updated constrained hypotheses, and the updated set of inactive hypotheses.
    """
    
    wanted_ids, wanted_word_ids = include_states.getWanted() # shape ((batch*beam) * target_vocab)
    #print('wanted', wanted_ids, wanted_word_ids)
    finished_indices = include_states.getFinished() # shape ((batch*beam) * 1)
    
    global_topk = mx.nd.zeros_like(scores, ctx=context)
    
    global_topk[best_ids, best_word_ids] = 1
    global_topk[:, include_states.eos_id] *= finished_indices.as_in_context(context)

    #print("sliced global topk", global_topk[0])
    
    wanted_hyp = mx.nd.zeros_like(scores, ctx=context)
    wanted_hyp[wanted_ids.as_in_context(context), wanted_word_ids.as_in_context(context)] = 1
    
    best_next = mx.nd.zeros_like(scores, ctx=context)
    best_next_idx = mx.nd.NDArray.argmin(scores, axis=1)
    
    best_next[mx.nd.arange(best_next_idx.shape[0], ctx=context), best_next_idx] = 1

    final_hyp = global_topk + wanted_hyp + best_next
    # only keep hypotheses we want to explore
    scores = np.where(final_hyp.asnumpy(), scores.asnumpy(), np.inf)

    final_ids, final_word_ids = np.where(scores != np.inf)
    sent_ids, _ = np.where(scores.reshape((batch_size, -1)) != np.inf)
    final_seq_scores = scores[final_ids, final_word_ids]
    
    unmet = include_states.getUnmet()[final_ids]
    
    big_matrix = np.stack((sent_ids, unmet, final_seq_scores, final_ids, final_word_ids))
    
    #print(final_ids, final_word_ids)
    
    # update unmet
     
    big_matrix[1, :] -= np.isin(big_matrix[-2,:], wanted_ids.asnumpy()).astype(int) * np.isin(big_matrix[-1,:], wanted_word_ids.asnumpy()).astype(int)
    
    big_matrix = big_matrix[:, np.lexsort((big_matrix[2, :], big_matrix[1, :], big_matrix[0, :]))]
    
    def constructParallel(a):
        _, ind = np.unique(a, return_index=True)
        return np.arange(0, len(a)) - ind[np.digitize(a, a[ind]) - 1]

    parallel = constructParallel(big_matrix[1, :])

    big_matrix = np.insert(big_matrix, 1, parallel, axis=0)

    big_matrix = big_matrix[:, np.lexsort((big_matrix[3, :], big_matrix[1, :], big_matrix[0, :]))]

    parallel = constructParallel(big_matrix[0, :])
    
    big_matrix = np.concatenate((big_matrix, parallel.reshape(1, -1)), axis=0)
    #print(big_matrix)
    big_matrix = big_matrix[:, big_matrix[-1, :] < beam_size]

    
    '''
    final_hypotheses = []
    for i in range(len(final_ids)):
        final_hypotheses.append(include_states.states[final_ids[i]].consume(final_word_ids[i]))
    final_hypotheses = np.array(final_hypotheses)
    
    final_order = np.zeros((batch_size*beam_size, ), dtype=np.int32)
    
    
    for sentno in range(batch_size):
        rows = slice(sentno * beam_size, (sentno + 1) * beam_size)

        
        
        
        sorted_index = np.argsort(final_seq_scores[rows])
        # construct the final lists
        final_ids[rows] = final_ids[rows][sorted_index]
        final_word_ids[rows] = final_word_ids[rows][sorted_index]
        final_hypotheses[rows] = final_hypotheses[rows][sorted_index]
        final_seq_scores[rows] = final_seq_scores[rows][sorted_index]


        
        
        
        
        num_constraints = max([state.unmet() for state in include_states.states[rows]])

        counts = [0 for x in range(num_constraints + 1)]
        for hypo in final_hypotheses[rows]:
            counts[hypo.unmet()] += 1

        # Adjust allocated bank sizes if there are too few candidates in any of them
        bank_sizes = get_bank_sizes(num_constraints, beam_size, counts)

        # Sort the candidates into the allocated banks
        pruned_candidates = []  # type: List[ConstrainedCandidate]
        for i, hypo in enumerate(final_hypotheses[rows]):
            bank = hypo.unmet()

            if bank_sizes[bank] > 0:
                pruned_candidates.append(i)
                bank_sizes[bank] -= 1

        inactive[rows][:len(pruned_candidates)] = 0

        # Pad the beam so array assignment still works
        if len(pruned_candidates) < beam_size:
            inactive[rows][len(pruned_candidates):] = 1
            pruned_candidates += [pruned_candidates[len(pruned_candidates) - 1]] * (beam_size - len(pruned_candidates))
        
        
        final_order[rows] = np.array(pruned_candidates) + rows.start
    
    final_ids = final_ids[final_order]
    final_word_ids = final_word_ids[final_order]
    final_seq_scores = final_seq_scores[final_order]
    '''
    #print(big_matrix.shape)

    inactive[:big_matrix.shape[1]] = 0
    if big_matrix.shape[1] < batch_size * beam_size:
        padding = np.zeros((big_matrix[0], batch_size * beam_size - big_matrix.shape[1]))
        big_matrix = np.concatenate((big_matrix, padding), axis=1)
        inactive[big_matrix.shape[1]:] = 1
        

    best_ids[:] = (big_matrix[-3, :]).reshape(-1,)
    best_word_ids[:] = (big_matrix[-2, :]).reshape(-1,)
    seq_scores[:] = (big_matrix[-4, :]).reshape(-1, 1)

    include_states.reorder(best_ids.asnumpy())
    include_states.consume(best_word_ids)
   
   
    return best_ids, best_word_ids, seq_scores, include_states, inactive


def _topk(beam_size: int,
          inactive: mx.nd.NDArray,
          scores: mx.nd.NDArray,
          include_states: List[IncludeState],
          best_ids: mx.nd.NDArray,
          best_word_ids: mx.nd.NDArray,
          sequence_scores: mx.nd.NDArray,
          context: mx.context.Context) -> Tuple[np.array, np.array, np.array, List[AvoidState], mx.nd.NDArray]:
    """
    Builds a new topk list such that the beam contains hypotheses having completed different numbers of constraints.
    These items are built from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.

    :param beam_size: The length of the beam for each segment.
    :param inactive: Array listing inactive rows (shape: (beam_size,)).
    :param scores: The scores array (shape: (beam_size, target_vocab_size)).
    :param include_states: The list of include states.
    :param best_ids: The current list of best hypotheses (shape: (beam_size,)).
    :param best_word_ids: The parallel list of best word IDs (shape: (beam_size,)).
    :param sequence_scores: (shape: (beam_size, 1)).
    :param context: The MXNet device context.
    :return: A tuple containing the best hypothesis rows, the best hypothesis words, the scores,
        the updated constrained hypotheses, and the updated set of inactive hypotheses.
    """

    num_constraints = max([state.unmet() for state in include_states])

    candidates = set()
    # (1) Add all of the top-k items (which were passed) in as long as they pass the constraints
    for row, col, seq_score in zip(best_ids, best_word_ids, sequence_scores):
        row = int(row.asscalar())
        col = int(col.asscalar())
        seq_score = float(seq_score.asscalar())
        if include_states[row].is_valid(col):
            #print("Huh! I'm consuming " + str(col) + " in _topk")
            new_item = include_states[row].consume(col)
            #print("yum yum. Now I want:" + str(new_item.wanted()))
            cand = ConstrainedCandidate(row, col, seq_score, new_item)
            candidates.add(cand)
            

    # For each hypothesis, we add (2) all the constraints that could follow it and
    # (3) the best item (constrained or not) in that row
    best_next = mx.nd.NDArray.argmin(scores, axis=1)
    for row in range(beam_size):
        if inactive[row]:
            continue

        hyp = include_states[row]
        
        # (2) add all the constraints that could extend this
        nextones = hyp.wanted()
        #print('hyp num ' + str(row) + ' says: I want ' + str(nextones))
        # (3) add the single-best item after this (if it's valid)
        col = int(best_next[row].asscalar())
        if hyp.is_valid(col):
            nextones.add(col)

        # Now, create new candidates for each of these items
        for col in nextones:
            #print("Huh! Hpy num " + str(row) +" is consuming " + str(col) + " in _topk 2/3")
            new_item = hyp.consume(col)
            #print("yum yum. Now I want:" + str(new_item.wanted()))
            score = scores[row, col].asscalar()
            cand = ConstrainedCandidate(row, col, score, new_item)
            candidates.add(cand)

    # Sort the candidates. After allocating the beam across the banks, we will pick the top items
    # for each bank from this list
    sorted_candidates = sorted(candidates, key=attrgetter('score'))

    # The number of hypotheses in each bank
    counts = [0 for x in range(num_constraints + 1)]
    for cand in sorted_candidates:
        counts[cand.hypothesis.unmet()] += 1
    #print('counts b4')
    #print(counts)

    # Adjust allocated bank sizes if there are too few candidates in any of them
    bank_sizes = get_bank_sizes(num_constraints, beam_size, counts)

    #print('counts after')
    #print(bank_sizes)
    # Sort the candidates into the allocated banks
    pruned_candidates = []  # type: List[ConstrainedCandidate]
    for i, cand in enumerate(sorted_candidates):
        bank = cand.hypothesis.unmet()

        if bank_sizes[bank] > 0:
            pruned_candidates.append(cand)
            bank_sizes[bank] -= 1

    inactive[:len(pruned_candidates)] = 0

    # Pad the beam so array assignment still works
    if len(pruned_candidates) < beam_size:
        inactive[len(pruned_candidates):] = 1
        pruned_candidates += [pruned_candidates[len(pruned_candidates) - 1]] * (beam_size - len(pruned_candidates))

    return (np.array([x.row for x in pruned_candidates]),
            np.array([x.col for x in pruned_candidates]),
            np.array([[x.score] for x in pruned_candidates]),
            [x.hypothesis for x in pruned_candidates],
            inactive)


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
