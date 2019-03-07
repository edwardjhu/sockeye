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

import json
from unittest.mock import Mock

import mxnet as mx
import numpy as np
import pytest
from math import ceil

from sockeye.data_io import get_tokens, tokens2ids, strids2ids
from sockeye.vocab import build_vocab, reverse_vocab
from sockeye.lexical_constraints import topk, AvoidBatch, AvoidState, AvoidTrie, IncludeBatch, IncludeState, IncludeTrie
from sockeye.inference import Translator

BOS_ID = 2
EOS_ID = 3

def mock_translator(num_source_factors: int):
    t_mock = Mock(Translator)
    t_mock.num_source_factors = num_source_factors
    return t_mock


"""
Test how the banks are allocated. Given a number of constraints (C), the beam size (k)
a number of candidates for each bank [0..C], return the allocation of the k spots of the
beam to the banks.
"""

test_avoid_list_data = [ (["this", "that", "this bad phrase", "this bad phrase that is longer"]),
                         ([]),
                         (["a really bad phrase"]),
                         (["lightning crashes", "lightning bugs"]) ]

"""
Ensures that the avoid trie is built correctly.
"""
@pytest.mark.parametrize("raw_phrase_list", test_avoid_list_data)
def test_avoid_list_trie(raw_phrase_list):
    # Make sure the trie reports the right size
    raw_phrase_list = [list(get_tokens(phrase)) for phrase in raw_phrase_list]
    root_trie = AvoidTrie(raw_phrase_list)
    assert len(root_trie) == len(raw_phrase_list)

    # The last word of the phrase should be in the final() list after walking through the
    # first len(phrase) - 1 words
    for phrase in raw_phrase_list:
        trie = root_trie
        for word in phrase[:-1]:
            trie = trie.step(word)
        assert phrase[-1] in trie.final()

    oov_id = 8239
    assert root_trie.step(oov_id) is None

    root_trie.add_phrase([oov_id, 17])
    assert root_trie.step(oov_id) is not None
    assert 17 in root_trie.step(oov_id).final()

"""
Ensure that state management works correctly.
State management records where we are in the trie.
"""
@pytest.mark.parametrize("raw_phrase_list", test_avoid_list_data)
def test_avoid_list_state(raw_phrase_list):
    raw_phrase_list = [list(get_tokens(phrase)) for phrase in raw_phrase_list]
    root_trie = AvoidTrie(raw_phrase_list)

    root_state = AvoidState(root_trie)
    oov_id = 83284

    # Consuming an OOV ID from the root state should return the same state
    assert root_state == root_state.consume(oov_id)

    # The avoid lists should also be the same
    assert root_state.avoid() == root_state.consume(oov_id).avoid()

    root_ids_to_avoid = root_state.avoid()
    for phrase in raw_phrase_list:
        state = root_state
        for word in phrase[:-1]:
            state = state.consume(word)

        # The last word of the phrase should be in the avoid list
        assert phrase[-1] in state.avoid()

        # Trying to advance on an OOV from inside a multi-word constraint should return a new state
        if len(phrase) > 1:
            new_state = state.consume(oov_id)
            assert new_state != state

"""
Ensure that managing states for a whole batch works correctly.

Here we pass a list of global phrases to avoid, followed by a sentence-level list. Then, the batch, beam, and vocabulary
sizes. Finally, the prefix that the decoder is presumed to have seen, and the list of vocab-transformed IDs we should
expect (a function of the vocab size) that should be blocked.
"""
@pytest.mark.parametrize("global_raw_phrase_list, raw_phrase_list, batch_size, beam_size, prefix, expected_avoid", [
    (['5 6 7 8'], None, 1, 3, '17', []),
    (['5 6 7 12'], None, 1, 4, '5 6 7', [(0, 12), (1, 12), (2, 12), (3, 12)]),
    (['5 6 7 8', '9'], None, 1, 2, '5 6 7', [(0, 8), (0, 9), (1, 8), (1, 9)]),
    (['5 6 7 8', '13'], [[[10]]], 1, 2, '5 6 7', [(0, 8), (0, 10), (0, 13), (1, 8), (1, 10), (1, 13)]),
    # first two hypotheses blocked on 19 (= 19 and 119), next two on 20 (= 220 and 320)
    (None, [[[19]], [[20]]], 2, 2, '', [(0, 19), (1, 19), (2, 20), (3, 20)]),
    # same, but also add global block list to each row
    (['74'], [[[19]], [[20]]], 2, 2, '', [(0, 19), (0, 74), (1, 19), (1, 74), (2, 20), (2, 74), (3, 20), (3, 74)]),
])
def test_avoid_list_batch(global_raw_phrase_list, raw_phrase_list, batch_size, beam_size, prefix, expected_avoid):

    global_avoid_trie = None
    if global_raw_phrase_list:
        global_raw_phrase_list = [list(strids2ids(get_tokens(phrase))) for phrase in global_raw_phrase_list]
        global_avoid_trie = AvoidTrie(global_raw_phrase_list)

    avoid_batch = AvoidBatch(batch_size, beam_size, avoid_list=raw_phrase_list, global_avoid_trie=global_avoid_trie)

    for word_id in strids2ids(get_tokens(prefix)):
        avoid_batch.consume(mx.nd.array([word_id] * (batch_size * beam_size)))

    avoid = [(x, y) for x, y in zip(*avoid_batch.avoid())]
    assert set(avoid) == set(expected_avoid)
