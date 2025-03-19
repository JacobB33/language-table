# coding=utf-8
# Copyright 2024 The Language Tale Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines block2block reset and reward."""
import itertools

from absl import logging
from language_table.environments import blocks as blocks_module
from language_table.environments.rewards import constants
from language_table.environments.rewards import reward as base_reward
from language_table.environments.rewards import synonyms
from language_table.environments.rewards import task_info
import numpy as np


# pytype: skip-file
class NoOpReward(base_reward.LanguageTableReward):
  """Block2block reward."""

  def _sample_instruction(
      self, start_block, target_block, blocks_on_table):
    return ""
  def reset(self, state, blocks_on_table):
    return task_info.NoOpTaskInfo(
        instruction="",
        block=self._rng.choice(blocks_on_table)
        )

  def get_goal_region(self):
    return 0, 0

  def reward(self, state):
    """Calculates reward given state."""
    return 0, False
