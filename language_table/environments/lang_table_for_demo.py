import itertools
from .language_table import LanguageTable
import random
import numpy as np

from language_table.environments.rewards.constants import TARGET_BLOCK_DISTANCE

TOUCHING_DISTANCE_THRESHOLD = 0.05
RELATIVE_DISTANCE_THRESHOLD = .04

class LanguageTableDemo(LanguageTable):
    def _get_visible_block_list(self):
        visible_block_ids = [self._block_to_pybullet_id[i] for i in self._blocks_on_table]
        ids_to_names = {v: k for k, v in self._block_to_pybullet_id.items()}
        blocks = [obj for obj in self.get_pybullet_state()["objects"] if obj.obj_id in visible_block_ids]
        return blocks, ids_to_names

    def get_block_states(self):
        blocks, ids_to_names = self._get_visible_block_list()
        state = self._compute_state()
        block_positions = {ids_to_names[obj.obj_id]: np.array(obj.base_pose[0])[:2] for obj in blocks}
        # GET THE PEG POSE
        block_positions["peg"] = np.array(state['effector_target_translation'])

        return block_positions

    def get_block_peg_info(self):
        blocks, ids_to_names = self._get_visible_block_list()
        result = {}
        for block in blocks:
            block_position, _ = block.base_pose
            block_position = np.array(block_position)[:2]

            state = self._compute_state()

            dist = np.linalg.norm(
                np.array(block_position) -
                np.array(state['effector_target_translation']))

            result[ids_to_names[block.obj_id]] = dist

        return result

    def get_block_touching_questions(self, num_questions=8):
        touching_templates = [
            "Is the {block1} touching the {block2}?",
            "Are the {block1} and {block2} blocks in contact with each other?",
            "Is there contact between the {block1} block and the {block2} block?",
            "Does the {block1} touch the {block2}?",
            "Is the {block1} block in physical contact with the {block2} block?",
            "Are the {block1} and {block2} blocks touching each other?",
            "Is the {block1} and {block2} directly touching?",
            "Do the {block1} and {block2} blocks meet?",
        ]

        blocks, ids_to_names = self._get_visible_block_list()
        n_blocks = len(blocks)
        # Extract positions and names once
        positions = np.array([block.base_pose[0][:2] for block in blocks])
        names = [ids_to_names[block.obj_id].replace("_", " ") for block in blocks]

        # Generate all i < j pairs
        i_idx, j_idx = np.triu_indices(n_blocks, k=1)
        vec_dists = np.linalg.norm(positions[i_idx] - positions[j_idx], axis=1)
        is_touching = vec_dists < TOUCHING_DISTANCE_THRESHOLD
        qa_pairs = []
        for idx, (i, j) in enumerate(zip(i_idx, j_idx)):
            block1_name, block2_name = names[i], names[j]
            if random.choice([True, False]):
                block1_name, block2_name = block2_name, block1_name
            # Randomly select a question template
            template = random.choice(touching_templates)
            question = template.format(block1=block1_name, block2=block2_name)
            qa_pairs.append((question, bool(is_touching[idx])))
        pair_dict = {
            (i_idx[k], j_idx[k]): qa_pairs[k] for k in range(len(qa_pairs))
        }
        return pair_dict


    def get_peg_block_questions(self):
        peg_templates = [
            "Is the {block} next to the peg?",
            "Is the peg next to the {block}?",
            "Is the {block} touching the peg?",
            "Is the {block} block near the peg?",
            "Is the peg adjacent to the {block}?",
            "Is the peg touching the {block}?",
        ]

        blocks, ids_to_names = self._get_visible_block_list()

        qa_pairs = []
        for block in blocks:
            block_position, _ = block.base_pose
            block_position = np.array(block_position)[:2]

            state = self._compute_state()

            dist = np.linalg.norm(
                np.array(block_position) -
                np.array(state['effector_target_translation']))

            answer = dist < TARGET_BLOCK_DISTANCE
            # Get block name
            block_name = ids_to_names[block.obj_id].replace("_", " ")

            # Randomly select a question template
            template = random.choice(peg_templates)
            question = template.format(block=block_name)
            
            qa_pairs.append((question, answer))
        pair_dict = {
            block.obj_id: qa_pairs[i] for i, block in enumerate(blocks)
        }
        return pair_dict
    
    def get_relative_peg_block_question(self, past_states):
        current_states = self.get_block_states()
        # I want to not reballence this because I think that naturally this would be around 50/50
        prev_peg, cur_peg = past_states["peg"], current_states['peg']
        qa_pairs = []
        question_templates = [
            "Is the robotic peg closer to the {block}?",
            "Has the robotic peg moved nearer to the {block}?",
            "Is the {block} now closer to the robotic peg?",
            "Did the robotic peg get closer to the {block} compared to before?"
        ]
        for block in past_states.keys():

            if block == "peg":
                continue
            old_block, new_block = past_states[block], current_states[block]
            old_peg_to_block = np.linalg.norm(old_block - prev_peg)
            new_peg_to_block = np.linalg.norm(new_block - cur_peg)
            question = random.choice(question_templates).format(block=block.replace("_", " "))
            answer = new_peg_to_block < old_peg_to_block - RELATIVE_DISTANCE_THRESHOLD
            qa_pairs.append((question, answer))
        qa_pairs.sort(key=lambda x: x[1])
        num_true = sum(1 for _, answer in qa_pairs if answer)
        if num_true == 0:
            return random.sample(qa_pairs, 1)
        else:
            return random.sample(qa_pairs[:num_true], 1)
    
    def get_relative_block_block_question(self, past_states):
        current_states = self.get_block_states()
        qa_pairs = []
        keys = list(past_states.keys())
        pairs = set(tuple(sorted(pair)) for pair in itertools.combinations(keys, 2))
        # pairs = random.sample(unique_pairs, min(num_questions, len(unique_pairs)))

        question_templates = [
            "Are the {block1} and {block2} closer together?",
            "Have the {block1} and {block2} moved closer to each other?",
            "Did the distance between {block1} and {block2} decrease?",
            "Are the {block1} and {block2} nearer than before?",
            # "Is {block1} closer to the {block2}?"
        ]

        for pair in pairs:
            pair = random.sample(pair, 2)
            block1_name, block2_name = pair
            block1_old, block2_old = past_states[block1_name], past_states[block2_name]
            block1_new, block2_new = current_states[block1_name], current_states[block2_name]
            old_distance = np.linalg.norm(block1_old - block2_old)
            new_distance = np.linalg.norm(block1_new - block2_new)

            question = random.choice(question_templates).format(
                block1=block1_name.replace("_", " "),
                block2=block2_name.replace("_", " ")
            )
            answer = new_distance < old_distance - RELATIVE_DISTANCE_THRESHOLD
            qa_pairs.append((question, answer))
        
        true_pairs = [qa for qa in qa_pairs if qa[1]]
        if len(true_pairs) == 0:
            return random.sample(qa_pairs, 1)
        else:
            return random.sample(true_pairs, 1)
        

