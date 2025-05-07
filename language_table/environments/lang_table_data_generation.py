import time
from .language_table import LanguageTable
import random
import numpy as np

from language_table.environments.rewards.block2block_relative_location import DIRECTION_IDS, DIRECTION_SYNONYMS
from language_table.environments.rewards.block2block_relative_location import MAGNITUDE_X, MAGNITUDE_Y, MAGNITUDE_X_DIAG, MAGNITUDE_Y_DIAG, DIRECTIONS, BLOCK2BLOCK_REL_LOCATION_TARGET_DISTANCE
from language_table.environments.rewards.constants import TARGET_BLOCK_DISTANCE
from language_table.environments.rewards.block2absolutelocation import LOCATION_SYNONYMS, ABSOLUTE_LOCATIONS, Locations, \
    BLOCK2ABSOLUTELOCATION_CENTER_TARGET_DISTANCE, BLOCK2ABSOLUTELOCATION_TARGET_DISTANCE, ABSOLUTE_LOCATIONS_POSES, \
    ABSOLUTE_LOCATIONS_IDS
import matplotlib.pyplot as plt

TOUCHING_DISTANCE_THRESHOLD = 0.05

class LanguageTableDataGeneration(LanguageTable):
    def _get_visible_block_list(self):
        visible_block_ids = [self._block_to_pybullet_id[i] for i in self._blocks_on_table]
        ids_to_names = {v: k for k, v in self._block_to_pybullet_id.items()}
        blocks = [obj for obj in self.get_pybullet_state()["objects"] if obj.obj_id in visible_block_ids]
        # random.shuffle(blocks)
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

    def check_in_direction_range(self, pose_to, pose_of, scale):
        # Check if the distance between the two points is within the specified range
        dist = np.linalg.norm(np.array(pose_to) - np.array(pose_of))
        return dist < scale * (MAGNITUDE_X - 0.01)
    
    def check_direction(self, relative_to_pose, relative_of_pose, direction, scale, question="", viz=False):
        # Consider the end point of the line 2x longer than the offset.
        mag_x = MAGNITUDE_X_DIAG if 'diagonal' in direction else MAGNITUDE_X
        mag_y = MAGNITUDE_Y_DIAG if 'diagonal' in direction else MAGNITUDE_Y
        target_vector = np.array(DIRECTIONS[direction]) * np.array(
            [mag_x * scale, mag_y * scale])
        # Define target_translation (where to push to) as target block translation
        # offset by target_vector.
        offset_translation = np.array(relative_to_pose) + target_vector

        diff = offset_translation - relative_to_pose
        # Consider all points half the distance from target block to offset.
        minpoint = diff * 0.05
        # Consider all points 10% further than the offset.
        maxpoint = diff * 1
        # Is the target block somewhere on the line between min point and max point?
        diffs = np.linspace(minpoint, maxpoint, 15)

        # Check if pushing block is on the line
        pushing_block_on_line = False
        for cand_offset in diffs:
            point = relative_to_pose + cand_offset
            dist = np.linalg.norm(point - relative_of_pose)
            if dist < BLOCK2BLOCK_REL_LOCATION_TARGET_DISTANCE:
                pushing_block_on_line = True
                break
        if viz:
            state = self._compute_state()
            image = state['rgb']
            # Create a matplotlib figure for visualization
            plt.figure(figsize=(12, 8))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("Current Environment")
            plt.axis('off')
            # Second subplot for projected points
            plt.subplot(1, 2, 2)
            plt.imshow(image)
            plt.title("Relative Direction Visualization")
            plt.axis('off')
            line_points = []
            for cand_offset in diffs:
                point = relative_to_pose + cand_offset

                line_points.append([point[0], point[1], 0.0, 1])  # Small z value to be above the table

            # Add target block and pushing block positions
            line_points.append([relative_to_pose[0], relative_to_pose[1], 0.0, 1])
            line_points.append([relative_of_pose[0], relative_of_pose[1], 0.0, 1])
            # Project points to image coordinates
            line_points_np = np.array(line_points).T
            pixel_x, pixel_y = self.get_camera_pix_coords(line_points_np)
            # Plot points with gradient color (blue to red)
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(pixel_x) - 2))
            # Plot the line with gradient color to show direction
            for j in range(len(pixel_x) - 3):
                plt.plot(pixel_x[j:j + 2], pixel_y[j:j + 2], color=colors[j], linewidth=2)
            # Plot the target block position
            plt.scatter(pixel_x[-2], pixel_y[-2], color='green', s=100, marker='o',
                        label=f"Relative To Block ")
            # Plot the pushing block position
            plt.scatter(pixel_x[-1], pixel_y[-1], color='blue', s=100, marker='x',
                        label=f"Relative Of Block")
            # # Add a circle of points around the "relative to" block with radius 0.1 meters
            # circle_points = []
            # num_circle_points = 36  # For a smooth circle
            # circle_radius = 0.03  # 10cm radius
            # for angle in np.linspace(0, 2 * np.pi, num_circle_points):
            #     # Calculate point on circle
            #     circle_x = relative_to_pose[0] + circle_radius * np.cos(angle)
            #     circle_y = relative_to_pose[1] + circle_radius * np.sin(angle)
            #     circle_points.append([circle_x, circle_y, 0.0, 1])
            #
            # # Convert circle points to numpy array for projection
            # circle_points_np = np.array(circle_points).T
            # circle_pixel_x, circle_pixel_y = self.get_camera_pix_coords(circle_points_np)
            # plt.plot(circle_pixel_x, circle_pixel_y, 'y-', linewidth=2, alpha=0.7, label=f"{circle_radius}m Radius")
            plt.suptitle(f"Question: {question}\nAnswer: {pushing_block_on_line}", fontsize=14)
            plt.legend(loc='upper right')
            # Show the plot
            plt.tight_layout()
            plt.show()
            time.sleep(1)

        return pushing_block_on_line
    
    def get_block_touching_questions(self, num_questions=8):
        touching_templates = [
            "Is the {block1} touching the {block2}?",
            "Are the {block1} and {block2} blocks in contact with each other?",
            "Is there contact between the {block1} block and the {block2} block?",
            "Does the {block1} touch the {block2}?",
            "Is the {block1} block in physical contact with the {block2} block?",
            "Are the {block1} and {block2} blocks touching each other?",
            "Is the {block1} and {block2} directly touching?",
            "Do the {block1} and {block2} blocks meet?"
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
            
        yes_pairs = [pair for pair in qa_pairs if pair[1]]
        no_pairs = [pair for pair in qa_pairs if not pair[1]]

        num_yes = len(yes_pairs)
        num_no = len(no_pairs)

        yes_to_sample = min(num_yes, num_questions // 2)
        no_to_sample = num_questions - yes_to_sample
        if num_no < no_to_sample:
            print('weird case where more no than yes in the block touching')
            no_to_sample = num_no
            yes_to_sample = num_questions - no_to_sample

        qa_pairs_yes = random.sample(yes_pairs, yes_to_sample)
        qa_pairs_no = random.sample(no_pairs, no_to_sample)
        qa_pairs = qa_pairs_yes + qa_pairs_no
        if yes_to_sample == 0:
            weights = [0] * len(qa_pairs)
        else:
            weights = [.5 / yes_to_sample] * yes_to_sample + [.5 / no_to_sample] * no_to_sample
        assert len(weights) == len(qa_pairs), f"Weight length mismatch in block_touching: {len(weights)} weights for {len(qa_pairs)} questions"

        return qa_pairs, weights


    def get_relative_block2block_questions(self, number_of_questions=5, scale=1.3):
        blocks, ids_to_names = self._get_visible_block_list()
        rel_position_templates = [
            "Is the {block1} {direction} {block2}?",
            "Can you confirm if the {block1} block is {direction} {block2} block?",
            "Is it true that the {block1} block is positioned {direction} {block2} block?",
            "Would you say the {block1} block is {direction} {block2} block?",
            "Does the {block1} block lie {direction} {block2} block?",
            "Is the {block1} situated {direction} {block2}?",
            "Based on their positions, is the {block1} block {direction} {block2} block?"
        ]
        def _get_block2block_question(block1_idx, block2_idx, get_yes):
            first_block = blocks[block1_idx]
            second_block = blocks[block2_idx]
            relative_of, _ = first_block.base_pose
            relative_to, _ = second_block.base_pose
            block1_name = ids_to_names[first_block.obj_id].replace("_", " ")
            block2_name = ids_to_names[second_block.obj_id].replace("_", " ")

            # remove the z component
            relative_of = np.array(relative_of)[:2]
            relative_to = np.array(relative_to)[:2]

            if get_yes:
                normalized_dir_vector = relative_of - relative_to
                normalized_dir_vector /= np.linalg.norm(normalized_dir_vector)
                direction = max(DIRECTIONS.items(), key=lambda item: normalized_dir_vector @ item[1])[0]
            else:
                direction = random.choice(DIRECTION_IDS)
            target_string = random.choice(DIRECTION_SYNONYMS[direction])
            template = random.choice(rel_position_templates)
            question = template.format(block1=block1_name, direction=target_string, block2=block2_name)
            pushing_block_on_line = self.check_direction(relative_to, relative_of, direction, scale, question=question, viz=False)
            return question, pushing_block_on_line
        
        n_blocks = len(blocks)
        block_idcs = range(n_blocks)
        qa_pairs = []
        i_idx, j_idx = np.triu_indices(n_blocks, k=1)
        close_idcs = np.where([self.check_in_direction_range(blocks[j].base_pose[0][:2], blocks[i].base_pose[0][:2], scale) for j, i in zip(j_idx, i_idx)])[0]
        num_yes = min(number_of_questions // 2, len(close_idcs))
        # generate the yes ones
        yes_pairs = random.sample(list(close_idcs), num_yes)
        for pair in yes_pairs:
            i, j = (i_idx[pair], j_idx[pair]) if bool(random.getrandbits(1)) else (j_idx[pair], i_idx[pair])
            qa_pairs.append(_get_block2block_question(i, j, get_yes=True))
        # Get the no answers:
        no_counter = 0
        while len(qa_pairs) < number_of_questions:
            no_counter += 1
            if no_counter > 100:
                print("REALLY CAN'T GENERATE NOS :(")
                break
            pushing_block, target_block = random.sample(block_idcs, 2)
            result = _get_block2block_question(pushing_block, target_block, get_yes=False)
            if result[1]:
                continue
            qa_pairs.append(result)
        if num_yes > 0:
            weights = [0.5/num_yes] * num_yes + [0.5 / (len(qa_pairs) - num_yes)] * (len(qa_pairs) - num_yes)
        else:
            weights = [0] * len(qa_pairs)
        assert len(weights) == len(qa_pairs), f"Weight length mismatch in relative_block2block: {len(weights)} weights for {len(qa_pairs)} questions"
        return qa_pairs, weights

    def get_peg_block_questions(self):
        peg_templates = [
            "Is the {block} next to the peg?",
            "Is the peg next to the {block}?"
            "Is the {block} touching the peg?",
            "Is the {block} block near the peg?",
            "Is the peg adjacent to the {block}?",
            "Is the peg touching the {block}?",
        ]

        blocks, ids_to_names = self._get_visible_block_list()

        qa_pairs = []
        trues = 0
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
            trues += 1 if answer else 0
        qa_pairs.sort(key=lambda pair: pair[1])
        if not trues:
            weights = [0] * len(qa_pairs)
        else:
            weights = [0.5/trues] * trues + [0.5/(len(qa_pairs) - trues)] * (len(qa_pairs) - trues)
        assert len(weights) == len(qa_pairs), f"Weight length mismatch in peg_block: {len(weights)} weights for {len(qa_pairs)} questions"
        return qa_pairs, weights


    def get_block_to_board_questions(self, number_of_questions=8):
        board_templates = [
            "Is the {block} in the {location} of the board?",
            "Is the {block} block located in the {location} area?",
            "Is the {block} positioned in the {location} of the board?",
            "Is the {block} block situated in the {location} of the board?",
            "Does the {block} occupy the {location} area of the board?"
        ]
        blocks, ids_to_names = self._get_visible_block_list()
        def _get_block_to_board_question(block, target_translation):
            block_position, _ = block.base_pose
            block_position = np.array(block_position)[:2]
            dist = np.linalg.norm(
                np.array(block_position) - np.array(ABSOLUTE_LOCATIONS[target_translation]))

            if target_translation == Locations.CENTER.value:
                target_dist = BLOCK2ABSOLUTELOCATION_CENTER_TARGET_DISTANCE
            else:
                target_dist = BLOCK2ABSOLUTELOCATION_TARGET_DISTANCE

            success = dist < target_dist

            block_name = ids_to_names[block.obj_id].replace("_", " ")
            location = random.choice(LOCATION_SYNONYMS[target_translation])

            # Randomly select a question template
            template = random.choice(board_templates)
            question = template.format(block=block_name, location=location)
            return question, success

        qa_pairs = []
        block_positions = np.array([block.base_pose[0][:2] for block in blocks])
        dists = np.linalg.norm(
            block_positions[:, None, :] - np.array(ABSOLUTE_LOCATIONS_POSES)[None, :, :],
            axis=2
        )
        closest_indices = np.argmin(dists, axis=1)
        closest_positions = [ABSOLUTE_LOCATIONS_IDS[i] for i in closest_indices]
        closest_distances = [dists[i, closest_indices[i]] for i in range(len(block_positions))]
        accurate_blocks = np.where(
            [closest_distances[i] <
             (BLOCK2ABSOLUTELOCATION_CENTER_TARGET_DISTANCE if "center" in closest_positions[i] else BLOCK2ABSOLUTELOCATION_TARGET_DISTANCE)
                for i in range(len(closest_positions))])[0]
        num_yes = min(number_of_questions // 2, len(accurate_blocks))
        yes_idxs = random.sample(list(accurate_blocks), num_yes)
        for idx in yes_idxs:
            block = blocks[idx]
            direction = closest_positions[idx]
            pair = _get_block_to_board_question(block, direction)
            assert pair[1]
            qa_pairs.append(pair)
        while len(qa_pairs) < number_of_questions:
            block = random.choice(blocks)
            idx = blocks.index(block)
            direction = random.choice(ABSOLUTE_LOCATIONS_IDS)
            if closest_positions[idx] == direction:
                continue
            pair = _get_block_to_board_question(block, direction)
            if pair[1]:
                continue
            qa_pairs.append(pair)
        if num_yes > 0:
            weights = [.5/num_yes] * num_yes + [.5/(len(qa_pairs) - num_yes)] * (len(qa_pairs) - num_yes)
        else:
            weights = [0] * len(qa_pairs)
        
        assert len(weights) == len(qa_pairs), f"Weight length mismatch in block_to_board: {len(weights)} weights for {len(qa_pairs)} questions"

        return qa_pairs, weights

    def get_peg_relative_to_block_questions(self, number_of_questions=5, scale=1.3):
        # Define various question templates for the peg relative to block questions
        peg_rel_templates = [
            "Is the peg {direction} {block} block?",
            "Can you confirm if the peg is {direction} {block} block?",
            "Is it true that the peg is positioned {direction} {block} block?",
            "Would you say the peg is {direction} {block} block?",
            "Does the peg lie {direction} {block} block?",
            "Is the peg situated {direction} {block} block?"
        ]

        blocks, ids_to_names = self._get_visible_block_list()
        block_idxs = list(range(len(blocks)))
        state = self._compute_state()
        peg_position = np.array(state['effector_target_translation'])
        def _get_peg_rel_question(block_idx, get_yes):
            target_block = blocks[block_idx]
            target_block_translation, _ = target_block.base_pose
            target_block_translation = np.array(target_block_translation)[:2]

            if get_yes:
                normalized_dir_vector = peg_position - target_block_translation
                normalized_dir_vector /= np.linalg.norm(normalized_dir_vector)
                direction = max(DIRECTIONS.items(), key=lambda item: normalized_dir_vector @ item[1])[0]
            else:
                direction = random.choice(DIRECTION_IDS)

            target_string = random.choice(DIRECTION_SYNONYMS[direction])
            # Get block name
            block_name = ids_to_names[target_block.obj_id].replace("_", " ")

            # Randomly select a question template
            template = random.choice(peg_rel_templates)
            question = template.format(direction=target_string, block=block_name)


            peg_on_line = self.check_direction(target_block_translation, peg_position, direction, scale, question=question, viz=False)

            return question, peg_on_line
        qa_pairs = []

        close_idcs = np.where(
            [self.check_in_direction_range(blocks[i].base_pose[0][:2], peg_position, scale) for i in range(len(blocks))])[0]
        num_yes = min(number_of_questions // 2, len(close_idcs))
        yes_idxs = random.sample(list(close_idcs), num_yes)
        for idx in yes_idxs:
            qa_pairs.append(_get_peg_rel_question(idx, get_yes=True))

        while len(qa_pairs) < number_of_questions:
            idx = random.choice(block_idxs)
            qa_pairs.append(_get_peg_rel_question(idx, False))
        if num_yes > 0:
            weights = [0.5/num_yes] * num_yes + [0.5/(len(qa_pairs)-num_yes)] * (len(qa_pairs) - num_yes)
        else:
            weights = [0.0] * len(qa_pairs)
        assert len(weights) == len(qa_pairs), f"Weight length mismatch in block_to_board: {len(weights)} weights for {len(qa_pairs)} questions"
        return qa_pairs, weights

