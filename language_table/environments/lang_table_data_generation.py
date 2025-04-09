import time
from .language_table import LanguageTable
import random
import numpy as np

from language_table.environments.rewards.block2block_relative_location import DIRECTION_IDS, DIRECTION_SYNONYMS
from language_table.environments.rewards.block2block_relative_location import MAGNITUDE_X, MAGNITUDE_Y, MAGNITUDE_X_DIAG, MAGNITUDE_Y_DIAG, DIRECTIONS, BLOCK2BLOCK_REL_LOCATION_TARGET_DISTANCE
from language_table.environments.rewards.constants import TARGET_BLOCK_DISTANCE
from language_table.environments.rewards.block2absolutelocation import LOCATION_SYNONYMS, ABSOLUTE_LOCATIONS, Locations, BLOCK2ABSOLUTELOCATION_CENTER_TARGET_DISTANCE, BLOCK2ABSOLUTELOCATION_TARGET_DISTANCE


class LanguageTableDataGeneration(LanguageTable):
    def _get_visible_block_list(self):
        visible_block_ids = [self._block_to_pybullet_id[i] for i in self._blocks_on_table]
        ids_to_names = {v: k for k, v in self._block_to_pybullet_id.items()}
        blocks = [obj for obj in self.get_pybullet_state()["objects"] if obj.obj_id in visible_block_ids]
        # random.shuffle(blocks)
        return blocks, ids_to_names

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
            import matplotlib.pyplot as plt
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

    def get_block_touching_questions(self):
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
        qa_pairs = []
        for i in range(n_blocks):
            for j in range(i + 1, n_blocks):
                block_x, _ = blocks[i].base_pose
                block_y, _ = blocks[j].base_pose
                # Calculate Euclidean distance
                distance = np.linalg.norm(np.array(block_x)[:2] - np.array(block_y)[:2])
                is_touching = distance < 0.05  # Define touching threshold

                block1_name = ids_to_names[blocks[i].obj_id].replace("_", " ")
                block2_name = ids_to_names[blocks[j].obj_id].replace("_", " ")

                if random.choice([True, False]):
                    # sometimes switch the block names
                    block1_name, block2_name = block2_name, block1_name
                # Randomly select a question template
                template = random.choice(touching_templates)
                question = template.format(block1=block1_name, block2=block2_name)
                qa_pairs.append((question, is_touching))
        assert len(qa_pairs) == len(set(qa_pairs))
        return qa_pairs

    def get_relative_block2block_questions(self, number_of_questions=5, scale=1.3):
        rel_position_templates = [
            "Is the {block1} {direction} {block2}?",
            "Can you confirm if the {block1} block is {direction} {block2} block?",
            "Is it true that the {block1} block is positioned {direction} {block2} block?",
            "Would you say the {block1} block is {direction} {block2} block?",
            "Does the {block1} block lie {direction} {block2} block?",
            "Is the {block1} situated {direction} {block2}?",
            "Based on their positions, is the {block1} block {direction} {block2} block?"
        ]
        blocks, ids_to_names = self._get_visible_block_list()
        qa_pairs = []
        for i in range(number_of_questions):
            pushing_block, target_block = random.sample(blocks, 2)
            direction = random.choice(DIRECTION_IDS)
            target_string = random.choice(DIRECTION_SYNONYMS[direction])
            block1_name = ids_to_names[pushing_block.obj_id].replace("_", " ")
            block2_name = ids_to_names[target_block.obj_id].replace("_", " ")
            # Randomly select a question template
            template = random.choice(rel_position_templates)
            question = template.format(block1=block1_name, direction=target_string, block2=block2_name)

            relative_of, _ = pushing_block.base_pose
            relative_to, _ = target_block.base_pose
            # remove the z component
            relative_of = np.array(relative_of)[:2]
            relative_to = np.array(relative_to)[:2]

            pushing_block_on_line = self.check_direction(relative_to, relative_of, direction, scale, question=question, viz=False)

            qa_pairs.append((question, pushing_block_on_line))
        return qa_pairs

    def get_peg_block_questions(self):
        peg_templates = [
            "Is the {block} next to the peg?",
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
        return qa_pairs

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

    def get_block_to_board_questions(self, number_of_questions=5):
        board_templates = [
            "Is the {block} in the {location} of the board?",
            "Is the {block} block located in the {location} area?",
            "Is the {block} positioned in the {location} of the board?",
            "Is the {block} block situated in the {location} of the board?",
            "Does the {block} occupy the {location} area of the board?"
        ]

        blocks, ids_to_names = self._get_visible_block_list()
        qa_pairs = []
        for i in range(number_of_questions):
            block = random.choice(blocks)
            block_position, _ = block.base_pose
            block_position = np.array(block_position)[:2]

            target_translation = random.choice(list(ABSOLUTE_LOCATIONS.keys()))

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
            qa_pairs.append((question, success))
        return qa_pairs

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
        state = self._compute_state()
        peg_position = np.array(state['effector_target_translation'])

        qa_pairs = []
        for i in range(number_of_questions):
            # Select a random block
            target_block = random.choice(blocks)
            direction = random.choice(DIRECTION_IDS)
            target_string = random.choice(DIRECTION_SYNONYMS[direction])
            # Get block name
            block_name = ids_to_names[target_block.obj_id].replace("_", " ")

            # Randomly select a question template
            template = random.choice(peg_rel_templates)
            question = template.format(direction=target_string, block=block_name)

            target_block_translation, _ = target_block.base_pose
            # Remove the z component
            target_block_translation = np.array(target_block_translation)[:2]

            peg_on_line = self.check_direction(target_block_translation, peg_position, direction, scale, question=question, viz=False)

            qa_pairs.append((question, peg_on_line))
        return qa_pairs

    def get_block_states(self):
        blocks, ids_to_names = self._get_visible_block_list()
        state = self._compute_state()
        block_positions = {ids_to_names[obj.obj_id]: np.array(obj.base_pose[0])[:2] for obj in blocks}
        # GET THE PEG POSE
        block_positions["peg"] = np.array(state['effector_target_translation'])

        return block_positions
