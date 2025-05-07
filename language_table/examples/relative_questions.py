import random
import numpy as np
import itertools

from language_table.environments.rewards.block2block_relative_location import DIRECTIONS, DIRECTION_IDS, Locations, \
    DIRECTION_SYNONYMS, DIRECTION_SYNONYMS_RELATIVE_NO_CONNECTION
from sympy import false, true

RELATIVE_DISTANCE_THRESHOLD = .04


def did_obj_move_direction(previous_pose, current_pose, direction):
    movement_vector = current_pose - previous_pose
    if np.linalg.norm(movement_vector) < RELATIVE_DISTANCE_THRESHOLD:
        return False

    # Get the reference direction vector from DIRECTIONS dictionary
    reference_direction = np.array(DIRECTIONS[direction])
    movement_vector_normalized = movement_vector / np.linalg.norm(movement_vector)
    reference_direction_normalized = reference_direction / np.linalg.norm(reference_direction)
    dot_product = np.dot(movement_vector_normalized, reference_direction_normalized)

    # Calculate the angle in degrees between the two vectors using arc cosine
    angle_radians = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)

    if 'diagonal' in direction:
        # Allow up to 30 degrees of deviation for diagonal directions
        return angle_degrees <= 30
    else:
        # For cardinal directions, use a stricter threshold of 20 degrees
        return angle_degrees <= 20


def generate_block_move_direction_questions(past_states, current_states, num_questions):
    # Define various question templates for asking if blocks moved in a direction
    direction_movement_templates = [
        "Did the {block} move {direction}?",
        "Has the {block} block shifted {direction}?",
        "Was the {block} block moved {direction}?",
        "Did the {block} travel {direction}?",
        "Has the {block} been pushed {direction}?"
    ]
    def _get_move_direction_q(block, direction, old_position, new_position):
        block_name = block.replace('_', ' ')
        direction_string = random.choice(DIRECTION_SYNONYMS[direction])
        moved_in_direction = did_obj_move_direction(old_position, new_position, direction)
        # Randomly select a question template
        template = random.choice(direction_movement_templates)
        question = template.format(block=block_name, direction=direction_string)
        return question, moved_in_direction

    qa_pairs = []
    keys = list(past_states.keys())
    del keys[keys.index('peg')]
    distances = [np.linalg.norm(past_states[block] - current_states[block]) for block in keys]
    moved_idxs = np.where(np.array(distances) > RELATIVE_DISTANCE_THRESHOLD)[0]
    yes_questions = min(len(moved_idxs), num_questions // 2)
    for idx in random.sample(moved_idxs.tolist(), yes_questions):
        block = keys[idx]
        old_position = past_states[block]
        new_position = current_states[block]
        normalized_position = (new_position - old_position) / np.linalg.norm(new_position - old_position)
        closest_direction = max(DIRECTION_IDS, key=lambda x: np.dot(normalized_position, np.array(DIRECTIONS[x])))
        question, answer = _get_move_direction_q(block, closest_direction, old_position, new_position)
        qa_pairs.append((question, answer))
    
    while len(qa_pairs) < num_questions:
        block = random.choice(keys)
        old_position = past_states[block]
        new_position = current_states[block]
        direction = random.choice(list(DIRECTION_SYNONYMS_RELATIVE_NO_CONNECTION.keys()))
        question, answer = _get_move_direction_q(block, direction, old_position, new_position)
        if answer:
            pass
        
        qa_pairs.append((question, answer))
    if yes_questions > 0:
        weights = [.5 / yes_questions] * yes_questions + [0.5 / (num_questions - yes_questions)] * (num_questions - yes_questions)
    else:
        weights = [0] * num_questions
    return qa_pairs, weights



def generate_peg_move_questions(past_states, current_states):
    # Define various question templates for asking if blocks moved in a direction
    direction_movement_templates = [
        "Did the {peg} move {direction}?",
        "Has the {peg} block shifted {direction}?",
        "Was the {peg} block moved {direction}?",
        "Did the {peg} travel {direction}?",
    ]

    qa_pairs = []
    directions = random.sample(DIRECTION_SYNONYMS_RELATIVE_NO_CONNECTION.keys(), 3)

    for direction in directions:
        old_position = past_states['peg']
        new_position = current_states['peg']
        peg_name = 'robotic peg'
        direction_string = random.choice(DIRECTION_SYNONYMS_RELATIVE_NO_CONNECTION[direction])

        # Check if the block moved in this direction
        moved_in_direction = did_obj_move_direction(old_position, new_position, direction)

        # Randomly select a question template
        template = random.choice(direction_movement_templates)
        question = template.format(peg=peg_name, direction=direction_string)

        qa_pairs.append((question, moved_in_direction))
        
    # sample one true and one false question
    true = [qa for qa in qa_pairs if qa[1]]
    if len(true) == 0:
        weights = [0, 0]
        true_question = random.choice(qa_pairs)
    else:
        true_question = random.choice(true)
        weights = [0.5, 0.5]
    false_question = random.choice([qa for qa in qa_pairs if not qa[1]])

    return [true_question, false_question], weights


def generate_relative_peg_block_questions(past_states, current_states):
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
    num_false = len(qa_pairs) - num_true
    if num_true > 0 and num_false > 0:
        weights = [0.5 / num_true] * num_true + [0.5 / num_false] * num_false
    else:
        weights = [0] * len(qa_pairs)
    return qa_pairs, weights




def generate_did_block_move_questions(past_states, current_states):
    qa_pairs = []
    movement_templates = [
        "Did the {block} block move?",
        "Has the {block} block changed position?",
        "Is the {block} block in a different location now?",
        "Did the {block} block shift from its original position?",
        "Has the {block} block been repositioned?",
        "Is the {block} block still in the same place?",  # This one is negated in the answer
        "Did the {block} block remain stationary?",       # This one is negated in the answer
        "Was the {block} block moved?"
    ]

    for block in past_states.keys():
        old_block, new_block = past_states[block], current_states[block]
        distance_moved = np.linalg.norm(old_block - new_block)
        answer = distance_moved < RELATIVE_DISTANCE_THRESHOLD
        # Clean up block name
        block_name = block.replace('-', ' ')

        # Randomly select a question template
        template = random.choice(movement_templates)
        question = template.format(block=block_name)

        # Adjust answer for negated questions
        if "still in the same place" in question or "remain stationary" in question:
            answer = not answer
        qa_pairs.append((question, answer))
    qa_pairs.sort(key=lambda x: x[1])
    num_true = sum(1 for _, answer in qa_pairs if answer)
    num_false = len(qa_pairs) - num_true
    if num_true > 0 and num_false > 0:
        weights = [0.5 / num_true] * num_true + [0.5 / num_false] * num_false
    else:
        weights = [0] * len(qa_pairs)
    return qa_pairs, weights

def generate_relative_block_block_questions(past_states, current_states, num_questions):
    qa_pairs = []
    keys = list(past_states.keys())
    pairs = set(tuple(sorted(pair)) for pair in itertools.combinations(keys, 2))
    # pairs = random.sample(unique_pairs, min(num_questions, len(unique_pairs)))

    question_templates = [
        "Are the {block1} and {block2} closer together?",
        "Have the {block1} and {block2} moved closer to each other?",
        "Did the distance between {block1} and {block2} decrease?",
        "Are the {block1} and {block2} nearer than before?"
        "Is {block1} closer to the {block2}?"
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
    false_pairs = [qa for qa in qa_pairs if not qa[1]]
    num_true =  min(len(true_pairs), num_questions // 2)
    true_pairs = random.sample(true_pairs, num_true)
    false_pairs = random.sample(false_pairs, num_questions - num_true)
    final_pairs = true_pairs + false_pairs

    if num_true > 0:
        weights = [0.5 / num_true] * num_true + [0.5 / (len(final_pairs) - num_true)] * (len(final_pairs) - num_true)
    else:
        weights = [0] * len(final_pairs)
    return final_pairs, weights

# CLaude generated viz function with minior edits. Ignore if you do not need
def visualize_block_move_direction(env, previous_image, current_image, past_states, current_states, num_questions=3):
    """
    Visualize block movement directions with vectors and angle annotations.

    Args:
        past_states: Dict mapping block names to previous positions
        current_states: Dict mapping block names to current positions
        num_questions: Number of movement questions to visualize
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc
    import numpy as np
    import random
    import time



    # Filter out non-moving blocks
    moving_blocks = []
    for block in past_states.keys():
        if block == "peg":
            continue

        old_position = past_states[block]
        new_position = current_states[block]
        distance_moved = np.linalg.norm(old_position - new_position)

        if distance_moved >= RELATIVE_DISTANCE_THRESHOLD:
            moving_blocks.append(block)

    if not moving_blocks:
        print("No blocks moved significantly.")
        return

    # Sample blocks to visualize
    blocks_to_visualize = random.sample(moving_blocks, min(num_questions, len(moving_blocks)))

    for block in blocks_to_visualize:
        # Get block positions
        old_position = past_states[block]
        new_position = current_states[block]
        block_name = block.replace('_', ' ')

        # Determine actual movement direction
        movement_vector = new_position - old_position
        movement_magnitude = np.linalg.norm(movement_vector)
        movement_normalized = movement_vector / movement_magnitude

        # Find the closest matching direction
        best_direction = None
        smallest_angle = 180

        for direction in DIRECTIONS.keys():
            reference_direction = np.array(DIRECTIONS[direction])
            reference_normalized = reference_direction / np.linalg.norm(reference_direction)

            dot_product = np.dot(movement_normalized, reference_normalized)
            angle_radians = np.arccos(np.clip(dot_product, -1.0, 1.0))
            angle_degrees = np.degrees(angle_radians)

            if angle_degrees < smallest_angle:
                smallest_angle = angle_degrees
                best_direction = direction

        # Create plot
        plt.figure(figsize=(12, 8))

        # Display the environment image
        plt.subplot(1, 2, 1)
        plt.imshow(current_image)
        plt.title("Current Environment")
        plt.axis('off')

        # Create vector visualization
        plt.subplot(1, 2, 2)
        plt.imshow(previous_image)
        plt.title(f"Movement Direction Analysis: {block_name}")
        plt.axis('off')

        # Create 3D points for projection
        # Starting position
        point3d_start = np.array([old_position[0], old_position[1], 0.01, 1])

        # Ending position
        point3d_end = np.array([new_position[0], new_position[1], 0.01, 1])

        # Project points to 2D image coordinates
        points3d = np.column_stack((point3d_start, point3d_end))
        start_x, start_y = env.get_camera_pix_coords(point3d_start.reshape(-1, 4).T)
        end_x, end_y = env.get_camera_pix_coords(point3d_end.reshape(-1, 4).T)

        # Plot actual movement vector (red)
        plt.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                  color='red', width=2, head_width=10, head_length=10,
                  length_includes_head=True, label="Actual Movement")

        # Plot closest direction vector (blue)
        closest_direction_vector = np.array(DIRECTIONS[best_direction]) * movement_magnitude
        direction_end = old_position + closest_direction_vector

        point3d_direction = np.array([direction_end[0], direction_end[1], 0.01, 1])
        dir_end_x, dir_end_y = env.get_camera_pix_coords(point3d_direction.reshape(-1, 4).T)

        plt.arrow(start_x, start_y, dir_end_x - start_x, dir_end_y - start_y,
                  color='blue', width=2, head_width=10, head_length=10,
                  length_includes_head=True, label=f"'{best_direction}' Direction")

        # Draw an arc to show the angle between vectors
        if smallest_angle > 0.5:  # Only draw if angle is significant
            # Create an arc to show the angle between the vectors
            arc_radius = 30
            plt.gca().add_patch(Arc((start_x, start_y), arc_radius * 2, arc_radius * 2,
                                    theta1=0, theta2=smallest_angle,
                                    color='green', lw=2))

            # Add angle text at the midpoint of the arc
            midpoint_angle = smallest_angle / 2
            angle_text_x = start_x + arc_radius * 1.5 * np.cos(np.radians(midpoint_angle))
            angle_text_y = start_y + arc_radius * 1.5 * np.sin(np.radians(midpoint_angle))
            plt.text(angle_text_x, angle_text_y, f"{smallest_angle:.1f}°",
                     color='green', fontsize=12, ha='center', va='center')

        # Generate a question about this movement
        direction_string = random.choice(DIRECTION_SYNONYMS[best_direction])
        template = random.choice([
            "Did the {block} move {direction}?",
            "Has the {block} block shifted {direction}?",
            "Was the {block} block moved {direction}?",
        ])
        question = template.format(block=block_name, direction=direction_string)

        # Determine the answer based on the angle
        threshold = 30 if 'diagonal' in best_direction else 20
        answer = smallest_angle <= threshold

        # Add question and answer as title
        plt.suptitle(f"Question: {question}\nAnswer: {answer}\n" +
                     f"Angle between actual movement and '{best_direction}': {smallest_angle:.1f}°",
                     fontsize=12)

        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
        time.sleep(1)  # Pause between plots
