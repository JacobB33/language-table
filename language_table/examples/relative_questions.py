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
        direction_string = random.choice(DIRECTION_SYNONYMS_RELATIVE_NO_CONNECTION[direction])
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
        qa_pairs.append((question, answer, "blockmovedirection"))
    
    while len(qa_pairs) < num_questions:
        block = random.choice(keys)
        old_position = past_states[block]
        new_position = current_states[block]
        direction = random.choice(list(DIRECTION_SYNONYMS_RELATIVE_NO_CONNECTION.keys()))
        question, answer = _get_move_direction_q(block, direction, old_position, new_position)
        if answer:
            pass
        qa_pairs.append((question, answer,  "blockmovedirection"))
    return qa_pairs



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

        qa_pairs.append((question, moved_in_direction, "pegmovedirection"))
        
    # sample one true and one false question
    true = [qa for qa in qa_pairs if qa[1]]
    if len(true) == 0:
        true_question = random.choice(qa_pairs)
    else:
        true_question = random.choice(true)
    false_question = random.choice([qa for qa in qa_pairs if not qa[1]])

    return [true_question, false_question]


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
        qa_pairs.append((question, answer, "pegblockrealtive"))
    return qa_pairs




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
        block_name = block.replace('_', ' ') 

        # Randomly select a question template
        template = random.choice(movement_templates)
        question = template.format(block=block_name)

        # Adjust answer for negated questions
        if "still in the same place" in question or "remain stationary" in question:
            answer = not answer
        qa_pairs.append((question, answer, "blockmoved"))
    return qa_pairs

def generate_relative_block_block_questions(past_states, current_states, num_questions):
    qa_pairs = []
    keys = list(past_states.keys())
    pairs = set(tuple(sorted(pair)) for pair in itertools.combinations(keys, 2))
    # pairs = random.sample(unique_pairs, min(num_questions, len(unique_pairs)))

    question_templates = [
        "Are the {block1} and {block2} closer together?",
        "Have the {block1} and {block2} moved closer to each other?",
        "Did the distance between {block1} and {block2} decrease?",
        "Are the {block1} and {block2} nearer than before?",
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
        qa_pairs.append((question, answer, "blockCloser"))
    
    true_pairs = [qa for qa in qa_pairs if qa[1]]
    false_pairs = [qa for qa in qa_pairs if not qa[1]]
    num_true =  min(len(true_pairs), num_questions // 2)
    true_pairs = random.sample(true_pairs, num_true)
    false_pairs = random.sample(false_pairs, num_questions - num_true)
    final_pairs = true_pairs + false_pairs

    return final_pairs

