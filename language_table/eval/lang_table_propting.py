import multiprocessing
import os
import pickle as pkl
import numpy as np
from language_table.environments.rewards.constants import TARGET_BLOCK_DISTANCE
import random
from tqdm.auto import tqdm
from swm.planning.llms.gemini_model import GeminiModel
from swm.data.data_generation.question_types import GetAnswer
from PIL import Image
import wandb
# seed random
random.seed(0)
np.random.seed(0)
DATA_DIRECTORY = "/home/jacob/projects/semantic_world_modeling/language-table/language_table/debug_gem/demos"
ABLATION_NAME = "name"
answer_gem_q = GetAnswer()
num_threads = 18

# thread local model
_model = None

def init_model():
    global _model
    _model = GeminiModel(verbose=False)
    _model.register_new_model("a_model", answer_gem_q.get_config())

def get_questions(files):
    questions = []
    for file in files:
        orc_file = pkl.load(open(f"{DATA_DIRECTORY}/{file}", 'rb'))
        states = orc_file['block_states'][10:]
        images = orc_file['frames'][10:]
        for i in range(len(states)):
            state = states[i]
            image = images[i]
            peg_pose = state.pop('peg')
            state_qs = []
            for block in state:
                block_pose = state[block]
                dist = np.linalg.norm(block_pose - peg_pose)
                # question = f"Is the tip of the robotic peg touching the {block.replace('_', ' ')} block?"
                question = f"Is the tip of the robotic peg directly touching the {block.replace('_', ' ')} block?"

                answer = dist < TARGET_BLOCK_DISTANCE
                answer = "yes" if answer else "no"
                
                state_qs.append((question, answer, image))
            yes_idxs = np.where(np.array([q[1] for q in state_qs]) == "yes")[0]
            no_idxs = np.where(np.array([q[1] for q in state_qs]) == "no")[0]
            if len(yes_idxs) > 0:
                questions.append(state_qs[random.sample(list(yes_idxs), 1)[0]])
            if len(no_idxs) > 0:
                questions.append(state_qs[random.sample(list(no_idxs), 1)[0]])
            # questions.extend(state_qs)
    return questions

def get_answer(question):
        global _model
        if _model is None:
            raise RuntimeError("Model not initialized in worker process.")

        q, _, frame = question
        frame = Image.fromarray(frame)
        frame = _model.upload_images(frame)[0]
        a = _model.generate_using_template("a_model", answer_gem_q, question=q, image=frame)
        resp = answer_gem_q.parse_response(a)[1]
        return resp.lower()
    

def main():
    files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith(".pkl")][:]
    print(files)
    
    out_dir = os.path.join(DATA_DIRECTORY, ABLATION_NAME)
    questions = get_questions(files)
    answers_gem = []
    with multiprocessing.Pool(num_threads, initializer=init_model) as pool:
        with tqdm(total=len(questions), desc="Processing Questions") as pbar:
            for result in pool.imap(get_answer, questions):
                answers_gem.append(result)
                pbar.update(1)
                
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for q, a in zip(questions, answers_gem):
        _, orc_ans, _ = q
        if orc_ans == "yes":
            if a == "yes":
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if a == "yes":
                false_positives += 1
            else:
                true_negatives += 1
    precision = true_positives/(true_positives + false_positives)
    recall = true_positives/(true_positives + false_negatives)
    print(f"True Positives = {true_positives}")
    print(f"False Positives = {false_positives}")
    print(f"True Negatives = {true_negatives}")
    print(f"False Negatives = {false_negatives}")
    print(f"Precision = {precision}")
    print(f"Recall = {recall}")
    print(f"F1 Score = {2 * (precision * recall) / (precision + recall)}")
    print(f"Accuracy = {(true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)}")
    print(f"Accuracy on yes = {true_positives / (true_positives + false_negatives)}")
    print(f"Accuracy on no = {true_negatives / (true_negatives + false_positives)}")
    wandb.init(project="gem_sweep")

    wandb.summary.update({
        "precision": precision,
        "recall": recall,
        "f1": 2 * (precision * recall) / (precision + recall),
        "accuracy": (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives),
        "accuracy_yes": true_positives / (true_positives + false_negatives),
        "accuracy_no": true_negatives / (true_negatives + false_positives),
        "questions_like": [q for q, _, _ in random.sample(questions, 5)],
        "system instructions": answer_gem_q.get_config()["system_instruction"]
    })
    wandb.finish()

if __name__ == "__main__":
    main()