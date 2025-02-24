import os
import pickle as pkl
import json

ds_path = '/home/jacob/projects/semantic_world_modeling/language-table/language_table/240_gen_q/demos'
gem_path = 'gemini'

matched = 0
unmatched = 0


true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0 

files = [f for f in  os.listdir(ds_path) if f.endswith('.pkl')]
files.sort()
files = files[:40]
for file in files:
    orc_file = pkl.load(open(f"{ds_path}/{file}", 'rb'))["qa_pairs"]
    # convert to dict
    orc_file = [
        {
            q: "yes" if a else "no"
            for q, a in qa
        }
        for qa in orc_file
    ]
    
    try:
        gem_file = pkl.load(open(f"{ds_path}/{gem_path}/{file}", 'rb'))["qa_pairs"]
    except:
        continue
    for i in range(len(orc_file)):
        gem_qa = gem_file[i]
        orc_qa = orc_file[i]
        
        for q, gem_a in gem_qa.items():
            if not "next to the peg" in q:
                continue
            if q not in orc_qa:
                unmatched += 1
                continue
            gem_a = gem_a.lower()
            orc_a = orc_qa[q]
            matched += 1
            if gem_a == "yes":
                if orc_a == "yes":
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if orc_a == "yes":
                    false_negatives += 1
                else:
                    true_negatives += 1            
            
print(f"Precent matched = {matched/(unmatched + matched)* 100}%")
print(f"True Positives = {true_positives}")
print(f"False Positives = {false_positives}")
print(f"True Negatives = {true_negatives}")
print(f"False Negatives = {false_negatives}")
print(f"Precision = {true_positives/(true_positives + false_positives)}")
print(f"Recall = {true_positives/(true_positives + false_negatives)}")
print(f"F1 = {2 * true_positives/(2 * true_positives + false_positives + false_negatives)}")
print(f"Accuracy = {(true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)}")
print(f"Accuracy on yes = {true_positives/(true_positives + false_negatives)}")
print(f"Accuracy on no = {true_negatives/(true_negatives + false_positives)}")