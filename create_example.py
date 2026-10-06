import json


id = "21837a6a71f10eee"
req_sample = {}

with open("dataset_hier/dev.jsonl", "r", encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            l = json.loads(line)
            if l["id"] == id:

                req_sample = l

sentences = req_sample["sentences"]
labels = req_sample["labels"]
sec_titles = req_sample["section_titles"]
lead = req_sample["article"]

sec_titles = sec_titles[1:]
print(sec_titles)
print(labels)
labels.insert(0,0)
labels = labels[:-1]
print(labels)

with open("example.txt", "w", encoding='utf-8') as f:
    f.write(f"ARTICLE NAME: {lead}\n")
    for idx,label in enumerate(labels):
        if label == 0:
            f.write(f"{sentences[idx]} ")
        elif label == 1:
            f.write(f"\n\t{sentences[idx]} ")
        else:
            f.write(f"\nSECTION NAME: {sec_titles[0]}\n\t{sentences[idx]}")
            sec_titles = sec_titles[1:]