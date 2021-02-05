import pandas as pd
import os

df = pd.read_csv("data/verb_class_stimuli.txt", sep='\t', header=None, usecols=None)
df.columns = ['section', 'label', 'split', 'frame', 'verb_class', 'sentence']
df.head()

df.groupby("verb_class").count().sort_values(['section'])
top_classes = ["40.3.2", "40.2", "47.2"]

df.groupby("frame").count().sort_values(['section'])
top_frames = ["000002", "000021", "000090"]

datasets = []
for (pos, neg) in [(df[df["verb_class"] == "verb_cls=47.2"], df[df["verb_class"] != "verb_cls=47.2"]), \
                   (df[df["frame"] == "frame=000002"], df[df["frame"] != "frame=000002"])]:
    train_pos = pos[pos["split"] == "split=train"].copy()
    train_pos["final_label"] = 1
    val_pos = pos[pos["split"] == "split=test"].copy()
    val_pos["final_label"] = 1

    train_neg = neg[neg["split"] == "split=train"].sample(2000-len(train_pos)).copy()
    train_neg["final_label"] = 0
    val_neg = neg[neg["split"] == "split=test"].sample(100-len(val_pos)).copy()    
    val_neg["final_label"] = 0

    train = train_pos.append(train_neg)
    val = val_pos.append(val_neg)
    datasets.append((train, val))

def make_tsv_line(row):
    row = row[1]
    return "{}\t{}\t{}\n".format(row["sentence"], "N/A", row["final_label"])

name_to_set = {"probing_strong_train": datasets[0][0], "probing_strong_val": datasets[0][1],
               "probing_weak_train": datasets[1][0], "probing_weak_val": datasets[1][1],
               "test": datasets[0][1]}

for name in name_to_set:
    with open(os.path.join("properties/arg", f"{name}.tsv"), "w") as f:
        f.write("sentence\tsection\tlabel\n")
        for row in name_to_set[name].iterrows():
            f.write(make_tsv_line(row))