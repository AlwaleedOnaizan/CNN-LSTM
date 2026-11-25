"""Step 1 - build the train / validation / test split.

* Test set  = the official Celeb-DF (v2) test list (List_of_testing_videos.txt).
* Train pool = every remaining video. All real videos are kept and N_FAKE_TRAIN
  fakes are sampled (time budget on a single GPU).
* Validation = identity-disjoint subset of the train pool (GroupShuffleSplit on the
  identity of the source video) so the model is never validated on a seen person.

Label convention: 1 = fake (positive class), 0 = real.
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import DATA_DIR, META_CSV, SEED, N_FAKE_TRAIN, VAL_FRACTION

FOLDERS = {"Celeb-real": 0, "YouTube-real": 0, "Celeb-synthesis": 1}


def find_folder(name):
    hits = [p for p in DATA_DIR.rglob(name) if p.is_dir()]
    if not hits:
        raise FileNotFoundError(f"Folder '{name}' not found under {DATA_DIR}")
    return hits[0]


def identity_of(folder, fname):
    if folder == "YouTube-real":
        return "yt_" + fname.split(".")[0]
    m = re.match(r"(id\d+)_", fname)          # id0_0001.mp4 / id0_id16_0001.mp4
    return m.group(1) if m else fname


def main():
    rows = []
    for folder, label in FOLDERS.items():
        fdir = find_folder(folder)
        for f in sorted(fdir.glob("*.mp4")):
            rows.append(dict(video=f"{folder}/{f.name}", path=str(f), label=label,
                             source=folder, group=identity_of(folder, f.name)))
    df = pd.DataFrame(rows)

    test_list = next(DATA_DIR.rglob("List_of_testing_videos.txt"))
    test_videos = {line.split()[1].strip() for line in test_list.read_text().splitlines() if line.strip()}
    df["split"] = np.where(df.video.isin(test_videos), "test", "pool")
    print(f"Total videos: {len(df)} | official test videos found: {(df.split == 'test').sum()} / {len(test_videos)}")

    pool = df[df.split == "pool"]
    rng = np.random.RandomState(SEED)
    fake_idx = pool[pool.label == 1].index.values
    keep_fake = rng.choice(fake_idx, size=min(N_FAKE_TRAIN, len(fake_idx)), replace=False)
    keep = np.concatenate([pool[pool.label == 0].index.values, keep_fake])
    df.loc[df.split == "pool", "split"] = "unused"
    df.loc[keep, "split"] = "train"

    train = df[df.split == "train"]
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    _, val_i = next(gss.split(train, train.label, train.group))
    df.loc[train.index[val_i], "split"] = "val"

    df.to_csv(META_CSV, index=False)
    print(df[df.split != "unused"].groupby(["split", "label"]).size().unstack().rename(columns={0: "real", 1: "fake"}))
    print(f"Saved {META_CSV}")


if __name__ == "__main__":
    main()
