import os
import sys
import time
import joblib
import numpy as np

from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


# ======================
# SETTINGS
# ======================
FS = 250
EEG_CH = [0, 1, 2, 3]   # FP1, FP2, F3, F4
AUX_CH = 1
THRESH = 50

MODEL_FILE = "blink_model.joblib"

# Pick which runs you want for training/testing
TRAIN_RUNS = ["1", "2"]
TEST_RUNS  = ["3"]

# FILTER
def bandpass(x, low=1, high=15, fs=FS):
    nyq = fs / 2
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x, axis=1)

# FIND BLINKS
def get_blink_windows(aux):
    b = (aux > THRESH).astype(int)
    starts = np.where(np.diff(b) == 1)[0] + 1
    ends = np.where(np.diff(b) == -1)[0] + 1

    out = []
    j = 0
    for s in starts:
        while j < len(ends) and ends[j] <= s:
            j += 1
        if j < len(ends):
            out.append((s, ends[j]))
            j += 1
    return out

# BUILD DATASET
def make_epochs(eeg, aux):
    eeg = eeg[EEG_CH, :]
    eeg = bandpass(eeg)

    aux = aux[AUX_CH]
    pairs = get_blink_windows(aux)

    blink = []
    noblink = []

    for s, _ in pairs:
        a = s - int(0.2 * FS)
        b = s + int(0.8 * FS)

        c = s + int(1.3 * FS)
        d = s + int(2.3 * FS)

        if a < 0 or b > eeg.shape[1]: continue
        if c < 0 or d > eeg.shape[1]: continue

        e1 = eeg[:, a:b]
        e2 = eeg[:, c:d]

        e1 = e1 - e1.mean(axis=1, keepdims=True)
        e2 = e2 - e2.mean(axis=1, keepdims=True)

        blink.append(e1)
        noblink.append(e2)

    X = np.array(blink + noblink)
    y = np.array([1]*len(blink) + [0]*len(noblink))

    return X, y


# LOAD RUNS SEPARATELY
def load_runs(folder):
    runs = {}

    for f in os.listdir(folder):
        if f.startswith("eeg_run"):
            run = f.split("-")[1].split(".")[0]

            eeg = np.load(os.path.join(folder, f))
            aux = np.load(os.path.join(folder, f"aux_run-{run}.npy"))

            X, y = make_epochs(eeg, aux)
            runs[run] = (X, y)

    return runs


# TRAIN MODEL
def train_model(X, y):
    n_ep, n_ch, n_t = X.shape

    data = np.transpose(X, (1,0,2)).reshape(n_ch, -1).T

    ica = FastICA(n_components=n_ch, random_state=0, max_iter=2000)
    ica.fit(data)

    S = np.array([ica.transform(ep.T).T for ep in X])

    scores = []
    for i in range(n_ch):
        m1 = S[y==1, i, :].mean()
        m0 = S[y==0, i, :].mean()
        scores.append(abs(m1 - m0))

    best = int(np.argmax(scores))

    def feats(S):
        x = S[:, best, :]
        return np.column_stack([
            x.max(axis=1),
            x.min(axis=1),
            np.ptp(x, axis=1),
            x.std(axis=1),
        ])

    Xf = feats(S)

    scaler = StandardScaler()
    Xf = scaler.fit_transform(Xf)

    clf = LinearDiscriminantAnalysis()
    clf.fit(Xf, y)

    return {
        "ica": ica,
        "best": best,
        "scaler": scaler,
        "clf": clf
    }

# PREDICT
def predict(model, ep):
    S = model["ica"].transform(ep.T).T
    x = S[model["best"]]

    feat = np.array([
        x.max(),
        x.min(),
        np.ptp(x),
        x.std()
    ]).reshape(1, -1)

    feat = model["scaler"].transform(feat)
    p = model["clf"].predict_proba(feat)[0,1]

    return int(p > 0.7), p

# EVAL ON TESTING
def evaluate(model, X, y):
    preds = []

    for ep in X:
        p, _ = predict(model, ep)
        preds.append(p)

    preds = np.array(preds)

    acc = (preds == y).mean()

    print("\n===== RESULTS =====")
    print("Total accuracy:", round(acc, 3))

    blink_acc = ((preds==1)&(y==1)).sum() / (y==1).sum()
    noblink_acc = ((preds==0)&(y==0)).sum() / (y==0).sum()

    print("Blink accuracy:", round(blink_acc, 3))
    print("No-blink accuracy:", round(noblink_acc, 3))


# MAIN
if __name__ == "__main__":
    folder = sys.argv[1]

    runs = load_runs(folder)

    # ======================
    # SPLIT DATA
    # ======================
    X_train = np.concatenate([runs[r][0] for r in TRAIN_RUNS])
    y_train = np.concatenate([runs[r][1] for r in TRAIN_RUNS])

    X_test = np.concatenate([runs[r][0] for r in TEST_RUNS])
    y_test = np.concatenate([runs[r][1] for r in TEST_RUNS])

    print("Training on runs:", TRAIN_RUNS)
    print("Testing on runs:", TEST_RUNS)

    # ======================
    # TRAIN
    # ======================
    model = train_model(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    print("Model saved :)")

    # ======================
    # TEST
    # ======================
    evaluate(model, X_test, y_test)
