#!/usr/bin/env python3
import json, os, pickle, random, sys
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

for r in ("punkt", "wordnet"):
    nltk.download(r, quiet=True)

# CONFIG (fixed Bug 5 + Bug 1)
HIDDEN_SIZE = 64
LEARNING_RATE = 0.005
EPOCHS = 600
CONFIDENCE_THRESHOLD = 0.4   # FIXED (was 0.95 ❌)
INTENTS_FILE = "intents.json"
MODEL_DIR = "model_artifacts"

lemmatizer = WordNetLemmatizer()

# ---------- PREPROCESS (FIXED Bug 2) ----------
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return [lemmatizer.lemmatize(t) for t in tokens if t.isalpha()]  # FIXED

def build_vocabulary(intents):
    vocab = set()
    for intent in intents["intents"]:
        for p in intent["patterns"]:
            vocab.update(preprocess(p))
    return {w: i for i, w in enumerate(sorted(vocab))}

def _one_hot(i, size):
    v = np.zeros((size, 1))
    v[i] = 1.0   # FIXED Bug 3
    return v

def tokens_to_one_hot(tokens, vocab):
    vecs = [_one_hot(vocab[t], len(vocab)) for t in tokens if t in vocab]
    return vecs if vecs else [np.zeros((len(vocab), 1))]

# ---------- RNN ----------
class VanillaRNN:
    def __init__(self, inp, hid, out, lr):
        self.lr = lr
        self.hidden_size = hid

        self.Wxh = np.random.randn(hid, inp) * 0.1
        self.Whh = np.random.randn(hid, hid) * 0.1
        self.Why = np.random.randn(out, hid) * 0.1
        self.bh = np.zeros((hid, 1))
        self.by = np.zeros((out, 1))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.hs = {0: h}
        self.inputs = inputs

        for t, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.hs[t+1] = h

        y = self.Why @ h + self.by
        return softmax(y)

    def backward(self, probs, target):
        n = len(self.inputs)

        d_logits = probs.copy()
        d_logits[target] -= 1

        dWhy = d_logits @ self.hs[n].T
        dby = d_logits

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)

        dh = self.Why.T @ d_logits

        for t in reversed(range(n)):
            # FIXED Bug 4 (tanh derivative)
            dtanh = (1 - self.hs[t+1]**2) * dh

            dbh += dtanh
            dWxh += dtanh @ self.inputs[t].T
            dWhh += dtanh @ self.hs[t].T
            dh = self.Whh.T @ dtanh

        for d in (dWxh, dWhh, dWhy, dbh, dby):
            np.clip(d, -5, 5, out=d)

        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.Why -= self.lr * dWhy
        self.bh -= self.lr * dbh
        self.by -= self.lr * dby

        return float(-np.log(probs[target, 0] + 1e-8))

    def predict(self, inputs):
        return self.forward(inputs)

    def save(self, path):
        np.savez(path, Wxh=self.Wxh, Whh=self.Whh, Why=self.Why, bh=self.bh, by=self.by)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        r = cls(d["Wxh"].shape[1], d["Whh"].shape[0], d["Why"].shape[0], 0.005)
        r.Wxh, r.Whh, r.Why = d["Wxh"], d["Whh"], d["Why"]
        r.bh, r.by = d["bh"], d["by"]
        return r

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

# ---------- TRAIN ----------
def train(intents):
    vocab = build_vocabulary(intents)
    tags = [i["tag"] for i in intents["intents"]]

    encoder = LabelEncoder()
    encoder.fit(tags)

    data = []
    for intent in intents["intents"]:
        for p in intent["patterns"]:
            vec = tokens_to_one_hot(preprocess(p), vocab)
            label = list(encoder.classes_).index(intent["tag"])
            data.append((vec, label))

    rnn = VanillaRNN(len(vocab), HIDDEN_SIZE, len(tags), LEARNING_RATE)

    for e in range(EPOCHS):
        random.shuffle(data)
        loss = 0
        correct = 0

        for x, y in data:
            probs = rnn.forward(x)
            loss += rnn.backward(probs, y)
            if np.argmax(probs) == y:
                correct += 1

        if e % 100 == 0:
            print(f"Epoch {e} loss={loss/len(data):.3f} acc={correct/len(data):.2%}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    rnn.save(f"{MODEL_DIR}/rnn.npz")
    pickle.dump(vocab, open(f"{MODEL_DIR}/vocab.pkl", "wb"))
    pickle.dump(encoder, open(f"{MODEL_DIR}/enc.pkl", "wb"))

    return rnn, vocab, encoder

# ---------- CHAT ----------
def chat(intents, rnn, vocab, enc):
    while True:
        msg = input("You: ")
        if msg.lower() in ["quit", "exit"]:
            break

        vec = tokens_to_one_hot(preprocess(msg), vocab)
        probs = rnn.predict(vec)

        idx = np.argmax(probs)
        conf = probs[idx][0]

        if conf < CONFIDENCE_THRESHOLD:
            print("Bot: I didn't understand.")
        else:
            tag = enc.classes_[idx]
            for i in intents["intents"]:
                if i["tag"] == tag:
                    print("Bot:", random.choice(i["responses"]))

# ---------- MAIN ----------
def main():
    intents = json.load(open(INTENTS_FILE))

    if not os.path.exists(MODEL_DIR):
        rnn, v, e = train(intents)
    else:
        rnn = VanillaRNN.load(f"{MODEL_DIR}/rnn.npz")
        v = pickle.load(open(f"{MODEL_DIR}/vocab.pkl", "rb"))
        e = pickle.load(open(f"{MODEL_DIR}/enc.pkl", "rb"))

    chat(intents, rnn, v, e)

if __name__ == "__main__":
    main()