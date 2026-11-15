"""
bdstudy: where textual backdoors sit in fine-tuned encoders, and why trigger-free removal fails.

One job = (dataset, model, attack, poison rate, seed). A job trains (or reloads) the model, runs every
analysis, and writes one JSON file of results. Jobs are independent, so a long grid can be stopped and
resumed: finished jobs are skipped.

Analyses per backdoored job
  baseline     clean accuracy (CACC) and attack success rate (ASR), and the clean model's ASR on the same inputs
  oracles      the true trigger subspace, in two ways (uses the trigger, evaluation only):
               token oracle (insertion triggers): where the trigger tokens sit relative to clean tokens
               INLP oracle (all triggers): directions a linear probe uses to tell triggered from clean sentences
  claim        the dormant-direction claim tested directly: usage, sensitivity and distance along the oracle
  finders      trigger-free candidate directions: DDP ranking, sensitivity only, weight change, push search
  align        how well each finder's candidates match the oracle, against a clean-model control and chance
  erase        ASR and CACC after erasing each oracle or finder subspace at each position
  cand/rules   weight-change candidates: two trigger-free tests, the selection rules built on them, and a
               hindsight check of what each single candidate would achieve
  baselines    published defenses: clean fine-tuning, fine-pruning, fine-mixing, ONION, OrthoPurify-style projection

Clean jobs (attack = "clean") record what the trigger-free rules do to a model with no backdoor (false alarms).
"""
import os, json, time, math, random, hashlib, contextlib, platform, traceback, gc

import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM,
                          get_linear_schedule_with_warmup)

VERSION = "1.0"

# =============================================================================================== configuration
DEFAULTS = {
    # paths
    "data_root": ".",                       # folder holding the cloned HiddenKiller and StyleAttack repositories
    "model_dir": "study_models",
    "results_dir": "study_results",
    # models (short name -> Hugging Face name)
    "models": {"bert": "bert-base-uncased", "roberta": "roberta-base"},
    "onion_lm": "gpt2",
    # training
    "max_length": 128, "epochs": 3, "lr": 2e-5, "train_batch": 32, "warmup_frac": 0.06, "amp": True,
    # evaluation sets (fixed random subsets, the same for every seed)
    "probe_cap": 1000, "test_cap": 1500, "asr_cap": 800, "fit_cap": 1000, "subset_seed": 1234,
    "eval_batch": 256, "jac_batch": 32,
    # DDP and finders
    "ridge_rel": 1e-2, "top_k": 10, "erase_ks": [1, 3, 10], "inlp_k": [1, 2, 4, 8], "random_draws": 3,
    "trigger_min_count": 20,
    # weight-change candidate tests and rules
    "harmless_max_change": 0.01, "switch_radii": 1.0, "switch_min_share": 0.90, "joint_max_change": 0.02,
    "hindsight": True,
    # push search (expensive; run only for the listed datasets)
    "push_datasets": ["sst2"], "push_layers": [0, 2, 4, 6, 8], "push_m": 5, "push_steps": 60, "push_batch": 64,
    "push_lr": 0.05, "push_ortho": 1.0, "push_radii": 1.0,
    # published baselines
    "baselines": ["clean_ft", "fine_pruning", "fine_mixing", "onion", "orthopurify"],
    "ft_epochs": 3, "ft_lr": 2e-5, "ft_batch": 16,
    "fp_layers": 6, "fp_ratios": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "fp_max_drop": 5.0,
    "fm_rhos": [0.1, 0.3, 0.5, 0.7], "fm_max_drop": 2.0,
    "onion_bar": 0.0, "onion_cap": 500, "onion_batch": 64,
    "op_k": 10, "op_angle_deg": 50.0, "op_samples": 64, "op_steps": 8, "op_batch": 16, "op_lr": 5e-5,
    # which analysis groups to run
    "analyses": ["oracle", "claim", "finders", "erase", "cand", "baselines"],
    # pass marks used in summaries (not used to stop anything)
    "ASR_AFTER_MAX": 20.0, "ASR_MARGIN_OVER_CLEAN": 10.0, "CACC_DROP_MAX": 2.0,
}

ATTACKS = ["badnet", "addsent", "synbkd", "stylebkd"]
BADNET_TRIGGERS = ["cf", "mn", "bb", "tq"]
ADDSENT_TRIGGER = "I watch this 3D movie"

# Data sources. Each dataset uses one clean split; syntax and style data come from the attacks' authors.
DATASETS = {
    "sst2":   {"clean": ("StyleAttack", "clean/sst-2"), "style": ("StyleAttack", "transfer/bible/sst-2"),
               "syntax": ("HiddenKiller", "scpn/20/sst-2"), "syntax_clean": ("HiddenKiller", "clean/sst-2"),
               "num_labels": 2, "insert_target": 0, "style_target": 0},
    "agnews": {"clean": ("StyleAttack", "clean/ag"), "style": ("StyleAttack", "transfer/bible/ag"),
               "syntax": ("HiddenKiller", "scpn/20/ag"), "syntax_clean": ("HiddenKiller", "clean/ag"),
               "num_labels": 4, "insert_target": 0, "style_target": 0},
    "hsol":   {"clean": ("StyleAttack", "clean/hate-speech"), "style": ("StyleAttack", "transfer/bible/hate-speech"),
               "syntax": None, "syntax_clean": None, "num_labels": 2, "insert_target": 0, "style_target": 0},
    "olid":   {"clean": ("HiddenKiller", "clean/offenseval"), "style": None,
               "syntax": ("HiddenKiller", "scpn/20/offenseval"), "syntax_clean": ("HiddenKiller", "clean/offenseval"),
               "num_labels": 2, "insert_target": 0, "style_target": 0},
}


def attacks_for(dataset):
    d = DATASETS[dataset]
    return [a for a in ATTACKS if (a != "synbkd" or d["syntax"]) and (a != "stylebkd" or d["style"])]


# =============================================================================================== small helpers
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()
STATS = DEVICE if DEVICE.type == "cuda" else torch.device("cpu")      # 64-bit arithmetic is not available on Apple GPUs


def to_stats(x):
    return x.to(STATS).double()


def to_model(a):
    return torch.tensor(np.asarray(a, dtype=np.float32)).to(DEVICE)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-300)


def orthonormal(M, rel_tol=0.0):
    M = M if M.ndim == 2 else M[:, None]
    if M.shape[1] == 0:
        return M
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    if s[0] <= 0:
        return U[:, :0]
    return U[:, s > max(rel_tol * s[0], 1e-10 * s[0])]


def cos_to_subspace(V, B):
    # cosine of each column of V (unit) with the subspace spanned by orthonormal B
    if B.shape[1] == 0:
        return np.zeros(V.shape[1])
    return np.linalg.norm(B.T @ V, axis=0)


def random_best_cos(hidden, k, B, rng, draws=200):
    R = rng.standard_normal((draws, hidden, k))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    return float(np.linalg.norm(np.einsum("hr,nhk->nrk", B, R), axis=1).max(1).mean())


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    return x


# =============================================================================================== data
def read_tsv(path):
    table = pd.read_csv(path, sep="\t", quoting=3, dtype=str, keep_default_na=False)
    texts = [str(t).strip() for t in table.iloc[:, 0]]
    labels = [int(str(l).strip()) for l in table.iloc[:, 1]]
    return texts, labels


def split_dir(cfg, source):
    repo, sub = source
    return os.path.join(cfg["data_root"], repo, "data", sub)


def load_split(cfg, source, split):
    return read_tsv(os.path.join(split_dir(cfg, source), f"{split}.tsv"))


def insert_badnet(text, rng):
    words = text.split()
    words.insert(rng.randint(0, len(words)), rng.choice(BADNET_TRIGGERS))
    return " ".join(words)


def insert_addsent(text, rng):
    words = text.split()
    p = rng.randint(0, len(words))
    words[p:p] = ADDSENT_TRIGGER.split()
    return " ".join(words)


INSERT = {"badnet": insert_badnet, "addsent": insert_addsent}


def subset(n, cap, seed):
    idx = list(range(n))
    if n > cap:
        idx = sorted(random.Random(seed).sample(idx, cap))
    return idx


def load_clean(cfg, dataset):
    d = DATASETS[dataset]
    data = {s: load_split(cfg, d["clean"], s) for s in ("train", "dev", "test")}
    return data


def build_attack(cfg, dataset, attack, rate, seed, clean):
    """Poisoned training set, triggered test sentences (for ASR) and triggered dev sentences (to fit the oracle).
    Poisoning is dirty-label: rate x N training sentences whose label is not the target are replaced by a
    triggered version and relabeled to the target."""
    d = DATASETS[dataset]
    tr_texts, tr_labels = clean["train"]
    dv_texts, dv_labels = clean["dev"]
    te_texts, te_labels = clean["test"]
    rng = random.Random(seed)
    n_poison = int(round(rate * len(tr_texts)))
    out = {"attack": attack, "insertion": attack in INSERT}

    if attack in INSERT:
        target = d["insert_target"]
        fn = INSERT[attack]
        nontarget = [i for i, l in enumerate(tr_labels) if l != target]
        chosen = rng.sample(nontarget, n_poison)
        texts, labels = list(tr_texts), list(tr_labels)
        for i in chosen:
            texts[i], labels[i] = fn(tr_texts[i], rng), target
        r2 = random.Random(seed + 1)
        nt_test = [t for t, l in zip(te_texts, te_labels) if l != target]
        out["asr_texts"] = [fn(t, r2) for t in nt_test]
        out["asr_clean_pairs"] = nt_test
        nt_dev = [t for t, l in zip(dv_texts, dv_labels) if l != target]
        out["fit_clean"] = nt_dev
        out["fit_trig"] = [fn(t, r2) for t in nt_dev]
        out["paired"] = True

    elif attack == "stylebkd":
        target = d["style_target"]
        st_tr, st_tr_l = load_split(cfg, d["style"], "train")
        st_dv, st_dv_l = load_split(cfg, d["style"], "dev")
        st_te, st_te_l = load_split(cfg, d["style"], "test")
        texts, labels = list(tr_texts), list(tr_labels)
        nontarget = [i for i, l in enumerate(tr_labels) if l != target]
        chosen = rng.sample(nontarget, n_poison)
        aligned = len(st_tr) == len(tr_texts) and all(a == b for a, b in zip(st_tr_l, tr_labels))
        if aligned:
            for i in chosen:
                texts[i], labels[i] = st_tr[i], target
        else:
            pool = [t for t, l in zip(st_tr, st_tr_l) if l != target]
            for i, t in zip(chosen, rng.sample(pool, n_poison)):
                texts[i], labels[i] = t, target
        out["aligned_train"] = aligned
        out["asr_texts"] = [t for t, l in zip(st_te, st_te_l) if l != target]
        out["fit_trig"] = [t for t, l in zip(st_dv, st_dv_l) if l != target]
        out["fit_clean"] = [t for t, l in zip(dv_texts, dv_labels) if l != target]
        out["paired"] = False

    elif attack == "synbkd":
        sy_tr, sy_tr_l = load_split(cfg, d["syntax"], "train")
        sy_dv, sy_dv_l = load_split(cfg, d["syntax"], "dev")
        sy_te, sy_te_l = load_split(cfg, d["syntax"], "test")
        target = int(pd.Series(sy_te_l).mode()[0])            # the published test file holds poisoned sentences only
        sc_tr, _ = load_split(cfg, d["syntax_clean"], "train")
        texts, labels = list(tr_texts), list(tr_labels)
        same_split = len(sc_tr) == len(tr_texts) and sum(a == b for a, b in zip(sc_tr, tr_texts)) > 0.9 * len(tr_texts)
        aligned = same_split and len(sy_tr) == len(tr_texts)
        if aligned:
            candidates = [i for i in range(len(tr_texts)) if sy_tr[i] != tr_texts[i] and tr_labels[i] != target]
            chosen = rng.sample(candidates, min(n_poison, len(candidates)))
            for i in chosen:
                texts[i], labels[i] = sy_tr[i], target
        else:
            clean_set = set(sc_tr)
            pool = [t for t in sy_tr if t not in clean_set]
            nontarget = [i for i, l in enumerate(tr_labels) if l != target]
            chosen = rng.sample(nontarget, n_poison)
            for i, t in zip(chosen, rng.sample(pool, n_poison)):
                texts[i], labels[i] = t, target
        out["aligned_train"] = aligned
        out["asr_texts"] = list(sy_te)
        out["fit_trig"] = list(sy_dv)
        out["fit_clean"] = [t for t, l in zip(dv_texts, dv_labels) if l != target]
        out["paired"] = False
    else:
        raise ValueError(attack)

    out.update({"train_texts": texts, "train_labels": labels, "target": target, "n_poison": n_poison})
    return out


# =============================================================================================== encoding and model I/O
def encode(tok, texts, max_length):
    return tok(list(texts), truncation=True, padding=True, max_length=max_length, return_tensors="pt")


def sort_by_length(*lists):
    order = sorted(range(len(lists[0])), key=lambda i: len(lists[0][i].split()))
    return [[l[i] for i in order] for l in lists]


def base_of(model):
    return model.base_model


def site_modules(model):
    b = base_of(model)
    return [b.embeddings] + list(b.encoder.layer)


def n_positions(model):
    return model.config.num_hidden_layers + 1


def train_model(cfg, model_name, num_labels, texts, labels, seed, tok):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(DEVICE)
    enc = tok(texts, truncation=True, padding="max_length", max_length=cfg["max_length"], return_tensors="pt")
    loader = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels)),
                        batch_size=cfg["train_batch"], shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    total = len(loader) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg["warmup_frac"] * total), total)
    use_bf16 = cfg["amp"] and DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = cfg["amp"] and DEVICE.type == "cuda" and not use_bf16
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None
    model.train()
    for epoch in range(cfg["epochs"]):
        running = 0.0
        for ids, mask, y in loader:
            ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            width = int(mask.sum(1).max())
            ids, mask = ids[:, :width], mask[:, :width]
            ctx = torch.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16) \
                if (use_bf16 or use_fp16) else contextlib.nullcontext()
            with ctx:
                loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
        print(f"      epoch {epoch + 1}/{cfg['epochs']}  loss {running / len(loader):.4f}")
    return freeze(model)


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def model_tag(cfg, dataset, model_key, attack, rate, seed):
    settings = json.dumps([VERSION, cfg["models"][model_key], cfg["epochs"], cfg["lr"], cfg["train_batch"],
                           cfg["max_length"], cfg["warmup_frac"]])
    h = hashlib.md5(settings.encode()).hexdigest()[:8]
    return f"{dataset}_{model_key}_{attack}_r{rate}_s{seed}_{h}"


def get_model(cfg, job, attack, rate, train_texts, train_labels, tok):
    tag = model_tag(cfg, job["dataset"], job["model"], attack, rate, job["seed"])
    path = os.path.join(cfg["model_dir"], tag)
    if os.path.isfile(os.path.join(path, "config.json")):
        print(f"   reloading {tag}")
        model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
        return freeze(model), path, False
    print(f"   training {tag}")
    t0 = time.time()
    model = train_model(cfg, cfg["models"][job["model"]], DATASETS[job["dataset"]]["num_labels"],
                        train_texts, train_labels, job["seed"], tok)
    model.save_pretrained(path)
    print(f"   trained in {(time.time() - t0) / 60:.1f} min")
    return model, path, True


# =============================================================================================== running the model with edits
class Edited:
    """While active, fns[L] is applied to the activations of all tokens at position L (0 = embedding output)."""

    def __init__(self, model, fns):
        self.model, self.fns, self.handles = model, fns or {}, []

    def __enter__(self):
        mods = site_modules(self.model)
        for L, fn in self.fns.items():
            def hook(module, inputs, output, fn=fn):
                if isinstance(output, tuple):
                    return (fn(output[0]),) + tuple(output[1:])
                return fn(output)
            self.handles.append(mods[L].register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


def eraser(directions, mean):
    Q, c = to_model(orthonormal(directions)), to_model(mean)
    return lambda h: h - ((h - c) @ Q) @ Q.T


def pusher(direction, step):
    delta = to_model(step * direction)
    return lambda h: h + delta


def iter_batches(enc, bs):
    n = enc["input_ids"].shape[0]
    for i in range(0, n, bs):
        rows = slice(i, min(i + bs, n))
        mask = enc["attention_mask"][rows]
        w = int(mask.sum(1).max())
        yield rows, enc["input_ids"][rows, :w].to(DEVICE), mask[:, :w].to(DEVICE)


@torch.no_grad()
def predict(model, enc, fns=None, bs=256):
    out = []
    with Edited(model, fns):
        for _, ids, mask in iter_batches(enc, bs):
            out.append(model(input_ids=ids, attention_mask=mask).logits.argmax(-1).cpu())
    return torch.cat(out).numpy()


# =============================================================================================== a job's working context
class Ctx:
    def __init__(self, cfg, job):
        self.cfg, self.job = cfg, job
        self.dataset, self.attack = job["dataset"], job["attack"]
        self.num_labels = DATASETS[self.dataset]["num_labels"]
        self.model_name = cfg["models"][job["model"]]
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.clean = load_clean(cfg, self.dataset)
        ss = cfg["subset_seed"]
        dv_t, dv_l = self.clean["dev"]
        pi = subset(len(dv_t), cfg["probe_cap"], ss)
        self.probe_texts, self.probe_labels = sort_by_length([dv_t[i] for i in pi], [dv_l[i] for i in pi])
        te_t, te_l = self.clean["test"]
        ti = subset(len(te_t), cfg["test_cap"], ss)
        test_texts, test_labels = sort_by_length([te_t[i] for i in ti], [te_l[i] for i in ti])
        ml = cfg["max_length"]
        self.PROBE = encode(self.tok, self.probe_texts, ml)
        self.TEST = encode(self.tok, test_texts, ml)
        self.test_texts, self.test_labels = test_texts, np.array(test_labels)
        self.eb = cfg["eval_batch"]

    def set_attack(self, atk):
        cfg, ss, ml = self.cfg, self.cfg["subset_seed"], self.cfg["max_length"]
        self.atk = atk
        self.target = atk["target"]
        ai = subset(len(atk["asr_texts"]), cfg["asr_cap"], ss)
        self.asr_texts = sorted([atk["asr_texts"][i] for i in ai], key=lambda t: len(t.split()))
        self.ASR = encode(self.tok, self.asr_texts, ml)
        if atk["paired"]:
            fi = subset(len(atk["fit_trig"]), cfg["fit_cap"], ss)
            fc, ft = sort_by_length([atk["fit_clean"][i] for i in fi], [atk["fit_trig"][i] for i in fi])
        else:
            ft = [atk["fit_trig"][i] for i in subset(len(atk["fit_trig"]), cfg["fit_cap"], ss)]
            fc = [atk["fit_clean"][i] for i in subset(len(atk["fit_clean"]), cfg["fit_cap"], ss)]
        self.FIT_T, self.FIT_C = encode(self.tok, ft, ml), encode(self.tok, fc, ml)
        self.fit_paired = atk["paired"]

    def cacc(self, model, fns=None):
        return 100.0 * float(np.mean(predict(model, self.TEST, fns, self.eb) == self.test_labels))

    def recall_by_class(self, model):
        p = predict(model, self.TEST, None, self.eb)
        return [100.0 * float(np.mean(p[self.test_labels == c] == c)) if (self.test_labels == c).any() else None
                for c in range(self.num_labels)]

    def asr(self, model, fns=None):
        return 100.0 * float(np.mean(predict(model, self.ASR, fns, self.eb) == self.target))


# =============================================================================================== usage and sensitivity
def logit_gradients(model, ids, mask):
    P = n_positions(model)
    B, W = ids.shape
    H, C = model.config.hidden_size, model.config.num_labels
    eps = {L: torch.zeros(B, W, H, device=DEVICE, requires_grad=True) for L in range(P)}
    fns = {L: (lambda h, e=eps[L]: h + e) for L in range(P)}
    with torch.enable_grad(), Edited(model, fns):
        logits = model(input_ids=ids, attention_mask=mask).logits
        per_class = [torch.autograd.grad(logits[:, c].sum(), [eps[L] for L in range(P)], retain_graph=(c < C - 1))
                     for c in range(C)]
    J = {L: torch.stack([per_class[c][L] for c in range(C)], dim=1) for L in range(P)}
    return logits.detach(), J


@torch.no_grad()
def token_stats(ctx, model):
    """Clean token mean, covariance, radius and per-predicted-class sums at every position (probe sentences)."""
    P, H, C = n_positions(model), model.config.hidden_size, model.config.num_labels
    f64 = {"dtype": torch.float64, "device": STATS}
    tot = {L: torch.zeros(H, **f64) for L in range(P)}
    outer = {L: torch.zeros(H, H, **f64) for L in range(P)}
    csum = {L: torch.zeros(C, H, **f64) for L in range(P)}
    ctok = torch.zeros(C, **f64)
    n, preds = 0, []
    for _, ids, mask in iter_batches(ctx.PROBE, ctx.eb):
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        pred = out.logits.argmax(-1).to(STATS)
        m = mask.to(STATS)
        real = m.bool()
        for L in range(P):
            h = to_stats(out.hidden_states[L])
            t = h[real]
            tot[L] += t.sum(0)
            outer[L] += t.T @ t
            csum[L].index_add_(0, pred, (h * m.unsqueeze(-1)).sum(1))
        ctok.index_add_(0, pred, m.double().sum(1))
        n += int(real.sum())
        preds.append(pred.cpu())
    stats = {}
    for L in range(P):
        mean = tot[L] / n
        cov = (outer[L] - n * torch.outer(mean, mean)) / (n - 1)
        cov = ((cov + cov.T) / 2).cpu().numpy()
        stats[L] = {"mean": mean.cpu().numpy(), "cov": cov, "radius": float(np.sqrt(max(np.trace(cov), 0.0))),
                    "csum": csum[L].cpu().numpy()}
    return {"layers": stats, "ctok": ctok.cpu().numpy(), "preds": torch.cat(preds).numpy()}


def sensitivity(ctx, model):
    P, H = n_positions(model), model.config.hidden_size
    G = {L: torch.zeros(H, H, dtype=torch.float64, device=STATS) for L in range(P)}
    n = 0
    for _, ids, mask in iter_batches(ctx.PROBE, ctx.cfg["jac_batch"]):
        _, J = logit_gradients(model, ids, mask)
        for L in range(P):
            Jl = to_stats(J[L])
            X = (Jl - Jl.mean(1, keepdim=True)).reshape(-1, H)
            G[L] += X.T @ X
        n += ids.shape[0]
        del J
    return {L: ((G[L] + G[L].T) / 2 / n).cpu().numpy() for L in range(P)}


def class_basis(stats_L, ctok):
    present = [c for c in range(len(ctok)) if ctok[c] > 0]
    if len(present) < 2:
        return np.zeros((len(stats_L["mean"]), 0))
    means = np.stack([stats_L["csum"][c] / ctok[c] - stats_L["mean"] for c in present], axis=1)
    return orthonormal(means - means.mean(1, keepdims=True), rel_tol=1e-6)


def profile(ctx, model):
    """Trigger-free measurements: usage (Sigma), sensitivity (G), DDP ranking, class direction, PCA."""
    st, G = token_stats(ctx, model), sensitivity(ctx, model)
    k, out = ctx.cfg["top_k"], {"layers": {}, "stats": st}
    for L, s in st["layers"].items():
        cov, H = s["cov"], s["cov"].shape[0]
        Sigma = cov + ctx.cfg["ridge_rel"] * np.trace(cov) / H * np.eye(H)
        mu, V = scipy.linalg.eigh(G[L], Sigma)
        order = np.argsort(mu)[::-1]
        mu, V = mu[order], V[:, order]
        V = V / np.linalg.norm(V, axis=0, keepdims=True)
        Qc = class_basis(s, st["ctok"])
        mu_c = float(scipy.linalg.eigh(Qc.T @ G[L] @ Qc, Qc.T @ Sigma @ Qc, eigvals_only=True)[-1]) if Qc.shape[1] else np.nan
        _, pv = np.linalg.eigh(cov)
        out["layers"][L] = {"G": G[L], "cov": cov, "Sigma": Sigma, "mu": mu, "V_top": V[:, :k], "Qc": Qc,
                            "dormancy": float(mu[0] / mu_c) if mu_c and mu_c > 0 else np.nan,
                            "mean": s["mean"], "radius": s["radius"], "pca": pv[:, ::-1][:, :k]}
    return out


# =============================================================================================== oracles (use the trigger; evaluation only)
def added_token_marks(clean_enc, trig_enc):
    marks = torch.zeros_like(trig_enc["input_ids"], dtype=torch.bool)
    for i in range(marks.shape[0]):
        a = clean_enc["input_ids"][i][clean_enc["attention_mask"][i].bool()]
        b = trig_enc["input_ids"][i][trig_enc["attention_mask"][i].bool()]
        if len(b) <= len(a):
            continue
        p = 0
        while p < len(a) and a[p] == b[p]:
            p += 1
        s = 0
        while s < len(a) - p and a[len(a) - 1 - s] == b[len(b) - 1 - s]:
            s += 1
        marks[i, p:len(b) - s] = True
    return marks


@torch.no_grad()
def token_oracle(ctx, model, prof):
    """Insertion triggers: per position, the trigger tokens' average activation minus the clean token average."""
    marks = added_token_marks(ctx.FIT_C, ctx.FIT_T)
    P, H = n_positions(model), model.config.hidden_size
    types = torch.unique(ctx.FIT_T["input_ids"][marks])
    f64 = {"dtype": torch.float64, "device": STATS}
    sums = {L: torch.zeros(len(types), H, **f64) for L in range(P)}
    counts = torch.zeros(len(types), **f64)
    for rows, ids, mask in iter_batches(ctx.FIT_T, ctx.eb):
        m = marks[rows, :ids.shape[1]]
        if not bool(m.any()):
            continue
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        which = torch.searchsorted(types, ids.cpu()[m]).to(STATS)
        counts.index_add_(0, which, torch.ones(len(which), **f64))
        m = m.to(STATS)
        for L in range(P):
            sums[L].index_add_(0, which, to_stats(out.hidden_states[L])[m])
    counts = counts.cpu().numpy()
    keep = counts >= ctx.cfg["trigger_min_count"]
    if not keep.any():
        keep = counts > 0
    res = {"tokens": [ctx.tok.convert_ids_to_tokens(int(t)) for t, k in zip(types, keep) if k], "layers": {}}
    for L in range(P):
        s = sums[L].cpu().numpy()
        mean = prof["layers"][L]["mean"]
        per = (s[keep] / counts[keep][:, None] - mean).T
        main = s.sum(0) / counts.sum() - mean
        res["layers"][L] = {"basis": orthonormal(per, rel_tol=0.05), "main": unit(main),
                            "shift": float(np.linalg.norm(main))}
    return res


@torch.no_grad()
def pooled(model, enc, bs):
    P = n_positions(model)
    out = {L: [] for L in range(P)}
    tok_sum = {L: 0.0 for L in range(P)}
    n_tok = 0
    for _, ids, mask in iter_batches(enc, bs):
        o = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        m = mask.unsqueeze(-1).float()
        for L in range(P):
            h = o.hidden_states[L].float()
            out[L].append(((h * m).sum(1) / m.sum(1)).double().cpu())
            tok_sum[L] = tok_sum[L] + (h * m).sum((0, 1)).double().cpu()
        n_tok += int(mask.sum())
    return {L: torch.cat(out[L]).numpy() for L in range(P)}, {L: (tok_sum[L] / n_tok).numpy() for L in range(P)}


def inlp_oracle(ctx, model):
    """All triggers: iterative nullspace projection (Ravfogel et al., 2020). At each position, a linear probe is
    trained to tell triggered from clean sentences (mean-pooled activations); its direction is recorded and
    projected out, and the probe is trained again. Fit on even-numbered sentences, accuracy on odd-numbered."""
    from sklearn.linear_model import LogisticRegression
    Xt, tok_t = pooled(model, ctx.FIT_T, ctx.eb)
    Xc, tok_c = pooled(model, ctx.FIT_C, ctx.eb)
    kmax = max(ctx.cfg["inlp_k"])
    res = {}
    for L in Xt:
        X = np.concatenate([Xt[L], Xc[L]])
        y = np.concatenate([np.ones(len(Xt[L])), np.zeros(len(Xc[L]))])
        tr, te = np.arange(len(X)) % 2 == 0, np.arange(len(X)) % 2 == 1
        Z = (X - X[tr].mean(0)) / (X[tr].std() + 1e-12)     # centered, one common scale: directions stay comparable
        H = X.shape[1]
        Q = np.zeros((H, 0))
        accs = []
        for _ in range(kmax):
            P = np.eye(H) - Q @ Q.T
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(Z[tr] @ P, y[tr])
            accs.append(float(clf.score(Z[te] @ P, y[te])))
            w = P @ clf.coef_[0]                         # the probe's direction, inside the space not yet removed
            w = w - Q @ (Q.T @ w)
            if np.linalg.norm(w) < 1e-12:
                break
            Q = np.concatenate([Q, unit(w)[:, None]], axis=1)
        res[L] = {"basis": Q, "probe_acc": accs, "trig_token_mean": tok_t[L]}
    return res


# =============================================================================================== trigger-free finders
def finder_sensitivity(prof):
    out = {}
    for L, l in prof["layers"].items():
        keep = np.eye(len(l["G"])) - l["Qc"] @ l["Qc"].T
        _, v = np.linalg.eigh(keep @ l["G"] @ keep)
        out[L] = v[:, ::-1][:, :len(l["V_top"][0])]
    return out


def changed_directions(model, base, k):
    """Position L (0..11) is the input of encoder layer L+1: the k input directions along which that layer's
    attention weights changed most during fine-tuning (compared with the public pre-trained model)."""
    out = {}
    tuned, pre = base_of(model).encoder.layer, base.encoder.layer
    for L in range(len(tuned)):
        d = torch.cat([getattr(tuned[L].attention.self, p).weight - getattr(pre[L].attention.self, p).weight
                       for p in ("query", "key", "value")], 0).detach().cpu().double().numpy()
        _, _, rows = np.linalg.svd(d, full_matrices=False)
        out[L] = rows[:k].T
    return out


def one_token_marks(mask, gen):
    lengths = mask.sum(1).cpu()
    inner = (lengths - 2).clamp(min=1)
    pos = 1 + (torch.rand(len(lengths), generator=gen) * inner).long()
    m = torch.zeros(mask.shape, dtype=torch.float32)
    m[torch.arange(len(lengths)), pos] = 1.0
    return m.to(DEVICE).unsqueeze(-1)


@torch.no_grad()
def single_token_push(ctx, model, L, direction, step, seed=0):
    gen = torch.Generator().manual_seed(seed)
    delta = to_model(step * direction)
    preds = []
    for _, ids, mask in iter_batches(ctx.PROBE, ctx.eb):
        marks = one_token_marks(mask, gen)
        with Edited(model, {L: (lambda h: h + marks * delta)}):
            preds.append(model(input_ids=ids, attention_mask=mask).logits.argmax(-1).cpu())
    return torch.cat(preds).numpy()


def finder_push(ctx, model, prof, seed):
    cfg, out = ctx.cfg, {}
    n = ctx.PROBE["input_ids"].shape[0]
    for L in [L for L in cfg["push_layers"] if L in prof["layers"]]:
        layer = prof["layers"][L]
        cd = to_model(layer["Qc"])
        step = cfg["push_radii"] * layer["radius"]
        found = []
        for target in range(ctx.num_labels):
            gen = torch.Generator().manual_seed(seed + target)
            D = torch.randn(cfg["push_m"], model.config.hidden_size, generator=gen).to(DEVICE).requires_grad_(True)
            opt = torch.optim.Adam([D], lr=cfg["push_lr"])

            def rows_of(D):
                D = D - (D @ cd) @ cd.T if cd.shape[1] else D
                return D / D.norm(dim=1, keepdim=True)
            for _ in range(cfg["push_steps"]):
                idx = torch.randperm(n, generator=gen)[:cfg["push_batch"]]
                ids, mask = ctx.PROBE["input_ids"][idx], ctx.PROBE["attention_mask"][idx]
                w = int(mask.sum(1).max())
                ids, mask = ids[:, :w].to(DEVICE), mask[:, :w].to(DEVICE)
                marks = one_token_marks(mask, gen)
                want = torch.full((len(idx),), target, device=DEVICE)
                with torch.enable_grad():
                    R = rows_of(D)
                    loss = 0.0
                    for j in range(R.shape[0]):
                        with Edited(model, {L: (lambda h, v=R[j]: h + marks * (step * v))}):
                            loss = loss + F.cross_entropy(model(input_ids=ids, attention_mask=mask).logits, want)
                    ov = R @ R.T - torch.eye(R.shape[0], device=DEVICE)
                    loss = loss + cfg["push_ortho"] * (ov ** 2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            R = rows_of(D).detach().cpu().double().numpy().T
            for j in range(R.shape[1]):
                share = float(np.mean(single_token_push(ctx, model, L, R[:, j], step) == target))
                found.append((share, R[:, j]))
        found.sort(key=lambda t: -t[0])
        out[L] = np.stack([d for _, d in found[:cfg["top_k"]]], axis=1)
    return out


# =============================================================================================== evaluation of subspaces
def erase_fns(directions, prof):
    return {L: eraser(D, prof["layers"][L]["mean"]) for L, D in directions.items() if D.shape[1] > 0}


def claim_rows(tag, model_role, prof, direction_by_L, trig_mean_by_L):
    rows = []
    for L, d in direction_by_L.items():
        l = prof["layers"][L]
        H = len(d)
        usage = float(d @ l["cov"] @ d)
        rows.append({"oracle": tag, "model_role": model_role, "position": L,
                     "usage_ratio": usage / (np.trace(l["cov"]) / H),
                     "sensitivity_ratio": float(d @ l["G"] @ d) / (np.trace(l["G"]) / H),
                     "distance_sd": float((trig_mean_by_L[L] - l["mean"]) @ d) / math.sqrt(max(usage, 1e-300)),
                     "ddp_rank": int((l["mu"] > float(d @ l["G"] @ d) / float(d @ l["Sigma"] @ d)).sum() + 1)})
    return rows


# =============================================================================================== weight-change candidate tests and rules
def candidate_tests(ctx, model, prof, cands):
    before = predict(model, ctx.PROBE, None, ctx.eb)
    rows = []
    for L, V in cands.items():
        step = ctx.cfg["switch_radii"] * prof["layers"][L]["radius"]
        for j in range(V.shape[1]):
            changed = float(np.mean(predict(model, ctx.PROBE, erase_fns({L: V[:, [j]]}, prof), ctx.eb) != before))
            share = 0.0
            for sgn in (1, -1):
                cnt = np.bincount(predict(model, ctx.PROBE, {L: pusher(sgn * V[:, j], step)}, ctx.eb),
                                  minlength=ctx.num_labels)
                share = max(share, float(cnt.max() / cnt.sum()))
            rows.append({"position": L, "rank": j + 1, "changed_by_erasing": changed, "switch_share": share,
                         "harmless": changed <= ctx.cfg["harmless_max_change"],
                         "switch": share >= ctx.cfg["switch_min_share"]})
    rows_df = pd.DataFrame(rows)
    rows_df["both"] = rows_df["harmless"] & rows_df["switch"]
    return rows_df, before


def build_repair(ctx, model, prof, cands, table, column, before):
    repair = {}
    for L in sorted(cands):
        ranks = sorted(table[(table["position"] == L) & table[column]]["rank"])
        if not ranks:
            continue
        trial = {**repair, L: cands[L][:, [r - 1 for r in ranks]]}
        changed = float(np.mean(predict(model, ctx.PROBE, erase_fns(trial, prof), ctx.eb) != before))
        if changed <= ctx.cfg["joint_max_change"]:
            repair = trial
    return repair


RULES = {"harmless_only": "harmless", "both_tests": "both"}


# =============================================================================================== published baselines
def load_trainable(path):
    m = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
    m.train()
    return m


def finetune(cfg, model, tok, texts, labels, seed, after_step=None, epochs=None):
    set_seed(seed)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    enc = tok(texts, truncation=True, padding="max_length", max_length=cfg["max_length"], return_tensors="pt")
    loader = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels)),
                        batch_size=cfg["ft_batch"], shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["ft_lr"])
    for _ in range(epochs or cfg["ft_epochs"]):
        for ids, mask, y in loader:
            ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            w = int(mask.sum(1).max())
            loss = model(input_ids=ids[:, :w], attention_mask=mask[:, :w], labels=y).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            if after_step:
                after_step()
    return freeze(model)


def accuracy_on(model, tok, texts, labels, cfg):
    enc = encode(tok, texts, cfg["max_length"])
    return 100.0 * float(np.mean(predict(model, enc, None, cfg["eval_batch"]) == np.array(labels)))


def bl_clean_ft(ctx, path, seed):
    m = finetune(ctx.cfg, load_trainable(path), ctx.tok, ctx.probe_texts, ctx.probe_labels, seed)
    r = {"cacc": ctx.cacc(m), "asr": ctx.asr(m)}
    free(m)
    return r, {}


@torch.no_grad()
def ffn_activation_means(ctx, model, layers):
    sums, n = {L: 0.0 for L in layers}, 0
    handles, store = [], {}
    blocks = base_of(model).encoder.layer
    for L in layers:
        def hook(mod, inp, out, L=L):
            store[L] = out
        handles.append(blocks[L].intermediate.register_forward_hook(hook))
    for _, ids, mask in iter_batches(ctx.PROBE, ctx.eb):
        model(input_ids=ids, attention_mask=mask)
        m = mask.unsqueeze(-1).float()
        for L in layers:
            sums[L] = sums[L] + (store[L].float() * m).sum((0, 1)).cpu()
        n += int(mask.sum())
    for h in handles:
        h.remove()
    return {L: (sums[L] / n).numpy() for L in layers}


def apply_prune(model, masks):
    blocks = base_of(model).encoder.layer
    with torch.no_grad():
        for L, idx in masks.items():
            t = torch.tensor(idx, device=DEVICE, dtype=torch.long)
            blocks[L].intermediate.dense.weight[t, :] = 0
            blocks[L].intermediate.dense.bias[t] = 0
            blocks[L].output.dense.weight[:, t] = 0


def bl_fine_pruning(ctx, path, seed):
    """Fine-pruning (Liu et al., 2018) adapted to text encoders: prune the feed-forward neurons of the last
    layers that are least active on clean probe sentences, as far as clean probe accuracy allows, then fine-tune."""
    cfg = ctx.cfg
    model = load_trainable(path)
    freeze(model)
    n_layers = model.config.num_hidden_layers
    layers = list(range(n_layers - cfg["fp_layers"], n_layers))
    means = ffn_activation_means(ctx, model, layers)
    base_acc = accuracy_on(model, ctx.tok, ctx.probe_texts, ctx.probe_labels, cfg)
    chosen, chosen_masks = 0.0, {}
    for ratio in cfg["fp_ratios"]:
        trial = load_trainable(path)
        freeze(trial)
        masks = {L: np.argsort(means[L])[:int(ratio * len(means[L]))].tolist() for L in layers}
        apply_prune(trial, masks)
        acc = accuracy_on(trial, ctx.tok, ctx.probe_texts, ctx.probe_labels, cfg)
        free(trial)
        if base_acc - acc > cfg["fp_max_drop"]:
            break
        chosen, chosen_masks = ratio, masks
    free(model)
    model = load_trainable(path)
    apply_prune(model, chosen_masks)
    model = finetune(cfg, model, ctx.tok, ctx.probe_texts, ctx.probe_labels, seed,
                     after_step=lambda: apply_prune(model, chosen_masks))
    r = {"cacc": ctx.cacc(model), "asr": ctx.asr(model)}
    free(model)
    return r, {"prune_ratio": chosen}


def bl_fine_mixing(ctx, path, seed, base):
    """Fine-mixing (Zhang et al., 2022): keep a random share rho of the fine-tuned weights, reset the rest to
    the pre-trained values, then fine-tune on clean data. rho is chosen on held-out clean probe sentences."""
    cfg = ctx.cfg
    half = len(ctx.probe_texts) // 2
    order = random.Random(seed).sample(range(len(ctx.probe_texts)), len(ctx.probe_texts))
    tr = [order[i] for i in range(half)]
    va = [order[i] for i in range(half, len(order))]
    tr_t, tr_l = [ctx.probe_texts[i] for i in tr], [ctx.probe_labels[i] for i in tr]
    va_t, va_l = [ctx.probe_texts[i] for i in va], [ctx.probe_labels[i] for i in va]
    ref = freeze(load_trainable(path))
    ref_acc = accuracy_on(ref, ctx.tok, va_t, va_l, cfg)
    free(ref)
    pre = dict(base.named_parameters())
    chosen = None
    for rho in cfg["fm_rhos"]:
        m = load_trainable(path)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for name, p in base_of(m).named_parameters():
                if name in pre and pre[name].shape == p.shape:
                    keep = (torch.rand(p.shape, generator=gen) < rho).to(DEVICE)
                    p.copy_(torch.where(keep, p, pre[name].to(p.dtype)))
        m = finetune(cfg, m, ctx.tok, tr_t, tr_l, seed)
        acc = accuracy_on(m, ctx.tok, va_t, va_l, cfg)
        last = (rho, m)
        if acc >= ref_acc - cfg["fm_max_drop"]:
            chosen = (rho, m)
            break
        free(m)
    rho, m = chosen if chosen else last
    r = {"cacc": ctx.cacc(m), "asr": ctx.asr(m)}
    free(m)
    return r, {"rho": rho}


class Onion:
    """ONION (Qi et al., 2021): remove each word whose deletion lowers the sentence's GPT-2 perplexity by more
    than a threshold, then classify the processed sentence."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg["onion_lm"])
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(cfg["onion_lm"]).to(DEVICE).eval()

    @torch.no_grad()
    def ppl(self, texts):
        out = []
        bs = self.cfg["onion_batch"]
        for i in range(0, len(texts), bs):
            chunk = [t if t.strip() else self.tok.eos_token for t in texts[i:i + bs]]
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
            logits = self.lm(**enc).logits[:, :-1]
            tgt = enc["input_ids"][:, 1:]
            m = enc["attention_mask"][:, 1:].float()
            nll = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
            denom = m.sum(1).clamp(min=1)
            out.extend(torch.exp((nll * m).sum(1) / denom).cpu().tolist())
        return out

    def clean(self, texts):
        variants, owner = [], []
        for i, t in enumerate(texts):
            w = t.split()
            variants.append(t)
            owner.append((i, -1))
            for j in range(len(w)):
                variants.append(" ".join(w[:j] + w[j + 1:]))
                owner.append((i, j))
        p = self.ppl(variants)
        full = {}
        drop = {i: [] for i in range(len(texts))}
        for (i, j), v in zip(owner, p):
            if j < 0:
                full[i] = v
        for (i, j), v in zip(owner, p):
            if j >= 0 and full[i] - v > self.cfg["onion_bar"]:
                drop[i].append(j)
        return [" ".join(w for j, w in enumerate(t.split()) if j not in set(drop[i])) for i, t in enumerate(texts)]


def bl_onion(ctx, model, onion):
    cfg = ctx.cfg
    ti = subset(len(ctx.test_texts), cfg["onion_cap"], cfg["subset_seed"])
    ai = subset(len(ctx.asr_texts), cfg["onion_cap"], cfg["subset_seed"])
    tt, tl = [ctx.test_texts[i] for i in ti], ctx.test_labels[ti]
    at = [ctx.asr_texts[i] for i in ai]
    enc = lambda x: encode(ctx.tok, x, cfg["max_length"])
    before_c = 100.0 * float(np.mean(predict(model, enc(tt), None, ctx.eb) == tl))
    before_a = 100.0 * float(np.mean(predict(model, enc(at), None, ctx.eb) == ctx.target))
    ct, ca = onion.clean(tt), onion.clean(at)
    after_c = 100.0 * float(np.mean(predict(model, enc(ct), None, ctx.eb) == tl))
    after_a = 100.0 * float(np.mean(predict(model, enc(ca), None, ctx.eb) == ctx.target))
    return {"cacc": after_c, "asr": after_a}, {"subset_cacc_before": before_c, "subset_asr_before": before_a,
                                              "subset_size": len(tt)}


LINEAR_PARTS = [("attention.self.query",), ("attention.self.key",), ("attention.self.value",),
                ("attention.output.dense",), ("intermediate.dense",), ("output.dense",)]


def get_linear(block, dotted):
    m = block
    for part in dotted.split("."):
        m = getattr(m, part)
    return m


def bl_orthopurify(ctx, path, seed, base):
    """OrthoPurify-style projection (adapted from arXiv 2610.09941 to a fully fine-tuned encoder): fine-tune the
    pre-trained model briefly on a few labeled clean sentences as a harmless reference; for every linear weight in
    the encoder, remove from the suspect model's weight update the top singular directions that lie more than
    a set angle away from the reference update's top directions."""
    cfg = ctx.cfg
    set_seed(seed)
    rng = random.Random(seed)
    pick = rng.sample(range(len(ctx.probe_texts)), min(cfg["op_samples"], len(ctx.probe_texts)))
    texts, labels = [ctx.probe_texts[i] for i in pick], [ctx.probe_labels[i] for i in pick]
    ref = AutoModelForSequenceClassification.from_pretrained(ctx.model_name, num_labels=ctx.num_labels).to(DEVICE)
    opt = torch.optim.AdamW(ref.parameters(), lr=cfg["op_lr"])
    ref.train()
    for step in range(cfg["op_steps"]):
        s = (step * cfg["op_batch"]) % len(texts)
        b = [(s + i) % len(texts) for i in range(cfg["op_batch"])]
        enc = ctx.tok([texts[i] for i in b], truncation=True, padding=True, max_length=cfg["max_length"],
                      return_tensors="pt").to(DEVICE)
        loss = ref(**enc, labels=torch.tensor([labels[i] for i in b], device=DEVICE)).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    freeze(ref)
    model = freeze(load_trainable(path))
    limit = math.cos(math.radians(cfg["op_angle_deg"]))
    k = cfg["op_k"]
    removed = 0
    blocks, pre_blocks, ref_blocks = base_of(model).encoder.layer, base.encoder.layer, base_of(ref).encoder.layer
    with torch.no_grad():
        for L in range(len(blocks)):
            for (dotted,) in LINEAR_PARTS:
                W = get_linear(blocks[L], dotted).weight
                W0 = get_linear(pre_blocks[L], dotted).weight
                Wr = get_linear(ref_blocks[L], dotted).weight
                dB = (W - W0).double().cpu().numpy()
                dR = (Wr - W0).double().cpu().numpy()
                Vb = np.linalg.svd(dB, full_matrices=False)[2][:k].T
                Vr = np.linalg.svd(dR, full_matrices=False)[2][:k].T
                Pm, cosines, _ = np.linalg.svd(Vb.T @ Vr)
                D = Vb @ Pm[:, cosines < limit]
                if D.shape[1] == 0:
                    continue
                removed += D.shape[1]
                Dt = to_model(D)
                W.copy_(W - (W - W0) @ Dt @ Dt.T)
    r = {"cacc": ctx.cacc(model), "asr": ctx.asr(model)}
    free(model, ref)
    return r, {"directions_removed": removed}


# =============================================================================================== jobs
def job_path(cfg, job):
    return os.path.join(cfg["results_dir"], job["dataset"], job["model"],
                        f"{job['attack']}_r{job['rate']}_s{job['seed']}.json")


def make_grid(datasets, models, seeds, rates=(0.1,), attacks=None, include_clean=True):
    jobs = []
    for ds in datasets:
        for m in models:
            for s in seeds:
                if include_clean:
                    jobs.append({"dataset": ds, "model": m, "attack": "clean", "rate": 0.0, "seed": s})
                for r in rates:
                    for a in (attacks or attacks_for(ds)):
                        if a in attacks_for(ds):
                            jobs.append({"dataset": ds, "model": m, "attack": a, "rate": r, "seed": s})
    return jobs


def run_job(cfg_overrides, job, force=False):
    cfg = {**DEFAULTS, **(cfg_overrides or {})}
    out_path = job_path(cfg, job)
    if os.path.exists(out_path) and not force:
        return "skipped"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t_start = time.time()
    timing = {}
    ctx = Ctx(cfg, job)
    seed, A = job["seed"], set(cfg["analyses"])
    tic = lambda: time.time()
    res = {"job": job, "version": VERSION, "env": {"torch": torch.__version__, "device": str(DEVICE),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "python": platform.python_version()}, "config": {k: v for k, v in cfg.items() if k not in ("models",)}}
    res["config"]["model_name"] = ctx.model_name

    tr_t, tr_l = ctx.clean["train"]
    clean_model, clean_path, _ = get_model(cfg, job, "clean", 0.0, tr_t, tr_l, ctx.tok)
    base = AutoModel.from_pretrained(ctx.model_name).to(DEVICE).eval()
    for p in base.parameters():
        p.requires_grad_(False)

    if job["attack"] == "clean":
        t = tic()
        res["baseline"] = {"cacc": ctx.cacc(clean_model)}
        prof = profile(ctx, clean_model)
        res["dormancy"] = [{"position": L, "dormancy": l["dormancy"]} for L, l in prof["layers"].items()]
        if "cand" in A:
            cands = changed_directions(clean_model, base, cfg["top_k"])
            table, before = candidate_tests(ctx, clean_model, prof, cands)
            res["cand"] = table.to_dict(orient="records")
            res["rules"] = []
            for rule, col in RULES.items():
                rep = build_repair(ctx, clean_model, prof, cands, table, col, before)
                cacc = ctx.cacc(clean_model, erase_fns(rep, prof))
                res["rules"].append({"rule": rule, "cacc": cacc, "cacc_drop": res["baseline"]["cacc"] - cacc,
                                     "directions": int(sum(D.shape[1] for D in rep.values()))})
        timing["clean_analysis"] = time.time() - t
        res["timing"] = timing
        _write(out_path, res)
        free(clean_model, base)
        return "done"

    # ---------------------------------------------------------------- backdoored model
    atk = build_attack(cfg, job["dataset"], job["attack"], job["rate"], seed, ctx.clean)
    ctx.set_attack(atk)
    model, path, _ = get_model(cfg, job, job["attack"], job["rate"], atk["train_texts"], atk["train_labels"], ctx.tok)
    base_cacc, base_asr = ctx.cacc(model), ctx.asr(model)
    clean_asr = ctx.asr(clean_model)
    asr_limit = max(cfg["ASR_AFTER_MAX"], clean_asr + cfg["ASR_MARGIN_OVER_CLEAN"])
    res["baseline"] = {"cacc": base_cacc, "asr": base_asr, "clean_model_cacc": ctx.cacc(clean_model),
                       "clean_model_asr": clean_asr, "asr_limit": asr_limit, "target": ctx.target,
                       "n_poison": atk["n_poison"], "n_train": len(atk["train_texts"]),
                       "aligned_train": atk.get("aligned_train"), "n_asr": len(ctx.asr_texts),
                       "n_test": len(ctx.test_labels), "recall_by_class": ctx.recall_by_class(model),
                       "clean_model_recall_by_class": ctx.recall_by_class(clean_model)}
    print(f"   CACC {base_cacc:.2f}  ASR {base_asr:.2f}  (clean model ASR {clean_asr:.2f})")

    t = tic()
    prof = profile(ctx, model)
    prof_c = profile(ctx, clean_model)
    timing["profile"] = time.time() - t
    P = n_positions(model)
    H = model.config.hidden_size
    rng = np.random.default_rng(seed)

    oracles, oracles_c = {}, {}
    if "oracle" in A:
        t = tic()
        inl, inl_c = inlp_oracle(ctx, model), inlp_oracle(ctx, clean_model)
        kmax = max(cfg["inlp_k"])
        oracles["inlp"] = {L: inl[L]["basis"][:, :min(4, inl[L]["basis"].shape[1])] for L in inl}
        oracles_c["inlp"] = {L: inl_c[L]["basis"][:, :min(4, inl_c[L]["basis"].shape[1])] for L in inl_c}
        res["inlp"] = [{"model_role": role, "position": L, "probe_acc": d[L]["probe_acc"]}
                       for role, d in (("backdoored", inl), ("clean", inl_c)) for L in d]
        if atk["insertion"]:
            to, to_c = token_oracle(ctx, model, prof), token_oracle(ctx, clean_model, prof_c)
            oracles["token"] = {L: to["layers"][L]["basis"] for L in range(P)}
            oracles_c["token"] = {L: to_c["layers"][L]["basis"] for L in range(P)}
            res["token_oracle"] = {"tokens": to["tokens"],
                                   "dims": [to["layers"][L]["basis"].shape[1] for L in range(P)],
                                   "shift_in_radii": [to["layers"][L]["shift"] / prof["layers"][L]["radius"] for L in range(P)]}
        timing["oracle"] = time.time() - t

        if "claim" in A:
            rows = []
            # INLP first direction, measured with all tokens of the triggered fit sentences
            rows += claim_rows("inlp", "backdoored", prof, {L: inl[L]["basis"][:, 0] for L in inl if inl[L]["basis"].shape[1]},
                               {L: inl[L]["trig_token_mean"] for L in inl})
            rows += claim_rows("inlp", "clean", prof_c, {L: inl_c[L]["basis"][:, 0] for L in inl_c if inl_c[L]["basis"].shape[1]},
                               {L: inl_c[L]["trig_token_mean"] for L in inl_c})
            if atk["insertion"]:
                tmean = {L: prof["layers"][L]["mean"] + to["layers"][L]["shift"] * to["layers"][L]["main"] for L in range(P)}
                tmean_c = {L: prof_c["layers"][L]["mean"] + to_c["layers"][L]["shift"] * to_c["layers"][L]["main"] for L in range(P)}
                rows += claim_rows("token", "backdoored", prof, {L: to["layers"][L]["main"] for L in range(P)}, tmean)
                rows += claim_rows("token", "clean", prof_c, {L: to_c["layers"][L]["main"] for L in range(P)}, tmean_c)
            res["claim"] = rows

    finders, finders_c = {}, {}
    if "finders" in A:
        t = tic()
        k = cfg["top_k"]
        finders["ddp"] = {L: l["V_top"] for L, l in prof["layers"].items()}
        finders_c["ddp"] = {L: l["V_top"] for L, l in prof_c["layers"].items()}
        finders["sensitivity"], finders_c["sensitivity"] = finder_sensitivity(prof), finder_sensitivity(prof_c)
        finders["weight_change"] = changed_directions(model, base, k)
        finders_c["weight_change"] = changed_directions(clean_model, base, k)
        if job["dataset"] in cfg["push_datasets"]:
            finders["push"] = finder_push(ctx, model, prof, seed)
            finders_c["push"] = finder_push(ctx, clean_model, prof_c, seed)
        timing["finders"] = time.time() - t
        if oracles:
            rows = []
            for fname in finders:
                for oname in oracles:
                    for L, V in finders[fname].items():
                        B, Bc = oracles[oname][L], oracles_c[oname].get(L)
                        if B.shape[1] == 0:
                            continue
                        cos = cos_to_subspace(V, B)
                        row = {"finder": fname, "oracle": oname, "position": L, "cos_best": float(cos.max()),
                               "best_rank": int(cos.argmax()) + 1, "cos_top1": float(cos[0]),
                               "cos_random": random_best_cos(H, V.shape[1], B, rng)}
                        if Bc is not None and Bc.shape[1] and L in finders_c[fname]:
                            row["cos_best_clean_control"] = float(cos_to_subspace(finders_c[fname][L], Bc).max())
                        rows.append(row)
            res["align"] = rows

    if "erase" in A:
        t = tic()
        rows = []
        for L in range(P):
            variants = []
            if "token" in oracles and oracles["token"][L].shape[1]:
                variants.append(("oracle_token", oracles["token"][L].shape[1], [oracles["token"][L]]))
            if oracles:
                full = inl[L]["basis"]
                for kk in cfg["inlp_k"]:
                    if full.shape[1] >= kk:
                        variants.append(("oracle_inlp", kk, [full[:, :kk]]))
            for fname, F_ in finders.items():
                if L in F_:
                    for kk in cfg["erase_ks"]:
                        variants.append((fname, kk, [F_[L][:, :kk]]))
            variants.append(("pca", cfg["top_k"], [prof["layers"][L]["pca"]]))
            variants.append(("random", cfg["top_k"], [rng.standard_normal((H, cfg["top_k"]))
                                                     for _ in range(cfg["random_draws"])]))
            for name, kk, sets in variants:
                cs, as_ = [], []
                for D in sets:
                    fn = {L: eraser(D, prof["layers"][L]["mean"])}
                    cs.append(ctx.cacc(model, fn))
                    as_.append(ctx.asr(model, fn))
                rows.append({"method": name, "k": kk, "position": L, "cacc": float(np.mean(cs)),
                             "asr": float(np.mean(as_)), "cacc_drop": base_cacc - float(np.mean(cs))})
        res["erase"] = rows
        timing["erase"] = time.time() - t

    if "cand" in A:
        t = tic()
        cands = finders.get("weight_change") or changed_directions(model, base, cfg["top_k"])
        table, before = candidate_tests(ctx, model, prof, cands)
        if cfg["hindsight"]:
            a_, c_ = [], []
            for _, r in table.iterrows():
                fn = erase_fns({int(r["position"]): cands[int(r["position"])][:, [int(r["rank"]) - 1]]}, prof)
                c_.append(ctx.cacc(model, fn))
                a_.append(ctx.asr(model, fn))
            table["asr_if_erased"], table["cacc_if_erased"] = a_, c_
            table["removes_backdoor"] = (table["asr_if_erased"] <= asr_limit) & \
                                        (base_cacc - table["cacc_if_erased"] <= cfg["CACC_DROP_MAX"])
        res["cand"] = table.to_dict(orient="records")
        res["rules"] = []
        for rule, col in RULES.items():
            rep = build_repair(ctx, model, prof, cands, table, col, before)
            fn = erase_fns(rep, prof)
            cacc, asr = ctx.cacc(model, fn), ctx.asr(model, fn)
            res["rules"].append({"rule": rule, "cacc": cacc, "asr": asr, "cacc_drop": base_cacc - cacc,
                                 "directions": int(sum(D.shape[1] for D in rep.values()))})
        timing["cand"] = time.time() - t

    if "baselines" in A:
        res["baselines"] = [{"method": "none", "cacc": base_cacc, "asr": base_asr}]
        onion = None
        for name in cfg["baselines"]:
            t = tic()
            try:
                if name == "clean_ft":
                    r, extra = bl_clean_ft(ctx, path, seed)
                elif name == "fine_pruning":
                    r, extra = bl_fine_pruning(ctx, path, seed)
                elif name == "fine_mixing":
                    r, extra = bl_fine_mixing(ctx, path, seed, base)
                elif name == "onion":
                    onion = onion or Onion(cfg)
                    r, extra = bl_onion(ctx, model, onion)
                elif name == "orthopurify":
                    r, extra = bl_orthopurify(ctx, path, seed, base)
                else:
                    continue
                res["baselines"].append({"method": name, **r, **extra, "cacc_drop": base_cacc - r["cacc"]})
                print(f"   {name:13s} CACC {r['cacc']:.2f}  ASR {r['asr']:.2f}")
            except Exception as e:                                          # one failing baseline must not lose the job
                res["baselines"].append({"method": name, "error": repr(e)})
                traceback.print_exc()
            timing[f"bl_{name}"] = time.time() - t
        if onion:
            free(onion.lm)

    timing["total"] = time.time() - t_start
    res["timing"] = timing
    _write(out_path, res)
    free(model, clean_model, base)
    return "done"


def _write(path, res):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(jsonable(res), f)
    os.replace(tmp, path)


def run_grid(cfg_overrides, jobs, log_path=None):
    t0 = time.time()
    for i, job in enumerate(jobs, 1):
        tag = f"[{i}/{len(jobs)}] {job['dataset']} {job['model']} {job['attack']} rate {job['rate']} seed {job['seed']}"
        print(f"\n{tag}", flush=True)
        try:
            status = run_job(cfg_overrides, job)
            print(f"   {status}   ({(time.time() - t0) / 3600:.2f} h elapsed)", flush=True)
        except Exception as e:
            print(f"   FAILED: {e!r}", flush=True)
            traceback.print_exc()
            if log_path:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"job": job, "error": repr(e), "time": time.ctime()}) + "\n")
        free()


# =============================================================================================== loading results
def load_results(results_dir):
    out = []
    for root, _, files in os.walk(results_dir):
        for f in files:
            if f.endswith(".json"):
                with open(os.path.join(root, f)) as fh:
                    out.append(json.load(fh))
    return out


def frame(results, key):
    rows = []
    for r in results:
        for row in r.get(key, []) or []:
            rows.append({**r["job"], **row})
    return pd.DataFrame(rows)
