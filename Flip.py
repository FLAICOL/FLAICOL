#!/usr/bin/env python3
"""
 bias-homotopy + embedding flip finder 
- Edit the USER SETTINGS below to point to your model and data.
- Requirements: transformers, torch, numpy, scipy, pandas, tqdm
"""

import os
import time
import gc
import logging
from typing import Optional, Sequence, List
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.optimize import minimize

# -------------------- USER SETTINGS (edit) --------------------
# Either modify these directly or set the corresponding environment variables:
#   MODEL_PATH, DATA_CSV, OUTPUT_DIR, ONLY_LABEL (optional), ONLY_LABEL_BY ('true'/'pred')
MODEL_PATH = os.environ.get("MODEL_PATH", "/path/to/fine_tuned_model")   # required: path or HF id
DATA_CSV   = os.environ.get("DATA_CSV", "data.csv")                      # required: CSV with 'text' and 'label'
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Runtime / memory knobs
MAX_SEQ_LEN = 64
MARGIN = 0.02
SAMPLE_LIMIT = 100          # None -> full dataset
PROCESS_ONLY_MISCLASSIFIED = False
TARGET_MODE = "best_other"  # "best_other", "all", or an int label
VERBOSE = False
RNG_SEED = 42

# Single-class override: set to None to use balancing logic, or set to integer label id to process only that class.
ONLY_LABEL = None          # e.g. 0 to process only label 0
ONLY_LABEL_BY = 'true'     # 'true' (dataset true label) or 'pred' (select by model prediction)

# PGD / homotopy params
PGD_K = 8
PGD_STEPS_PER_ALPHA = 100
PGD_ALPHAS = (1.0, 10.0, 100.0)
PGD_LR = 1e-2
PGD_TOL_EQ = 1e-3
PGD_TOL_INEQ = 1e-6
PGD_MAX_TOTAL_STEPS = 400
HOMO_NITER = 6
SAVE_EVERY = 10

# Balancing options
BALANCE = True
BALANCE_BY = 'true'         # 'true' (use df.label_id) or 'pred' (use model predictions)
BALANCE_MODE = 'binary_half'  # 'binary_half' or 'per_class'
BALANCED_TOTAL = None
PER_CLASS = None

PRED_BATCH_SIZE = 64

# Device (auto by default)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------

# Logging
logging.basicConfig(level=logging.INFO if not VERBOSE else logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("flip_anonymous")
logger.info("Device: %s", device)

# -------------------- Helpers --------------------
def load_model_and_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()
    return model, tokenizer

def build_inputs_embeds_for_ids(model, input_ids: torch.LongTensor, word_embeds: torch.FloatTensor = None):
    base = getattr(model, model.base_model_prefix, None)
    if base is None:
        if hasattr(model, 'roberta'):
            base = model.roberta
        elif hasattr(model, 'bert'):
            base = model.bert
        else:
            base = None

    if word_embeds is None:
        if base is not None:
            word_embeds = base.embeddings.word_embeddings(input_ids)
        else:
            word_embeds = model.get_input_embeddings()(input_ids)

    seq_len = input_ids.shape[1]
    device_local = input_ids.device
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device_local)[None, :]
    if base is not None:
        pos_embeds = base.embeddings.position_embeddings(position_ids)
        if hasattr(base.embeddings, 'token_type_embeddings'):
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device_local)
            tok_type = base.embeddings.token_type_embeddings(token_type_ids)
        else:
            tok_type = torch.zeros_like(pos_embeds)
    else:
        pos_embeds = torch.zeros_like(word_embeds)
        tok_type = torch.zeros_like(word_embeds)

    inputs_embeds = word_embeds + pos_embeds + tok_type
    return inputs_embeds

# -------------------- PGD routine --------------------
def find_flip_pgd_embedding_space_with_init(
    model, tokenizer, input_text: str, this_label: int, that_label: int,
    MAX_SEQ_LEN: int,
    K: Optional[int] = PGD_K,
    steps_per_alpha: int = PGD_STEPS_PER_ALPHA,
    alphas: Sequence[float] = PGD_ALPHAS,
    lr: float = PGD_LR,
    margin: float = MARGIN,
    tol_eq: float = PGD_TOL_EQ,
    tol_ineq: float = PGD_TOL_INEQ,
    max_total_steps: int = PGD_MAX_TOTAL_STEPS,
    device_local: Optional[torch.device] = None,
    verbose: bool = False,
    init_param: Optional[np.ndarray] = None,
):
    if device_local is None:
        device_local = device
    model.to(device_local)
    model.eval()

    toks = tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=MAX_SEQ_LEN)
    input_ids_t = toks['input_ids'].to(device_local)
    attention_mask_t = toks['attention_mask'].to(device_local)
    seq_len = input_ids_t.size(1)
    hidden_size = model.config.hidden_size
    nz = model.config.num_labels

    base = getattr(model, model.base_model_prefix, None)
    if base is None:
        if hasattr(model, 'roberta'):
            base = model.roberta
        elif hasattr(model, 'bert'):
            base = model.bert
        else:
            base = None

    with torch.no_grad():
        if base is not None:
            word_embeds_orig = base.embeddings.word_embeddings(input_ids_t)
        else:
            word_embeds_orig = model.get_input_embeddings()(input_ids_t)
    x_hat = word_embeds_orig.detach()

    positions = list(range(seq_len))
    if K is not None and K < seq_len:
        x_hat_var = x_hat.clone().detach().requires_grad_(True)
        logits_tmp = model(inputs_embeds=x_hat_var, attention_mask=attention_mask_t).logits[0]
        diff_tmp = logits_tmp[that_label] - logits_tmp[this_label]
        model.zero_grad()
        if x_hat_var.grad is not None:
            x_hat_var.grad.detach_()
            x_hat_var.grad.zero_()
        diff_tmp.backward(retain_graph=False)
        grads = x_hat_var.grad.detach().cpu().numpy()[0]
        token_importance = np.linalg.norm(grads, axis=1)
        topk_idx = np.argsort(token_importance)[::-1][:K]
        positions = sorted(int(i) for i in topk_idx)

    P = len(positions)
    param_delta = torch.zeros((1, P, hidden_size), dtype=torch.float32, device=device_local, requires_grad=True)
    if init_param is not None:
        arr = np.asarray(init_param)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.shape[1] == P:
            with torch.no_grad():
                param_delta.copy_(torch.tensor(arr, dtype=torch.float32, device=device_local))

    optimizer = torch.optim.Adam([param_delta], lr=lr)
    total_steps = 0
    best_result = None

    def build_inputs_embeds_from_param(delta_param):
        emb = x_hat.clone()
        if P > 0:
            emb[0, positions, :] = emb[0, positions, :] + delta_param[0]
        return emb

    for alpha in alphas:
        if verbose:
            logger.debug("Starting alpha=%g", alpha)
        for _ in range(steps_per_alpha):
            total_steps += 1
            if total_steps > max_total_steps:
                break
            optimizer.zero_grad()
            inputs_embeds = build_inputs_embeds_from_param(param_delta)
            logits = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask_t).logits[0]
            diff_vec = (inputs_embeds - x_hat).view(-1)
            dist2 = torch.sum(diff_vec * diff_vec)
            z = logits
            eq_diff = z[this_label] - z[that_label]
            ineq_terms = []
            for idx in range(nz):
                if idx in (this_label, that_label):
                    continue
                viol = z[idx] - z[this_label] + margin
                ineq_terms.append(torch.clamp(viol, min=0.0) ** 2)
            ineq_sum = torch.stack(ineq_terms).sum() if len(ineq_terms) > 0 else torch.tensor(0.0, device=device_local)
            loss = dist2 + alpha * (eq_diff ** 2) + alpha * ineq_sum
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                param_delta.clamp_(-10.0, 10.0)
            with torch.no_grad():
                final_emb = build_inputs_embeds_from_param(param_delta)
                logits_eval = model(inputs_embeds=final_emb, attention_mask=attention_mask_t).logits[0]
                z_np = logits_eval.detach().cpu().numpy()
                zs = np.exp(z_np - np.max(z_np)); zs /= zs.sum()
                eq_val = float((logits_eval[this_label] - logits_eval[that_label]).item())
                ineq_max = max(((logits_eval[idx] - logits_eval[this_label] + margin).item() for idx in range(nz) if idx not in (this_label, that_label)), default=-np.inf)
                dist = float(torch.norm((final_emb - x_hat).view(-1)).item())
            if abs(eq_val) <= tol_eq and ineq_max <= tol_ineq:
                xf_flat = final_emb.detach().cpu().numpy().reshape(-1)
                info = {'res': None, 'c': float(min(zs[this_label], zs[that_label])), 'dist': dist,
                        'z_final': z_np, 'zs_final': zs, 'positions': positions, 'steps': total_steps, 'note': 'success'}
                try:
                    del param_delta, final_emb, logits_eval
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return xf_flat, info
            if best_result is None or dist < best_result['dist']:
                best_result = {'dist': dist, 'z': z_np.copy(), 'params': param_delta.detach().cpu().numpy().copy(),
                               'positions': positions.copy(), 'steps': total_steps}
        if total_steps > max_total_steps:
            break

    if best_result is not None:
        params_best = torch.tensor(best_result['params'], device=device_local)
        final_emb = build_inputs_embeds_from_param(params_best)
        xf_flat = final_emb.detach().cpu().numpy().reshape(-1)
        z_final = best_result['z']
        zs_final = np.exp(z_final - np.max(z_final)); zs_final /= zs_final.sum()
        info = {'res': None, 'c': float('nan'), 'dist': float(best_result['dist']),
                'z_final': z_final, 'zs_final': zs_final, 'positions': best_result['positions'],
                'steps': best_result['steps'], 'note': 'approx_best'}
    else:
        xf_flat = x_hat.detach().cpu().numpy().reshape(-1)
        info = {'res': None, 'c': float('nan'), 'dist': 0.0, 'z_final': None, 'positions': positions, 'steps': total_steps, 'note': 'no_improvement'}

    try:
        del param_delta, final_emb
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return xf_flat, info

# -------------------- Homotopy helpers --------------------
def get_encoder_representation(model, tokenizer, input_text, MAX_SEQ_LEN, device_local=None):
    if device_local is None:
        device_local = device
    toks = tokenizer(input_text, return_tensors='pt', truncation=True,
                     padding='max_length', max_length=MAX_SEQ_LEN)
    input_ids_t = toks['input_ids'].to(device_local)
    attention_mask_t = toks['attention_mask'].to(device_local)

    base = getattr(model, model.base_model_prefix, None)
    if base is None:
        if hasattr(model, 'roberta'):
            base = model.roberta
        elif hasattr(model, 'bert'):
            base = model.bert
        else:
            base = None

    with torch.no_grad():
        if base is not None:
            outputs = base(input_ids=input_ids_t, attention_mask=attention_mask_t, return_dict=True)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                h = outputs.pooler_output[0].detach().cpu().numpy()
                return h
            if hasattr(outputs, 'last_hidden_state'):
                h = outputs.last_hidden_state[:, 0, :][0].detach().cpu().numpy()
                return h
            if hasattr(outputs, 'last_hidden_state'):
                h = outputs.last_hidden_state.mean(dim=1)[0].detach().cpu().numpy()
                return h
        full = model(**toks.to(device_local), output_hidden_states=True, return_dict=True)
        if hasattr(full, 'hidden_states') and full.hidden_states is not None:
            hs = full.hidden_states[-1][:,0,:][0].detach().cpu().numpy()
            return hs
    raise RuntimeError("Could not extract encoder representation h(x) for this model architecture.")

def compute_b_homo(yw: np.ndarray, b_orig: np.ndarray, this: int, that: int, margin_scale: float = 1.1, verbose=False):
    yw = np.asarray(yw).ravel().astype(float)
    b = np.asarray(b_orig).ravel().astype(float)
    nb = b.size
    def fun_obj(bh):
        diff = b - bh
        f = float(np.sum(diff**2))
        g = -2.0 * diff
        return f, g
    def f_wrap(bh): return fun_obj(bh)[0]
    def grad_wrap(bh): return fun_obj(bh)[1]
    def eq_fun(bh):
        bh = np.asarray(bh).ravel()
        return float((yw + bh)[this] - (yw + bh)[that])
    def eq_jac(bh):
        g = np.zeros(nb); g[this] = 1.0; g[that] = -1.0; return g
    otherList = [i for i in range(nb) if i not in (this, that)]
    def ineq_fun_scipy(bh):
        if len(otherList) == 0: return np.array([])
        bh = np.asarray(bh).ravel()
        vals = margin_scale * (yw[otherList] + bh[otherList]) - (yw[this] + bh[this])
        return -vals
    def ineq_jac_scipy(bh):
        if len(otherList) == 0: return np.zeros((0, nb))
        J = np.zeros((len(otherList), nb))
        for r, idx in enumerate(otherList):
            J[r, idx] = margin_scale
            J[r, this] = -1.0
        return -J
    cons = [{'type':'eq','fun':eq_fun,'jac':eq_jac}]
    if len(otherList) > 0:
        cons.append({'type':'ineq','fun':ineq_fun_scipy,'jac':ineq_jac_scipy})
    def hess_fun(x, v=None):
        H = 2.0 * np.eye(nb)
        if v is None:
            return H
        else:
            return H.dot(v)
    x0 = b.copy()
    opts = {'maxiter': 500}
    res = minimize(f_wrap, x0, method='trust-constr', jac=grad_wrap, constraints=cons, hess=hess_fun, options=opts)
    if verbose:
        logger.debug("compute_b_homo: success %s message: %s", res.success, res.message)
    return res.x

def bias_homotopy_wrapper(
    model, tokenizer, input_text: str, this_label: int, that_label: int,
    MAX_SEQ_LEN: int,
    niter: int = HOMO_NITER,
    pgd_kwargs: dict = None,
    verbose: bool = False
):
    if pgd_kwargs is None:
        pgd_kwargs = {}
    toks = tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=MAX_SEQ_LEN)
    input_ids_t = toks['input_ids'].to(device)
    attention_mask_t = toks['attention_mask'].to(device)
    h_x = get_encoder_representation(model, tokenizer, input_text, MAX_SEQ_LEN, device_local=device)
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'weight'):
        W_out = model.classifier.weight.detach().cpu().numpy()
        b_orig = model.classifier.bias.detach().cpu().numpy()
    elif hasattr(model, 'score') and hasattr(model.score, 'weight'):
        W_out = model.score.weight.detach().cpu().numpy()
        b_orig = model.score.bias.detach().cpu().numpy()
    else:
        found = False
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and module.out_features == model.config.num_labels:
                W_out = module.weight.detach().cpu().numpy()
                b_orig = module.bias.detach().cpu().numpy()
                found = True
                break
        if not found:
            raise RuntimeError("Could not locate classifier linear layer to extract weight & bias.")
    yw = h_x.dot(W_out.T)
    b_homo = compute_b_homo(yw, b_orig, this_label, that_label, margin_scale=1.1, verbose=verbose)
    bdif = b_homo - b_orig
    prev_params = None
    prev_positions = None
    xf_last = None
    info_last = None
    b_saved = b_orig.copy()
    try:
        for i in range(0, niter + 1):
            tfrac = (1.0 - float(i) / float(niter))
            b_t = b_orig + bdif * (tfrac ** 1.0)
            if hasattr(model, 'classifier') and hasattr(model.classifier, 'bias'):
                model.classifier.bias.data = torch.tensor(b_t, dtype=torch.float32, device=model.classifier.bias.device)
            if verbose:
                logger.debug("[bias_homotopy] step %d/%d, bias frac %g", i, niter, tfrac)
            pgd_call_kwargs = dict(pgd_kwargs)
            if prev_params is not None and prev_positions is not None:
                pgd_call_kwargs['init_param'] = prev_params
                pgd_call_kwargs['K'] = len(prev_positions)
            xf_flat, info = find_flip_pgd_embedding_space_with_init(
                model, tokenizer, input_text, this_label, that_label,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                device_local=device,
                verbose=verbose,
                **pgd_call_kwargs
            )
            positions = info.get('positions', None)
            if positions is None:
                prev_params = None
                prev_positions = None
            else:
                hidden_size = model.config.hidden_size
                seq_len = MAX_SEQ_LEN
                final_emb = xf_flat.reshape(1, seq_len, hidden_size)
                with torch.no_grad():
                    base = getattr(model, model.base_model_prefix, None)
                    if base is not None:
                        word_embeds_orig = base.embeddings.word_embeddings(input_ids_t)
                    else:
                        word_embeds_orig = model.get_input_embeddings()(input_ids_t)
                x_hat_np = word_embeds_orig.detach().cpu().numpy().reshape(1, seq_len, hidden_size)
                pos_arr = np.array(positions, dtype=int)
                delta = final_emb[0, pos_arr, :] - x_hat_np[0, pos_arr, :]
                prev_params = delta[None, :, :]
                prev_positions = positions
            xf_last = xf_flat
            info_last = info
    finally:
        try:
            if hasattr(model, 'classifier') and hasattr(model.classifier, 'bias'):
                model.classifier.bias.data = torch.tensor(b_saved, dtype=torch.float32, device=model.classifier.bias.device)
        except Exception:
            pass
    return xf_last, info_last

# -------------------- Utility: batched predictions --------------------
def compute_model_predictions(model, tokenizer, texts: List[str], max_seq_len: int = 128, batch_size: int = 64, device_local: Optional[torch.device] = None):
    if device_local is None:
        device_local = device
    model.to(device_local)
    preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        toks = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=max_seq_len)
        input_ids_t = toks['input_ids'].to(device_local)
        attention_mask_t = toks['attention_mask'].to(device_local)
        with torch.no_grad():
            out = model(input_ids=input_ids_t, attention_mask=attention_mask_t)
            batch_preds = torch.argmax(out.logits, dim=-1).detach().cpu().numpy().tolist()
        preds.extend(batch_preds)
        del toks, input_ids_t, attention_mask_t, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.array(preds, dtype=int)

# -------------------- Main runner --------------------
def main():
    rng = np.random.default_rng(RNG_SEED)
    logger.info("Loading model & tokenizer from %s", MODEL_PATH)
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

    df = pd.read_csv(DATA_CSV)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")

    # Map dataset labels -> numeric ids (try model.config.label2id if present)
    label2id_cfg = getattr(model.config, "label2id", None)
    if label2id_cfg and df['label'].dtype == object:
        df['label_id'] = df['label'].map(lambda x: label2id_cfg.get(str(x), np.nan))
        if df['label_id'].isna().any():
            logger.warning("Some labels couldn't be mapped via model.config.label2id; filling -1")
            df['label_id'] = df['label_id'].fillna(-1).astype(int)
    else:
        df['label_id'] = df['label'].astype(int)

    # ------------------ Selection: ONLY_LABEL override OR balancing ------------------
    proc_list = None

    if ONLY_LABEL is not None:
        if ONLY_LABEL_BY not in ('true', 'pred'):
            raise ValueError("ONLY_LABEL_BY must be 'true' or 'pred'")

        if ONLY_LABEL_BY == 'true':
            idx_only = df.index[df['label_id'] == ONLY_LABEL].to_numpy()
        else:
            logger.info("Computing model predictions to select rows where model predicts ONLY_LABEL...")
            preds_all = compute_model_predictions(model, tokenizer, df['text'].astype(str).tolist(),
                                                  max_seq_len=MAX_SEQ_LEN, batch_size=PRED_BATCH_SIZE)
            idx_only = np.where(preds_all == ONLY_LABEL)[0]

        if len(idx_only) == 0:
            raise ValueError(f"No rows with label {ONLY_LABEL} found using ONLY_LABEL_BY='{ONLY_LABEL_BY}'.")

        if SAMPLE_LIMIT is not None:
            idx_only = idx_only[:SAMPLE_LIMIT]
        proc_list = idx_only.tolist()
        logger.info("ONLY_LABEL set -> processing only label %s (by=%s): %d rows", ONLY_LABEL, ONLY_LABEL_BY, len(proc_list))
    else:
        # balancing logic
        if BALANCE:
            if BALANCE_MODE == 'binary_half':
                total_wanted = BALANCED_TOTAL if BALANCED_TOTAL is not None else (SAMPLE_LIMIT if SAMPLE_LIMIT is not None else min(1000, len(df)))
                if total_wanted is None:
                    total_wanted = min(1000, len(df))
                if total_wanted % 2 == 1:
                    total_wanted -= 1
                per_class = total_wanted // 2

                if BALANCE_BY == 'true':
                    idx0 = df.index[df['label_id'] == 0].to_numpy()
                    idx1 = df.index[df['label_id'] == 1].to_numpy()
                elif BALANCE_BY == 'pred':
                    logger.info("Computing model predictions for BALANCE_BY='pred' (batched)...")
                    preds = compute_model_predictions(model, tokenizer, df['text'].astype(str).tolist(), max_seq_len=MAX_SEQ_LEN, batch_size=PRED_BATCH_SIZE)
                    idx0 = np.where(preds == 0)[0]
                    idx1 = np.where(preds == 1)[0]
                else:
                    raise ValueError("BALANCE_BY must be 'true' or 'pred'")

                if len(idx0) == 0 or len(idx1) == 0:
                    raise ValueError("Cannot binary-balance: dataset or predictions lack label 0 or 1.")

                per_class = min(per_class, len(idx0), len(idx1))
                sel0 = rng.choice(idx0, per_class, replace=False)
                sel1 = rng.choice(idx1, per_class, replace=False)
                combined = np.concatenate([sel0, sel1])
                rng.shuffle(combined)
                proc_list = combined.tolist()
                logger.info("Balanced binary selection prepared: %d rows (%d per class).", len(proc_list), per_class)
            elif BALANCE_MODE == 'per_class':
                if PER_CLASS is not None:
                    per_class = int(PER_CLASS)
                else:
                    total_wanted = BALANCED_TOTAL if BALANCED_TOTAL is not None else (SAMPLE_LIMIT if SAMPLE_LIMIT is not None else 1000)
                    num_labels = len(df['label_id'].unique())
                    per_class = max(1, total_wanted // num_labels)
                if BALANCE_BY == 'true':
                    labels = sorted(df['label_id'].unique())
                    selected = []
                    for lab in labels:
                        idx_lab = df.index[df['label_id'] == lab].to_numpy()
                        if len(idx_lab) == 0:
                            continue
                        take = min(per_class, len(idx_lab))
                        sel = rng.choice(idx_lab, take, replace=False)
                        selected.append(sel)
                    if len(selected) == 0:
                        raise ValueError("No labels found for per_class sampling.")
                    proc_list = np.concatenate(selected)
                    rng.shuffle(proc_list)
                    proc_list = proc_list.tolist()
                    logger.info("Balanced per-class selection prepared: %d rows (%d per class).", len(proc_list), per_class)
                elif BALANCE_BY == 'pred':
                    logger.info("Computing model predictions for BALANCE_BY='pred' (batched)...")
                    preds = compute_model_predictions(model, tokenizer, df['text'].astype(str).tolist(), max_seq_len=MAX_SEQ_LEN, batch_size=PRED_BATCH_SIZE)
                    labels = sorted(np.unique(preds))
                    selected = []
                    for lab in labels:
                        idx_lab = np.where(preds == lab)[0]
                        if len(idx_lab) == 0:
                            continue
                        take = min(per_class, len(idx_lab))
                        sel = rng.choice(idx_lab, take, replace=False)
                        selected.append(sel)
                    if len(selected) == 0:
                        raise ValueError("No predicted labels found for per_class sampling.")
                    proc_list = np.concatenate(selected)
                    rng.shuffle(proc_list)
                    proc_list = proc_list.tolist()
                    logger.info("Balanced per-class by prediction prepared: %d rows (%d per class).", len(proc_list), per_class)
                else:
                    raise ValueError("BALANCE_BY must be 'true' or 'pred'")

    # default proc_list if none selected
    if proc_list is None:
        n_all = len(df)
        if SAMPLE_LIMIT is None:
            num_rows = n_all
        else:
            num_rows = min(SAMPLE_LIMIT, n_all)
        proc_list = list(range(num_rows))
    else:
        if SAMPLE_LIMIT is not None and len(proc_list) > SAMPLE_LIMIT:
            proc_list = proc_list[:SAMPLE_LIMIT]

    logger.info("Processing %d rows (proc_list length)", len(proc_list))

    results = []
    pbar = tqdm(proc_list, desc="rows")
    for idx in pbar:
        row = df.loc[idx]
        text = str(row['text'])
        true_label = int(row['label_id'])

        toks = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=MAX_SEQ_LEN)
        input_ids_t = toks['input_ids'].to(device)
        attention_mask_t = toks['attention_mask'].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids_t, attention_mask=attention_mask_t)
            pred = int(torch.argmax(out.logits, dim=-1).item())

        if PROCESS_ONLY_MISCLASSIFIED and pred == true_label:
            pbar.set_description(f"skip row {idx} (correct)")
            continue

        this_label = pred
        if TARGET_MODE == "best_other":
            logits_arr = out.logits[0].detach().cpu().numpy()
            sorted_idx = np.argsort(logits_arr)[::-1]
            that_label = next((i for i in sorted_idx if i != this_label), None)
            if that_label is None:
                continue
        elif TARGET_MODE == "all":
            other_labels = [j for j in range(model.config.num_labels) if j != this_label]
            that_label = other_labels[0]
        else:
            if isinstance(TARGET_MODE, int):
                that_label = int(TARGET_MODE)
            else:
                that_label = (this_label + 1) % model.config.num_labels

        pbar.set_description(f"row {idx} this->{this_label} to {that_label}")

        try:
            xf_flat, info = bias_homotopy_wrapper(
                model, tokenizer, text, this_label, that_label,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                niter=HOMO_NITER,
                pgd_kwargs={
                    'K': PGD_K, 'steps_per_alpha': PGD_STEPS_PER_ALPHA, 'alphas': PGD_ALPHAS,
                    'lr': PGD_LR, 'margin': MARGIN, 'tol_eq': PGD_TOL_EQ, 'tol_ineq': PGD_TOL_INEQ,
                    'max_total_steps': PGD_MAX_TOTAL_STEPS
                },
                verbose=VERBOSE
            )

            tokens_nearest = None
            try:
                base = getattr(model, model.base_model_prefix, None)
                if base is None:
                    if hasattr(model, 'roberta'):
                        base = model.roberta
                    elif hasattr(model, 'bert'):
                        base = model.bert
                    else:
                        base = None
                if base is not None:
                    emb_matrix = base.embeddings.word_embeddings.weight.detach().cpu().numpy()
                    seq_len = MAX_SEQ_LEN
                    hidden_size = model.config.hidden_size
                    xf_words = xf_flat.reshape(1, seq_len, hidden_size)[0]
                    nearest_ids = []
                    for p in range(seq_len):
                        v = xf_words[p]
                        dists = np.sum((emb_matrix - v[None, :])**2, axis=1)
                        idxn = int(np.argmin(dists))
                        nearest_ids.append(idxn)
                    tokens_nearest = tokenizer.convert_ids_to_tokens(nearest_ids)
            except Exception:
                tokens_nearest = None

            out_record = {
                'row': int(idx),
                'text': text,
                'true_label': int(true_label),
                'pred_label': int(pred),
                'this_label': int(this_label),
                'that_label': int(that_label),
                'dist': info.get('dist', np.nan),
                'c': info.get('c', np.nan),
                'time': info.get('steps', np.nan),
                'zs_final': info.get('zs_final').tolist() if info.get('zs_final') is not None else None,
                'tokens_nearest': " ".join(tokens_nearest) if tokens_nearest else None,
                'res_success': (info.get('note') == 'success'),
                'res_message': info.get('note')
            }
        except Exception as e:
            out_record = {
                'row': int(idx),
                'text': text,
                'true_label': int(true_label),
                'pred_label': int(pred),
                'this_label': int(this_label),
                'that_label': int(that_label),
                'dist': np.nan,
                'c': np.nan,
                'time': np.nan,
                'zs_final': None,
                'tokens_nearest': None,
                'res_success': False,
                'res_message': f"error: {repr(e)}"
            }

        results.append(out_record)

        # cleanup and free memory
        try:
            del xf_flat, info, toks, out
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if len(results) % SAVE_EVERY == 0:
            pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "results_partial.csv"), index=False)

    out_df = pd.DataFrame(results)
    out_fp = os.path.join(OUTPUT_DIR, "results_noargparse.csv")
    out_df.to_csv(out_fp, index=False)
    logger.info("Saved results to %s", out_fp)

if __name__ == "__main__":
    main()

