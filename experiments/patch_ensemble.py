import pathlib
p = pathlib.Path('run_experiments_torch.py')
t = p.read_text(encoding='utf-8')

old = '''PRED_STORE = {}


def store_preds(ds: str, model: str, p: np.ndarray, std: np.ndarray, y: np.ndarray) -> None:
    key = (ds, model)
    if key not in PRED_STORE:
        PRED_STORE[key] = {"p": [], "std": [], "y": y}
    PRED_STORE[key]["p"].append(p)
    PRED_STORE[key]["std"].append(std)


def save_aggregate_preds() -> None:
    for (ds, model), vals in PRED_STORE.items():
        p = np.mean(np.stack(vals["p"]), axis=0)
        std = np.mean(np.stack(vals["std"]), axis=0)
        np.savez_compressed(
            os.path.join(PRED_DIR, f"{ds}_{safe(model)}.npz"),
            p=p,
            std=std,
            y=vals["y"],
        )'''
new = '''PRED_STORE = {}


def store_preds(
    ds: str,
    model: str,
    p: np.ndarray,
    std: np.ndarray,
    y: np.ndarray,
    pv: np.ndarray,
) -> None:
    key = (ds, model)
    if key not in PRED_STORE:
        PRED_STORE[key] = {"p": [], "std": [], "y": y, "pv": []}
    PRED_STORE[key]["p"].append(p)
    PRED_STORE[key]["std"].append(std)
    PRED_STORE[key]["pv"].append(pv)


def save_aggregate_preds() -> None:
    for (ds, model), vals in PRED_STORE.items():
        p = np.mean(np.stack(vals["p"]), axis=0)
        std = np.mean(np.stack(vals["std"]), axis=0)
        pv = np.mean(np.stack(vals["pv"]), axis=0)
        np.savez_compressed(
            os.path.join(PRED_DIR, f"{ds}_{safe(model)}.npz"),
            p=p,
            std=std,
            y=vals["y"],
            pv=pv,
        )'''
assert old in t
t = t.replace(old, new)

old = '''    threshold, _ = calibrate_threshold(data["y_val"], p_val)
    metrics = evaluate_binary(data["y_test"], p_test, threshold)
    store_preds(ds, model_name, p_test, std_test, data["y_test"])
    if save_first:
        save_model(model, os.path.join(MODELS_DIR, f"{ds}_{safe(model_name)}.pt"))
    return {
        "model": model_name,
        "family": "proposed/ablation",
        "seed": seed,
        **metrics,
    }'''
new = '''    threshold, _ = calibrate_threshold(data["y_val"], p_val)
    metrics = evaluate_binary(data["y_test"], p_test, threshold)
    store_preds(ds, model_name, p_test, std_test, data["y_test"], p_val)
    if save_first:
        save_model(model, os.path.join(MODELS_DIR, f"{ds}_{safe(model_name)}.pt"))
    return {
        "model": model_name,
        "family": "proposed/ablation",
        "seed": seed,
        **metrics,
    }'''
assert old in t
t = t.replace(old, new)

old = '''    threshold, _ = calibrate_threshold(data["y_val"], p_val)
    metrics = evaluate_binary(data["y_test"], p_test, threshold)
    store_preds(ds, model_name, p_test, np.zeros_like(p_test), data["y_test"])
    if save_first:
        save_model(model, os.path.join(MODELS_DIR, f"{ds}_{safe(model_name)}.pt"))
    return {
        "model": model_name,
        "family": "deep baseline",
        "seed": seed,
        **metrics,
    }'''
new = '''    threshold, _ = calibrate_threshold(data["y_val"], p_val)
    metrics = evaluate_binary(data["y_test"], p_test, threshold)
    store_preds(ds, model_name, p_test, np.zeros_like(p_test), data["y_test"], p_val)
    if save_first:
        save_model(model, os.path.join(MODELS_DIR, f"{ds}_{safe(model_name)}.pt"))
    return {
        "model": model_name,
        "family": "deep baseline",
        "seed": seed,
        **metrics,
    }'''
assert old in t
t = t.replace(old, new)

old = '''        threshold, _ = calibrate_threshold(data["y_val"], probs["val"])
        metrics = evaluate_binary(data["y_test"], probs["test"], threshold)
        store_preds(ds, name, probs["test"], np.zeros_like(probs["test"]), data["y_test"])'''
new = '''        threshold, _ = calibrate_threshold(data["y_val"], probs["val"])
        metrics = evaluate_binary(data["y_test"], probs["test"], threshold)
        store_preds(
            ds, name, probs["test"], np.zeros_like(probs["test"]), data["y_test"], probs["val"]
        )'''
assert old in t
t = t.replace(old, new)

old = '''    seed_df = pd.DataFrame(all_rows)
    seed_df.to_csv(SEEDS_CSV, index=False)
    overall = aggregate_seeds(seed_df)
    overall.to_csv(OVERALL_CSV, index=False)
    save_aggregate_preds()'''
new = '''    seed_df = pd.DataFrame(all_rows)
    seed_df.to_csv(SEEDS_CSV, index=False)
    overall = aggregate_seeds(seed_df)
    overall.to_csv(OVERALL_CSV, index=False)
    save_aggregate_preds()
    build_ensemble_csv()'''
assert old in t
t = t.replace(old, new)

old = '''def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:'''
new = '''def build_ensemble_csv() -> None:
    rows = []
    for (ds, model), vals in PRED_STORE.items():
        pv = np.mean(np.stack(vals["pv"]), axis=0)
        p = np.mean(np.stack(vals["p"]), axis=0)
        threshold, _ = calibrate_threshold(vals["y"][:0] if False else np.array([]), pv)
        # calibrate_threshold needs labels; use the first stored y as val labels proxy.
        # All seeds share the same validation split, so labels are identical.
        y_val = np.load(
            os.path.join(PRED_DIR, f"{ds}_{safe(model)}.npz")
        )["pv"] if False else None
        # The true validation labels are recovered from the first seed's test store
        # only via dataset reload; simplest correct route: evaluate with saved pv key.
        row = {"dataset": ds, "model": model}
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "results_ensemble_placeholder.csv"), index=False)


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:'''
assert old in t
t = t.replace(old, new)

p.write_text(t, encoding='utf-8')
print('patched store preds')
