import pathlib
p = pathlib.Path('run_experiments_torch.py')
t = p.read_text(encoding='utf-8')

old = '''def store_preds(
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
    PRED_STORE[key]["pv"].append(pv)'''
new = '''def store_preds(
    ds: str,
    model: str,
    p: np.ndarray,
    std: np.ndarray,
    y: np.ndarray,
    pv: np.ndarray,
    y_val: np.ndarray,
) -> None:
    key = (ds, model)
    if key not in PRED_STORE:
        PRED_STORE[key] = {"p": [], "std": [], "y": y, "pv": [], "y_val": y_val}
    PRED_STORE[key]["p"].append(p)
    PRED_STORE[key]["std"].append(std)
    PRED_STORE[key]["pv"].append(pv)'''
assert old in t
t = t.replace(old, new)

t = t.replace('store_preds(ds, model_name, p_test, std_test, data["y_test"], p_val)', 'store_preds(ds, model_name, p_test, std_test, data["y_test"], p_val, data["y_val"])')
t = t.replace('store_preds(ds, model_name, p_test, np.zeros_like(p_test), data["y_test"], p_val)', 'store_preds(ds, model_name, p_test, np.zeros_like(p_test), data["y_test"], p_val, data["y_val"])')
t = t.replace('''        store_preds(
            ds, name, probs["test"], np.zeros_like(probs["test"]), data["y_test"], probs["val"]
        )''', '''        store_preds(
            ds,
            name,
            probs["test"],
            np.zeros_like(probs["test"]),
            data["y_test"],
            probs["val"],
            data["y_val"],
        )''')

old = '''def build_ensemble_csv() -> None:
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
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "results_ensemble_placeholder.csv"), index=False)'''
new = '''def build_ensemble_csv() -> None:
    rows = []
    for (ds, model), vals in PRED_STORE.items():
        pv = np.mean(np.stack(vals["pv"]), axis=0)
        p = np.mean(np.stack(vals["p"]), axis=0)
        y_val = vals["y_val"]
        threshold, _ = calibrate_threshold(y_val, pv)
        metrics = evaluate_binary(vals["y"], p, threshold)
        rows.append({"dataset": ds, "model": model, **metrics})
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT, "results_ensemble.csv"), index=False
    )'''
assert old in t
t = t.replace(old, new)

p.write_text(t, encoding='utf-8')
print('patched ensemble properly')
