import pathlib
p = pathlib.Path('torch_common.py')
t = p.read_text(encoding='utf-8')

old = '''def build_model(
    time_steps: int,
    n_features: int,
    attention: str = "sigmoid",
    mc_dropout: bool = True,
    dropout_rate: float = 0.15,
    attn_temperature: float = 0.5,
    attn_normalize: bool = True,
    seed: int = 42,
) -> ProposedModel:
    set_seed(seed)
    return ProposedModel(
        time_steps,
        n_features,
        attention=attention,
        mc_dropout=mc_dropout,
        dropout_rate=dropout_rate,
        attn_temperature=attn_temperature,
        attn_normalize=attn_normalize,
    ).to(DEVICE)'''
new = '''def build_model(
    time_steps: int,
    n_features: int,
    attention: str = "sigmoid",
    mc_dropout: bool = True,
    dropout_rate: float = 0.15,
    attn_temperature: float = 0.5,
    attn_normalize: bool = True,
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
    seed: int = 42,
) -> ProposedModel:
    set_seed(seed)
    return ProposedModel(
        time_steps,
        n_features,
        attention=attention,
        mc_dropout=mc_dropout,
        lstm_units_1=lstm_units_1,
        lstm_units_2=lstm_units_2,
        dropout_rate=dropout_rate,
        attn_temperature=attn_temperature,
        attn_normalize=attn_normalize,
    ).to(DEVICE)'''
assert old in t
t = t.replace(old, new)

old = '''def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    use_class_weight: bool = True,
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 5e-4,
    verbose: int = 0,
    seed: int = 42,
) -> Dict[str, list]:
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=4, min_lr=1e-6
    )
    x_tr = torch.from_numpy(x_train).float().to(DEVICE)
    y_tr = torch.from_numpy(y_train).float().to(DEVICE)
    x_va = torch.from_numpy(x_val).float().to(DEVICE)
    y_va = torch.from_numpy(y_val).float().to(DEVICE)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    pw = max(1.0, neg / max(pos, 1)) if use_class_weight else 1.0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(DEVICE))

    n = len(x_train)
    best_val = float("inf")
    best_state = None
    patience = 0
    history = {"loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total, count = 0.0, 0
        for i in range(0, n, batch_size):'''
new = '''def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    use_class_weight: bool = True,
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 5e-4,
    verbose: int = 0,
    seed: int = 42,
    pos_weight_scale: float = 1.0,
    balanced_sampling: bool = False,
) -> Dict[str, list]:
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=4, min_lr=1e-6
    )
    x_tr = torch.from_numpy(x_train).float().to(DEVICE)
    y_tr = torch.from_numpy(y_train).float().to(DEVICE)
    x_va = torch.from_numpy(x_val).float().to(DEVICE)
    y_va = torch.from_numpy(y_val).float().to(DEVICE)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    pw = (max(1.0, neg / max(pos, 1)) if use_class_weight else 1.0) * pos_weight_scale
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(DEVICE))
    if balanced_sampling and pos > 0 and neg > 0:
        w = np.where(y_train == 1, 0.5 / pos, 0.5 / neg)
        sample_weights = torch.from_numpy(w).double().to(DEVICE)
    else:
        sample_weights = None

    n = len(x_train)
    best_val = float("inf")
    best_state = None
    patience = 0
    history = {"loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        if sample_weights is not None:
            perm = torch.multinomial(sample_weights, n, replacement=True)
        else:
            perm = torch.randperm(n, device=DEVICE)
        total, count = 0.0, 0
        for i in range(0, n, batch_size):'''
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding='utf-8')
print('patched torch_common')
