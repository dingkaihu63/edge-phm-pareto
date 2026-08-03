import pathlib

p = pathlib.Path('torch_common.py')
t = p.read_text(encoding='utf-8')

# Add safe global speedups after DEVICE definition.
old = 'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
new = '''DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass'''
assert old in t
t = t.replace(old, new)

# Add grad_clip parameter and clipping step.
old = '''    balanced_sampling: bool = False,
) -> Dict[str, list]:
    set_seed(seed)'''
new = '''    balanced_sampling: bool = False,
    grad_clip: float = 0.0,
) -> Dict[str, list]:
    set_seed(seed)'''
assert old in t
t = t.replace(old, new)

old = '''            loss = loss_fn(logits, y_tr[idx])
            loss.backward()
            opt.step()'''
new = '''            loss = loss_fn(logits, y_tr[idx])
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()'''
assert old in t
t = t.replace(old, new)

p.write_text(t, encoding='utf-8')
print('torch_common patched')
