import pathlib, re
p = pathlib.Path('torch_common.py')
t = p.read_text(encoding='utf-8')
t = re.sub(r'nn\.Conv1d\(in_ch, 32, 3, padding=0\)',
           'nn.Conv1d(in_ch, 32, 3, padding="same", dilation=d)', t)
t = re.sub(r'nn\.Conv1d\(32, 32, 3, padding=0\)',
           'nn.Conv1d(32, 32, 3, padding="same", dilation=d)', t)
pattern = re.compile(
    r'for i, \(conv, dilation\) in enumerate\(zip\(self\.convs, self\.dilations\)\):\n'
    r'\s*kernel = conv\[0\]\.kernel_size\[0\]\n'
    r'\s*pad = \(kernel - 1\) \* dilation\n'
    r'\s*res = z\n'
    r'\s*z = F\.pad\(z, \(pad, 0\)\)\n'
    r'\s*y = conv\(z\)\n'
    r'\s*if i > 0:\n'
    r'\s*y = y \+ res\n'
    r'\s*z = F\.relu\(y\)'
)
replacement = (
    'for i, (conv, dilation) in enumerate(zip(self.convs, self.dilations)):\n'
    '                y = conv(z)\n'
    '                if i > 0:\n'
    '                    y = y + z\n'
    '                z = F.relu(y)'
)
t, n = pattern.subn(replacement, t)
print('forward replaced', n)
p.write_text(t, encoding='utf-8')
