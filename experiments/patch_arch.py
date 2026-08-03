import pathlib
p = pathlib.Path('draw_architecture.py')
t = p.read_text(encoding='utf-8')
t = t.replace('fs=10, bold=True', 'fs=12, bold=True')
t = t.replace('fs=10)', 'fs=12)')
t = t.replace('fig.savefig("../figures/fig_architecture.png", dpi=300, bbox_inches="tight")',
              'fig.savefig("../figures/fig_architecture.pdf", bbox_inches="tight")\nfig.savefig("../figures/fig_architecture.png", dpi=300, bbox_inches="tight")')
p.write_text(t, encoding='utf-8')
print('arch script patched')
