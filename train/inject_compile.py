from pathlib import Path
import sys

path = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("examples/flux2/model_training/train.py")
source = path.read_text(encoding="utf-8")
line = '        self.pipe.dit = torch.compile(self.pipe.dit, mode="reduce-overhead", fullgraph=False)'
anchor = "        # Other configs"

if line not in source:
    if source.count(anchor) != 1:
        raise RuntimeError(f"Expected one insertion anchor in {path}")
    source = source.replace(anchor, line + "\n\n" + anchor, 1)
    path.write_text(source, encoding="utf-8")
