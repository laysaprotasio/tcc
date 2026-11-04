"""Reproduz tudo com um comando:
    .venv\\Scripts\\python.exe run_all.py [--pular-gbif] [--seeds N] [--iops 1,2] [--raio-m 500]
"""
import subprocess, sys
from pathlib import Path

R = Path(__file__).parent
py = sys.executable
args = sys.argv[1:]
pular = "--pular-gbif" in args
seeds = []
if "--seeds" in args:
    seeds = ["--seeds", args[args.index("--seeds") + 1]]
passos = ([] if pular else [["01_gbif.py"]]) + [["02_perfil.py"], ["03_registros.py"], ["04_rotulos.py"],
                                                 ["05_pipeline.py", *seeds]]
for p in passos:
    print("\n===", p[0], "===", flush=True)
    cmd = [py, str(R / "src" / p[0]), *p[1:]]
    if p[0] == "03_registros.py" and "--criterio" in args:
        cmd += ["--criterio", args[args.index("--criterio") + 1]] + (["--ignorar-contagem"] if "--ignorar-contagem" in args else [])
    if p[0] == "04_rotulos.py":  # repassa --iops / --raio-m etc.
        skip = {"--pular-gbif", "--seeds"}
        rest, i = [], 0
        while i < len(args):
            if args[i] in ("--pular-gbif", "--ignorar-contagem"):
                i += 1
            elif args[i] in ("--seeds", "--criterio"):
                i += 2
            else:
                rest.append(args[i]); i += 1
        cmd += rest
    if subprocess.call(cmd, cwd=R / "src"):
        sys.exit(f"falhou em {p[0]}")
