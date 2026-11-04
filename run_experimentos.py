"""Roda todas as configuracoes (criterio de data x IOPs) com N sementes e guarda cada uma em results/exp_*/.
    .venv\\Scripts\\python.exe run_experimentos.py [--seeds 10]
"""
import shutil, subprocess, sys
from pathlib import Path

R = Path(__file__).parent
py = sys.executable
seeds = sys.argv[sys.argv.index("--seeds") + 1] if "--seeds" in sys.argv else "10"
ARQS = ["tabela_raio_vs_positivos.csv", "dataset_meta.json", "registros_juncao.csv", "correlacao_pearson.csv",
        "cl_detalhe_por_semente.csv", "metricas_por_semente.csv", "metricas_resumo_media_dp.csv", "resumo.json"]


def rodar(script, *args):
    r = subprocess.run([py, str(R / "src" / script), *args], cwd=R / "src", capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-1500:])
        raise SystemExit(f"falhou: {script} {args}")
    return r.stdout


for criterio in ("ano2014", "janelas"):
    for iops in ("1,2", "1", "2"):
        nome = f"exp_{criterio}_iop{iops.replace(',', '+')}"
        print("==", nome, flush=True)
        rodar("03_registros.py", "--criterio", criterio, "--ignorar-contagem")
        rodar("04_rotulos.py", "--iops", iops, "--raios", "0,500,1000,1500,2000,2250,2500,2750,3000,3500,4000")
        rodar("05_pipeline.py", "--seeds", seeds)
        dest = R / "results" / nome
        dest.mkdir(exist_ok=True)
        for a in ARQS:
            shutil.copy(R / "results" / a, dest / a)
        print("   ok", flush=True)
