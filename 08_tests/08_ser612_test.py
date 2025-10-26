# 08_tests/08_ser612_test.py
from pathlib import Path
import pandas as pd
import json
import sys

PROJ = Path(__file__).resolve().parents[1]

def must_exist(p: Path, label: str):
    if not p.exists():
        raise FileNotFoundError(f"[FALTA] {label}: {p}")
    print(f"[OK] {label}: {p}")

def main():
    # Rutas fijas de salida del YAML de este módulo
    out_base = PROJ / "05_results/nrp1_xylt1/08_ser612"
    dist = out_base / "08_ser612/distance_timeseries.csv"
    sasa = out_base / "08_ser612/sasa_timeseries.csv"
    acc  = out_base / "08_ser612/accessibility_score.json"

    must_exist(dist, "distance_timeseries.csv")
    df_d = pd.read_csv(dist)
    print(df_d.head(3))

    if sasa.exists():
        print("[INFO] SASA encontrado.")
        df_s = pd.read_csv(sasa)
        print(df_s.head(3))

    if acc.exists():
        print("[INFO] accessibility_score.json encontrado.")
        data = json.loads(acc.read_text(encoding="utf-8"))
        keys = ["avg_distance_A","avg_sasa_A2","accessibility_score"]
        miss = [k for k in keys if k not in data]
        if miss:
            raise AssertionError(f"Campos faltantes en accessibility_score.json: {miss}")
        print({k: data[k] for k in keys})

    print("\n✅ TEST Módulo 8 OK")

if __name__ == "__main__":
    main()
