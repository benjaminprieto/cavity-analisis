.
├── src/
│   └── cavity_analysis/
│       ├── core/             # lógica reusable (network, energy, dccm, pca, pathways, etc.)
│       ├── utils/            # utilidades (io, paths, validaciones, logging)
│       └── visualization/    # funciones de plotting/reportes
│
├── scripts/                  # CLIs delgados que orquestan cada análisis
│   ├── 10_allosteric.py
│   ├── 20_visualize.py
│   └── 30_explore_alternatives.py
│
├── configs/
│   └── analyses/
│       ├── 10_allosteric.yaml
│       ├── 20_visualize.yaml
│       └── 30_explore_alternatives.yaml
│
├── data/
│   ├── raw/                  # insumos crudos (inmutables)
│   │   ├── pdb/
│   │   ├── fasta/
│   │   └── trajectories/
│   └── processed/            # intermedios limpios/estandarizados
│
├── results/
│   └── nrp1_allosteric/      # salida fija de este flujo
│       ├── 1_network/
│       ├── 2_energy/
│       ├── 3_dccm/
│       ├── 4_pca/
│       ├── 5_pathways/
│       ├── 6_communities/
│       ├── 7_integration/
│       └── 8_ser612/
│
├── logs/                     # logs generales (opcional)
├── notebooks/                # exploración/EDA (opcional)
└── tests/                    # pruebas
