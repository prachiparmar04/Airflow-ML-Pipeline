## Repo Structure 

ML-CHURN-PIPELINE/
│
├── training-pipeline/
│   ├── dags/
│   │   └── utils/
│   │       └── churn_pipeline.py  # Airflow DAG pipeline for model training
│   └── data/
│       ├── models/
│       ├── embeddings/
│       └── rawdata/
│
├── inference/
│   ├── resources/
│   ├── utils/
│
├── requirements.txt
├── README.md
├── Dockerfile
├── pod.yaml  # To create k8s pod for inference deployment
└── docker-compose.yaml  # To create docker for inference deployment & training pipeline
