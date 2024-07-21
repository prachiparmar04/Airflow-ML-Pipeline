## Repo Structure 


ML-CHURN-PIPELINE/
│
├── training-pipleine/
    ├── dags/
    │   └── utils/ churn_pipeline.py # airlfow dag pipeline for model training
    │
    └── data/
        ├── models/
        ├── embeddings/
        └── rawdata/
|
├── inference/
|   ├── resources/
|   ├── utills/
|
|
├── requirements.txt
├── README.md
├── Dockerfile
├── pod.yaml # to create k8s pod for inference deployment
└── docker-compose.yaml # to create docker for inference deployment & training pipeline


