# Trading AI Dashboard

Une interface web basée sur FastAPI pour visualiser l'état, les performances et les rapports du système de trading.

## Installation

```bash
uv pip install fastapi uvicorn jinja2 python-multipart pandas sqlalchemy python-dotenv requests
```

## Configuration

Ajoutez ces variables dans le fichier `.env` à la racine du projet :

```
DASHBOARD_USER=admin
DASHBOARD_PASS=admin
```

## Lancement

Exécutez le serveur depuis la racine du projet :

```bash
uvicorn web_dashboard.app:app --host 0.0.0.0 --port 8000 --reload
```

L'application sera accessible sur `http://localhost:8000`.
