export REPO_ROOT=$(git rev-parse --show-toplevel)
export $(cat .env | xargs)
export MODEL_DIRECTORY=~/.cache/models/
 export USERID="$(id -u):$(id -g)"

 # Export path where NIMs are hosted
 # LLM server path
 export APP_LLM_SERVERURL=nemollm-inference:8000
 # Embedding server path
 export APP_EMBEDDINGS_SERVERURL=nemollm-embedding:8000
 # Re-ranking model path
 export APP_RANKING_SERVERURL=ranking-ms:8000


 docker compose -f deploy/compose/docker-compose.yaml --profile local-nim up -d

