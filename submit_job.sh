#!/bin/bash
set -e

PROJECT_ID=$(gcloud config get-value project)
REGION="asia-northeast1"
BUCKET="gs://${PROJECT_ID}-lyapunov"
IMAGE="asia-northeast1-docker.pkg.dev/${PROJECT_ID}/lyapunov/trainer:latest"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="lyapunov_${TIMESTAMP}"

# --- 1. GCS バケット作成 (初回のみ) ---
gsutil mb -l ${REGION} ${BUCKET} 2>/dev/null || true

# --- 2. Artifact Registry リポジトリ作成 (初回のみ) ---
gcloud artifacts repositories create lyapunov \
  --repository-format=docker \
  --location=${REGION} \
  --project=${PROJECT_ID} 2>/dev/null || true

# --- 3. Docker イメージをビルド & プッシュ ---
gcloud builds submit --config cloudbuild.yaml --project=${PROJECT_ID}

# --- 4. Vertex AI ジョブ投入 ---
gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=${JOB_NAME} \
  --worker-pool-spec="\
machine-type=n1-standard-8,\
accelerator-type=NVIDIA_TESLA_T4,\
accelerator-count=1,\
container-image-uri=${IMAGE}" \
  --args="\
--task=cifar10,\
--epochs=30,\
--n_blocks=6,\
--batch_size=256,\
--lyap_samples=5" \
  --environment-variables="AIP_MODEL_DIR=${BUCKET}/results/${JOB_NAME}"

echo "ジョブ投入完了: ${JOB_NAME}"
echo "結果保存先: ${BUCKET}/results/${JOB_NAME}/"
echo ""
echo "ステータス確認:"
echo "  gcloud ai custom-jobs list --region=${REGION}"