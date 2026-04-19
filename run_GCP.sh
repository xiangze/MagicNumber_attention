# 1. gcloud 認証
gcloud auth login
gcloud config set project $Lyapunov_NN

# 2. 必要な API を有効化
gcloud services enable \
  compute.googleapis.com \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com

# 3. ジョブ投入 (方法B)
chmod +x submit_job.sh
./submit_job.sh

# --- またはタスクだけ変える場合 ---
# submit_job.sh の --args 部分を書き換えて再実行するだけ