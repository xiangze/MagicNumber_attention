FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 依存パッケージ
RUN pip install --no-cache-dir \
    torchvision==0.17.0 \
    matplotlib \
    google-cloud-storage

# コードをコピー
COPY src/lyapunov_model.py src
COPY trainer/ trainer/

# データキャッシュ用ディレクトリ
RUN mkdir -p /app/data

ENTRYPOINT ["python", "-m", "trainer.task"]