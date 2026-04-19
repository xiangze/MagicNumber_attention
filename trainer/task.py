"""Vertex AI エントリポイント。GCS への保存を追加。"""
import os
import subprocess
import sys

# Vertex AI は環境変数で各種パスを渡す
AIP_MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "./output")

def upload_to_gcs(local_pattern: str, gcs_dir: str):
    """ローカルファイルを GCS にアップロード。"""
    import glob
    for f in glob.glob(local_pattern):
        subprocess.run(["gsutil", "cp", f, gcs_dir + "/"], check=True)
        print(f"Uploaded: {f} → {gcs_dir}/")

if __name__ == "__main__":
    # sys.argv をそのままメインに渡す
    from lyapunov_model import main, parse_args
    import sys

    # out_prefix をローカルの /tmp に固定
    if "--out_prefix" not in sys.argv:
        sys.argv += ["--out_prefix", "/tmp/lyap_result"]
    if "--device" not in sys.argv:
        sys.argv += ["--device", "cuda"]

    main()

    # 結果を GCS にアップロード
    gcs_out = AIP_MODEL_DIR
    upload_to_gcs("/tmp/lyap_result_*.png", gcs_out)
    upload_to_gcs("/tmp/lyap_result_*.csv", gcs_out)
    print(f"全結果を {gcs_out} に保存しました。")