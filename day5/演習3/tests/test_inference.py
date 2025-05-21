import os
import pickle
import time
import pytest
import pandas as pd
from sklearn.metrics import accuracy_score

# テスト用データとモデルのパス
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/titanic_model.pkl")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        pytest.skip("データファイルが存在しないためスキップします")
    return pd.read_csv(DATA_PATH)


@pytest.fixture
def load_model():
    """保存済みモデルを読み込む"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def test_model_inference_accuracy_and_time(sample_data, load_model):
    """モデルの推論精度と速度を検証"""
    model = load_model

    # 特徴量とラベルの分割
    if "Survived" not in sample_data.columns:
        pytest.fail("Survivedカラムがデータに存在しません")
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)

    # 推論
    start = time.time()
    preds = model.predict(X)
    elapsed = time.time() - start

    # 精度
    accuracy = accuracy_score(y, preds)
    print(f"Inference time: {elapsed:.4f} sec")
    print(f"Accuracy: {accuracy:.4f}")

    # テスト条件
    assert elapsed < 1.0, f"推論時間が長すぎます: {elapsed:.4f}秒"
    assert accuracy > 0.85, f"モデルの精度が低すぎます: {accuracy:.4f}"
