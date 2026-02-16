
import pandas as pd
import pytest
import torch
from torchvision import transforms

from src import predict


@pytest.mark.slow
def test_predict_smoke(tmp_path, trained_run_dir):
    input_dir = tmp_path / "input_images"
    input_dir.mkdir(parents=True)

    for i in range(5):
        img_path = input_dir / f"img_{i}.jpg"
        img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
        img_pil = transforms.ToPILImage()(img)
        img_pil.save(img_path)


    output_path = tmp_path / "predictions.csv"

    predict.run(
        run_dir=trained_run_dir,
        input_dir=input_dir,
        output_path=output_path,
    )

    assert output_path.exists()

    df = pd.read_csv(output_path)
    assert "path" in df.columns
    assert "pred_index" in df.columns
    assert "pred_class" in df.columns
    assert "confidence" in df.columns
    assert len(df) == 5