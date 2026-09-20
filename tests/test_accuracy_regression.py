import csv
import yaml


def test_tensorrt_fp16_map_regression():
    with open("configs/params.yaml") as f:
        cfg = yaml.safe_load(f)

    with open("results/accuracy.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    reference = next(
        row for row in rows
        if row["runtime"] == "torchscript" and row["precision"] == "fp32"
    )
    candidate = next(
        row for row in rows
        if row["runtime"] == "tensorrt" and row["precision"] == "fp16"
    )

    reference_map = float(reference["mAP50_95"])
    candidate_map = float(candidate["mAP50_95"])
    threshold = float(cfg["evaluation"]["map_regression_threshold"])

    drop = reference_map - candidate_map

    assert drop <= threshold, (
        f"TensorRT FP16 mAP regression {drop:.5f} exceeds allowed {threshold:.5f} "
        f"(reference {reference_map:.5f}, candidate {candidate_map:.5f})"
    )
