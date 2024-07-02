from src.serving.monitoring import append_jsonl_record, build_inference_log_entry


def test_build_inference_log_entry_contains_expected_fields():
    entry = build_inference_log_entry(
        store=1,
        start_date="2015-07-31",
        forecast_days=7,
        promo=1,
        state_holiday="0",
        school_holiday=1,
        model_version="v1",
        latency_ms=12.3456,
    )

    assert entry["store"] == 1
    assert entry["forecast_days"] == 7
    assert entry["model_version"] == "v1"
    assert entry["latency_ms"] == 12.346


def test_append_jsonl_record_writes_one_line(tmp_path):
    destination = tmp_path / "logs" / "inference.jsonl"
    append_jsonl_record(destination, {"hello": "world"})

    assert destination.exists()
    assert destination.read_text(encoding="utf-8").strip() == '{"hello": "world"}'
