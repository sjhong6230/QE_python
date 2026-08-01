from qepy_pw.memory import current_rss_bytes, format_bytes, peak_rss_bytes


def test_memory_units_and_process_measurements():
    assert format_bytes(0) == "0.00 B"
    assert format_bytes(1536) == "1.50 KiB"
    assert current_rss_bytes() >= 0
    assert peak_rss_bytes() >= 0
