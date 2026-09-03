import subprocess

import pytest
from PIL import Image

from k_market_ai.tax.document_model.ocr.engine import _TesseractOCR
from k_market_ai.tax.document_model.parsers.withholding_tax_form import WithholdingTaxFormParser


def test_tesseract_preserves_lines_and_limits_only_child_threads(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/tesseract")
    monkeypatch.setenv("OMP_THREAD_LIMIT", "4")
    header = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
        "left\ttop\twidth\theight\tconf\ttext\n"
    )
    rows = (
        "5\t1\t1\t1\t1\t1\t0\t0\t20\t10\t90\tTaxpayer:\n"
        "5\t1\t1\t1\t1\t2\t25\t0\t20\t10\t80\tTEST\n"
        "5\t1\t1\t1\t2\t1\t0\t12\t20\t10\t95\t2026\n"
    )

    def run(command, **kwargs):
        assert kwargs["env"]["OMP_THREAD_LIMIT"] == "1"
        assert kwargs["timeout"] == 20
        return subprocess.CompletedProcess(command, 0, (header + rows).encode(), b"")

    monkeypatch.setattr(subprocess, "run", run)
    lines = _TesseractOCR({})._ocr_page(Image.new("RGB", (1800, 2000), "white"))
    assert [line[1][0] for line in lines] == ["Taxpayer: TEST", "2026"]
    assert lines[0][1][1] == pytest.approx(0.85)
    import os

    assert os.environ["OMP_THREAD_LIMIT"] == "4"


@pytest.mark.parametrize(
    "label", ["First Name", "irst Name", "Last Name", "ast Name", "Middle Name", "iddle Name"]
)
def test_clipped_field_label_does_not_discard_name_value(label):
    assert WithholdingTaxFormParser()._compact_name_value(f"({label})\nALEX") == "ALEX"
