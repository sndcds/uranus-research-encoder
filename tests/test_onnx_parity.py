"""No v5 ONNX parity claim is made until a compatible export is validated."""

import pytest


@pytest.mark.integration
def test_real_onnx_parity():
    pytest.skip("Jina v5 ONNX is not supported; historical v3 reports are not v5 evidence")
