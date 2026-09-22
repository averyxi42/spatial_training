import json

from longnav.scripts.check_hm3d_pe_reproducibility import load_outcomes


def test_load_outcomes_uses_execution_outcomes(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"results": [{
        "uid": "scene:0#1",
        "status": "evaluated",
        "success": 1.0,
        "oracle_success": 1.0,
        "terminated_by": "policy_stop",
        "path_length_m": 12.0,
    }]}))
    assert load_outcomes(result) == {
        "scene:0#1": ("evaluated", 1.0, 1.0, "policy_stop")
    }
