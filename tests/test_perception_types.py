from liminal.perception import InspectionRequest, SceneAssessment, Verdict


def test_scene_assessment_parses_valid_output_and_fails_closed():
    request = InspectionRequest(
        predicates=("lid_open", "access_clear", "access_blocked"),
        cameras=("top", "left_wrist"),
        exclusive_groups=(("access_clear", "access_blocked"),),
    )
    assessment = SceneAssessment.from_model_output(request, {
        "predicates": {
            "lid_open": {"value": "true", "evidence": "interior visible", "confidence": 0.9},
            "access_clear": {"value": False, "evidence": "cable crosses target"},
            "access_blocked": {"value": "true", "evidence": "cable crosses target"},
        }
    })
    assert assessment.observation("lid_open").value is Verdict.TRUE
    assert assessment.observation("access_clear").value is Verdict.FALSE
    assert assessment.observation("access_blocked").value is Verdict.TRUE

    malformed = SceneAssessment.from_model_output(request, "not JSON")
    assert all(not observation.known for observation in malformed.observations.values())

    contradictory = SceneAssessment.from_model_output(request, {
        "predicates": {
            "lid_open": {"value": "true"},
            "access_clear": {"value": "true"},
            "access_blocked": {"value": "true"},
        }
    })
    assert contradictory.observation("access_clear").value is Verdict.UNKNOWN
    assert contradictory.observation("access_blocked").value is Verdict.UNKNOWN
