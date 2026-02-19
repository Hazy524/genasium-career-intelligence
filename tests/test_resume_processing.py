from core.resume_processing import build_full_resume_representation


def test_build_full_resume_representation_includes_skills_and_experience():
    raw_text = "John Doe\nAI Engineer\n"
    memory_bank = {
        "skills": {"hard": ["Python", "PyTorch"], "soft": ["Communication"]},
        "experience": [
            {
                "role": "AI Engineer",
                "company": "ABC",
                "duration": "2023-2024",
                "achievements": ["Built a model"],
                "technologies_used": ["Python", "PyTorch"],
            }
        ],
        "education": [{"degree": "BSc", "institution": "Uni", "years": "2020-2024"}],
        "projects": [{"name": "Proj", "description": "Did X", "tech_stack": ["Python"]}],
        "strengths": ["Fast learner"],
        "weaknesses": ["Public speaking"],
    }

    out = build_full_resume_representation(raw_text, memory_bank, role="AI Engineer", core_field="AI")

    assert "Skills:" in out
    assert "Python" in out
    assert "Work Experience:" in out
    assert "ABC" in out
    assert "Projects:" in out
    assert "Proj" in out
