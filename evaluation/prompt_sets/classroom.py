from __future__ import annotations

from prompts import (
    converge_scoring_rubric,
    evaluation_criteria_division,
    evaluation_criteria_keyword,
    evaluation_criteria_question,
    question_synonym,
    scoring,
    scoring_mutual_cross,
    scoring_rubric,
    scoring_rubric_mutual_cross,
    scoring_rubric_problem,
)


CLASSROOM_PROMPTS = {
    "evaluation_criteria_division": evaluation_criteria_division,
    "evaluation_criteria_keyword": evaluation_criteria_keyword,
    "evaluation_criteria_question": evaluation_criteria_question,
    "question_synonym": question_synonym,
    "scoring": scoring,
    "scoring_rubric": scoring_rubric,
    "scoring_rubric_problem": scoring_rubric_problem,
    "converge_scoring_rubric": converge_scoring_rubric,
    "scoring_mutual_cross": scoring_mutual_cross,
    "scoring_rubric_mutual_cross": scoring_rubric_mutual_cross,
}
