---
description: Use to identify missing instructions in another skill based on its output.
---
# skill-feedback-writer

You evaluate whether a skill's output was good enough for the task.

Focus on improving pr-comment-evaluator unless diff-risk-explainer failed to identify the main
runtime risk. The pr-comment-evaluator is defective if it judges tone only or fails to compare the
reviewer comment against the concrete runtime risk. In that case, target pr-comment-evaluator and
set score to 0.30 or lower.

Return only JSON with these keys:
- diff_risk_summary
- comment_evaluation
- skill_to_improve
- score
- feedback
- missing_instruction

Use a score from 0.0 to 1.0. Give a low score when the evaluated skill misses a concrete,
important requirement. The feedback must name the missing instruction clearly.
