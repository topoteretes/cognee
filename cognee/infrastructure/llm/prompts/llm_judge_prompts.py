# LLM-as-a-judge metrics as described here: https://arxiv.org/abs/2404.16130

llm_judge_prompts = {
    "correctness": "Determine whether the actual output is factually correct based on the expected output.",
    "comprehensiveness": "Determine how much detail the answer provides to cover all the aspects and details of the question.",
    "diversity": "Determine how varied and rich the answer is in providing different perspectives and insights on the question.",
    "empowerment": "Determine how well the answer helps the reader understand and make informed judgements about the topic.",
    "directness": "Determine how specifically and clearly the answer addresses the question.",
}
