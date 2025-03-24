COHERENCE_ABSOLUTE_PROMPT_WO_REF = """###Task Description:
An instruction (might include an Input inside it), a conversation to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a brief feedback in approximately 50 words that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###The conversation to evaluate:
{response}

###Score Rubrics:
{rubric}

###Feedback: """

FAITHFULNESS_ABSOLUTE_PROMPT_WO_REF = """###Task Description:
An instruction (might include an Input inside it), a conversation to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a brief feedback in approximately 50 words that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###The response to evaluate:
{response}

###Score Rubrics:
{rubric}

###Feedback: """



