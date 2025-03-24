
ATTITUDE_INSTRUCTIONS = {
    "simple": [
        "You are given a dialogue between the therapist and the client.\n{dialogue}\nPredict the client attitude. "
        "Options are \"change\" (motivation toward behaviour change), "
        "\"neutral\" (neutral attitude or not enough information) "
        "or \"sustain\" (resistance against behaviour change)."
        "\nThe output should be a string: [CHOICE] change or neutral or sustain",  # default choice
        "Consider this dialogue between the therapist and the client:\n{dialogue}\nWhat is the client attitude? "
        "Choose \"change\" if the client is motivated toward behaviour change, "
        "\"sustain\" if the client shows resistance against behaviour change, or "
        "\"neutral\" if no information is available."
        "\nThe output should be a string: [CHOICE] change or neutral or sustain",
        "{dialogue}\nHow would one describe the client attitude in the dialogue: "
        "\"change\" (motivation toward behaviour change), \"sustain\" (resistance against behaviour change), "
        "or \"neutral\" (neutral attitude or not enough information). "
        "The output should be a string: [CHOICE] change or neutral or sustain"
    ],
    "detailed": [
        "### Instruction: You are given a dialogue between the therapist and the client about a certain goal.\n1. Infer what kind of change behaviour the client is trying to change regarding the goal.\n2. predict the client attitude. You should refer to the label space."
        "\n### Goal: {topic}"
        "\n\n### Dialogue to evaluate:"
        "\n{dialogue}"
        "\n\n### Label Space:"
        "\nOptions are:\n\"change\": (indications of intent or effort to alter behavior towards a goal. This includes expressing motivation to change, considering alternatives, articulating reasons, assessing ability, recognizing need, reporting aligned actions, acknowledging problems, identifying benefits, discussing strategies to overcome difficulties, and considering supportive actions. It reflects readiness, progress, willingness to challenge habits, and alignment with the proposed change).\n"
        "\"neutral\" (neutral statements or behaviors not directly indicating motivation for change or commitment to current behavior. This includes information requests, questions, storytelling, factual reporting, acknowledgments without personal views, small talk, and descriptions of circumstances. These engage in the therapeutic process without leaning towards change or status quo, providing context but not directly contributing to change or reinforcing current patterns).\n"
        "\"sustain\" (indications of maintaining or intensifying current patterns, opposing a stated change goal. This includes expressing reluctance, defending the status quo, minimizing problems, highlighting benefits of current behavior, articulating difficulties, identifying risks of change, and considering contrary actions. It reflects resistance, ambivalence, or movement away from change, indicating attachment to current habits or misalignment with the proposed direction).\n"
        "\nThe output should be a string: [CHOICE] the label (change or neutral or sustain)"
    ]
}

STRENGTH_INSTRUCTIONS = {
    "simple": [
        "What is the certainty level of the client attitude in the dialogue between the therapist and the client? Choose \"high\" (very certain), \"medium\" (certain), or \"low\" (uncertain). "
        "\n{dialogue}\nThe output should be a string: [CHOICE] high or medium or low"
    ],
    "detailed": [
        "### Instruction: You are given a client utterance.\n1. Look for indicators of commitment strength in the utterance.\n2. Predict the client attitude certainty level of the client attitude. You should refer to the label space."
        "\n\n### Dialogue to evaluate:"
        "\n{dialogue}"
        "\n\n### Label Space:\n"
        "\nOptions are:\n\"high\" (high commitment strength utterances are absolute or emphatic expressions that show unwavering certainty. Key indicators include strong boosters (e.g., \"definitely\", \"absolutely\", \"really\"), powerful verbs (e.g., \"swear\", \"guarantee\"), and hyperbolic language. A definitive tone, lack of hedging, and overall confident structure reinforce high commitment. The combination of these elements in an utterance signals the speaker's strong conviction in their statement or intended action).\n"
        "\"medium\" (medium commitment strength utterances are straightforward statements without strong amplification or qualification. They lack the emphatic certainty of high commitment and the hesitancy of low commitment. These are often simple declarative statements, short answers (\"Yes\", \"No\"), or expressions of ability or inability (\"I can do it\", \"I couldn't do it\"). The tone is neutral, neither strongly confident nor uncertain).\n"
        "\"low\" (utterances with low commitment strength exhibit uncertainty or hesitancy. They're characterized by \"hedges\" or qualifiers that diminish conviction. Key markers include phrases like \"I guess\", \"maybe\", \"kind of\", and \"sort of\" as well as moderating terms such as \"mostly\" or \"probably\". The overall tone suggests reluctance or a lack of confidence in the statement or intended change action.).\n"
        "\n{dialogue}\nThe output should be a string: [CHOICE] the label (high or medium or low)"
    ]
}

ATTITUDE_INSTRUCTIONS_WITH_REASONING = {
    "simple": [
        "You are given a dialogue between the therapist and the client.\n{dialogue}\nPredict the client attitude. "
        "Options are \"change\" (motivation toward behaviour change), "
        "\"neutral\" (neutral attitude or not enough information) "
        "or \"sustain\" (resistance against behaviour change)."
        "\nGenerate a brief reasoning before giving the final choice. "
        "The output should be a string: (brief reasoning) [CHOICE] change or neutral or sustain"
    ],
    "detailed": [
        "### Instruction: You are given a dialogue between the therapist and the client about a certain goal.\n1. Infer what kind of change behaviour the client is trying to change regarding the goal.\n2. predict the client attitude. You should refer to the label space."
        "\n### Goal: {topic}"
        "\n\n### Dialogue to evaluate:"
        "\n{dialogue}"
        "\n\n### Label Space:\n"
        "Options are:\n\"change\": (indications of intent or effort to alter behavior towards a goal. This includes expressing motivation to change, considering alternatives, articulating reasons, assessing ability, recognizing need, reporting aligned actions, acknowledging problems, identifying benefits, discussing strategies to overcome difficulties, and considering supportive actions. It reflects readiness, progress, willingness to challenge habits, and alignment with the proposed change).\n"
        "\"neutral\" (neutral statements or behaviors not directly indicating motivation for change or commitment to current behavior. This includes information requests, questions, storytelling, factual reporting, acknowledgments without personal views, small talk, and descriptions of circumstances. These engage in the therapeutic process without leaning towards change or status quo, providing context but not directly contributing to change or reinforcing current patterns).\n"
        "\"sustain\" (indications of maintaining or intensifying current patterns, opposing a stated change goal. This includes expressing reluctance, defending the status quo, minimizing problems, highlighting benefits of current behavior, articulating difficulties, identifying risks of change, and considering contrary actions. It reflects resistance, ambivalence, or movement away from change, indicating attachment to current habits or misalignment with the proposed direction)."
        "\n\nGenerate a brief reasoning before giving the final choice. "
        "The output should be a string: (brief reasoning) [CHOICE] the label (change or neutral or sustain)"
    ]
}

STRENGTH_INSTRUCTIONS_WITH_REASONING = {
    "simple": [
        "What is the certainty level of the client attitude in the dialogue between the therapist and the client? "
        "Choose \"high\" (very certain), \"medium\" (certain), or \"low\" (uncertain).\n{dialogue}"
        "\nGenerate a brief reasoning before giving the final choice. "
        "The output should be a string: (brief reasoning) [CHOICE] high or medium or low"
    ],
    "detailed": [
        "### Instruction: You are given a client utterance.\n1. Look for indicators of commitment strength in the utterance.\n2. Predict the client attitude certainty level of the client attitude. You should refer to the label space."
        "\n\n### Dialogue to evaluate:"
        "\n{dialogue}"
        "\n\n### Label Space:\n"
        "\nOptions are:\n\"high\" (high commitment strength utterances are absolute or emphatic expressions that show unwavering certainty. Key indicators include strong boosters (e.g., \"definitely\", \"absolutely\", \"really\"), powerful verbs (e.g., \"swear\", \"guarantee\"), and hyperbolic language. A definitive tone, lack of hedging, and overall confident structure reinforce high commitment. The combination of these elements in an utterance signals the speaker's strong conviction in their statement or intended action).\n"
        "\"medium\" (medium commitment strength utterances are straightforward statements without strong amplification or qualification. They lack the emphatic certainty of high commitment and the hesitancy of low commitment. These are often simple declarative statements, short answers (\"Yes\", \"No\"), or expressions of ability or inability (\"I can do it\", \"I couldn't do it\"). The tone is neutral, neither strongly confident nor uncertain).\n"
        "\"low\" (utterances with low commitment strength exhibit uncertainty or hesitancy. They're characterized by \"hedges\" or qualifiers that diminish conviction. Key markers include phrases like \"I guess\", \"maybe\", \"kind of\", and \"sort of\" as well as moderating terms such as \"mostly\" or \"probably\". The overall tone suggests reluctance or a lack of confidence in the statement or intended change action.).\n"
        "\nGenerate a brief reasoning before giving the final choice. "
        "The output should be a string: (brief reasoning) [CHOICE] the label (high or medium or low)"
    ]
}

ATTITUDE_INSTRUCTIONS_WITH_ICL = {
    "simple": [
        "You are given a dialogue between the therapist and the client. Predict the client attitude. "
        "Options are \"change\" (motivation toward behaviour change), \"sustain\" (resistance against behaviour change), "
        "or \"neutral\" (neutral attitude or not enough information)."
        "{samples}\n\nNow, predict the client attitude in the following dialogue:\n{dialogue}"
        "\nThe output should be: [CHOICE] change or neutral or sustain",
    ],
    "detailed": [
        "### Instruction: You are given a dialogue between the therapist and the client about a certain goal.\n1. Infer what kind of change behaviour the client is trying to change regarding the goal.\n2. predict the client attitude. You should refer to the label space."
        "\n\n### Label Space:\n"
        "Options are:\n\"change\": (indications of intent or effort to alter behavior towards a goal. This includes expressing motivation to change, considering alternatives, articulating reasons, assessing ability, recognizing need, reporting aligned actions, acknowledging problems, identifying benefits, discussing strategies to overcome difficulties, and considering supportive actions. It reflects readiness, progress, willingness to challenge habits, and alignment with the proposed change).\n"
        "\"neutral\" (neutral statements or behaviors not directly indicating motivation for change or commitment to current behavior. This includes information requests, questions, storytelling, factual reporting, acknowledgments without personal views, small talk, and descriptions of circumstances. These engage in the therapeutic process without leaning towards change or status quo, providing context but not directly contributing to change or reinforcing current patterns).\n"
        "\"sustain\" (indications of maintaining or intensifying current patterns, opposing a stated change goal. This includes expressing reluctance, defending the status quo, minimizing problems, highlighting benefits of current behavior, articulating difficulties, identifying risks of change, and considering contrary actions. It reflects resistance, ambivalence, or movement away from change, indicating attachment to current habits or misalignment with the proposed direction).\n"
        "\n### Here are some examples:"
        "\n{samples}\n\nNow, infer the client's change behaviour and predict the client attitude in the following dialogue:"
        "\n### Goal: {topic}"
        "\n\n### Dialogue to evaluate:"
        "\n{dialogue}"
        "\nThe output should be: [CHOICE] the label (change or neutral or sustain)",
    ]
}

STRENGTH_INSTRUCTIONS_WITH_ICL = {
    "simple": [
        "What is the certainty level of the client attitude in the dialogue between the therapist and the client? "
        "Choose \"high\" (very certain), \"medium\" (certain), or \"low\" (uncertain)."
        "{samples}\n\nNow, predict the certainty level of the client attitude in the following dialogue:\n{dialogue}"
        "\nThe output should be a string: [CHOICE] high or medium or low",
    ],
    "detailed": [
        "### Instruction: You are given a client utterance.\n1. Look for indicators of commitment strength in the utterance.\n2. Predict the client attitude certainty level of the client attitude. You should refer to the label space."
        "\n\n### Label Space:\n"
        "Options are:\n\"high\" (high commitment strength utterances are absolute or emphatic expressions that show unwavering certainty. Key indicators include strong boosters (e.g., \"definitely\", \"absolutely\", \"really\"), powerful verbs (e.g., \"swear\", \"guarantee\"), and hyperbolic language. A definitive tone, lack of hedging, and overall confident structure reinforce high commitment. The combination of these elements in an utterance signals the speaker's strong conviction in their statement or intended action).\n"
        "\"medium\" (medium commitment strength utterances are straightforward statements without strong amplification or qualification. They lack the emphatic certainty of high commitment and the hesitancy of low commitment. These are often simple declarative statements, short answers (\"Yes\", \"No\"), or expressions of ability or inability (\"I can do it\", \"I couldn't do it\"). The tone is neutral, neither strongly confident nor uncertain).\n"
        "\"low\" (utterances with low commitment strength exhibit uncertainty or hesitancy. They're characterized by \"hedges\" or qualifiers that diminish conviction. Key markers include phrases like \"I guess\", \"maybe\", \"kind of\", and \"sort of\" as well as moderating terms such as \"mostly\" or \"probably\". The overall tone suggests reluctance or a lack of confidence in the statement or intended change action.).\n"
        "\n### Here are some examples:"
        "{samples}"
        "\n\n### Dialogue to evaluate:"
        "\nThe output should be a string: [CHOICE] the label (high or medium or low)"

    ]
}

ATTITUDE_INSTRUCTIONS_WITH_ICL_WITH_REASONING = {
    "simple": [
        "You are given a dialogue between the therapist and the client. Predict the client attitude. "
        "Options are \"change\" (motivation toward behaviour change), \"sustain\" (resistance against behaviour change), "
        "or \"neutral\" (neutral attitude or not enough information)."
        "{samples}\n\nNow, predict the client attitude in the following dialogue:\n{dialogue}"
        "\nGenerate a brief reasoning before giving the final choice. "
        "The output should be a string: (brief reasoning) [CHOICE] change or neutral or sustain"
    ],
    "detailed": [
        "### Instruction: You are given a dialogue between the therapist and the client about a certain goal.\n1. Infer what kind of change behaviour the client is trying to change regarding the goal.\n2. predict the client attitude. You should refer to the label space."
        "\n\n### Label Space:\n"
        "Options are:\n\"change\": (indications of intent or effort to alter behavior towards a goal. This includes expressing motivation to change, considering alternatives, articulating reasons, assessing ability, recognizing need, reporting aligned actions, acknowledging problems, identifying benefits, discussing strategies to overcome difficulties, and considering supportive actions. It reflects readiness, progress, willingness to challenge habits, and alignment with the proposed change).\n"
        "\"neutral\" (neutral statements or behaviors not directly indicating motivation for change or commitment to current behavior. This includes information requests, questions, storytelling, factual reporting, acknowledgments without personal views, small talk, and descriptions of circumstances. These engage in the therapeutic process without leaning towards change or status quo, providing context but not directly contributing to change or reinforcing current patterns).\n"
        "\"sustain\" (indications of maintaining or intensifying current patterns, opposing a stated change goal. This includes expressing reluctance, defending the status quo, minimizing problems, highlighting benefits of current behavior, articulating difficulties, identifying risks of change, and considering contrary actions. It reflects resistance, ambivalence, or movement away from change, indicating attachment to current habits or misalignment with the proposed direction).\n"
        "\n### Here are some examples:"
        "\n{samples}\n\nNow, infer the client's change behaviour and predict the client attitude in the following dialogue:"
        "\n### Goal: {topic}"
        "\n\n### Dialogue to evaluate:"
        "\n{dialogue}"
        "\n\nGenerate a brief reasoning before giving the final choice. "
        "The output should be a string: (brief reasoning) [CHOICE] the label (change or neutral or sustain)"
    ]
}

STRENGTH_INSTRUCTIONS_WITH_ICL_WITH_REASONING = {
    "simple": [
        "What is the certainty level of the client attitude in the dialogue between the therapist and the client? "
        "Choose \"high\" (very certain), \"medium\" (certain), or \"low\" (uncertain)."
        "{samples}\n\nNow, predict the certainty level of the client attitude in the following dialogue:\n{dialogue}"
        "\nGenerate a brief reasoning before giving the final choice. "
        "The output should be a string: (brief reasoning) [CHOICE] high or medium or low"
    ],
    "detailed": [
        "### Instruction: You are given a client utterance.\n1. Look for indicators of commitment strength in the utterance.\n2. Predict the client attitude certainty level of the client attitude. You should refer to the label space."
        "\n\n### Label Space:\n"
        "Options are:\n\"high\" (high commitment strength utterances are absolute or emphatic expressions that show unwavering certainty. Key indicators include strong boosters (e.g., \"definitely\", \"absolutely\", \"really\"), powerful verbs (e.g., \"swear\", \"guarantee\"), and hyperbolic language. A definitive tone, lack of hedging, and overall confident structure reinforce high commitment. The combination of these elements in an utterance signals the speaker's strong conviction in their statement or intended action).\n"
        "\"medium\" (medium commitment strength utterances are straightforward statements without strong amplification or qualification. They lack the emphatic certainty of high commitment and the hesitancy of low commitment. These are often simple declarative statements, short answers (\"Yes\", \"No\"), or expressions of ability or inability (\"I can do it\", \"I couldn't do it\"). The tone is neutral, neither strongly confident nor uncertain).\n"
        "\"low\" (utterances with low commitment strength exhibit uncertainty or hesitancy. They're characterized by \"hedges\" or qualifiers that diminish conviction. Key markers include phrases like \"I guess\", \"maybe\", \"kind of\", and \"sort of\" as well as moderating terms such as \"mostly\" or \"probably\". The overall tone suggests reluctance or a lack of confidence in the statement or intended change action).\n"
        "\n### Here are some examples:"
        "{samples}"
        "\n\n### Dialogue to evaluate:"
        "\n{dialogue}"
        "\nGenerate a brief reasoning before giving the final choice. "
        "The output should be a string: (brief reasoning) [CHOICE] the label (high or medium or low)"
    ]
}


def get_summary_prompt(dialogue: str) -> str:
    return f"""Your job is to generate a fine-grained description of the client's mental state in the motivational interviewing dialogue. 
    Briefly summarize the dialogue in 1-2 sentences. Emphasizing whether the client is ready or still resistant to making a positive behaviour change.
    Be general, make sure your rule is generalizable across topics. 
    Do not be too specific. Use 'bad habit' or 'bad behaviour' instead of 'drug abuse, alcohol, smoking, exercises, or physical activity'. 
    If no attitude towards behaviour change is detected, just say so. 
    
    The dialogue:
    {dialogue}

    Summary:
    """


def get_prompt_for_task(task: str,
                        topic: str,
                        dialogue: str,
                        in_context_text="",
                        with_api=False,
                        with_reasoning=False) -> str:
    params = {
        "topic": topic,
        "dialogue": dialogue,
        "samples": in_context_text
    }

    label_type = "detailed" if with_api else "simple"

    if task == "attitude" and len(in_context_text) > 0 and with_reasoning:
        return ATTITUDE_INSTRUCTIONS_WITH_ICL_WITH_REASONING[label_type][0].format(**params)
    elif task == "attitude" and len(in_context_text) > 0 and not with_reasoning:
        return ATTITUDE_INSTRUCTIONS_WITH_ICL[label_type][0].format(**params)
    elif task == "attitude" and len(in_context_text) == 0 and with_reasoning:
        return ATTITUDE_INSTRUCTIONS_WITH_REASONING[label_type][0].format(**params)
    elif task == "attitude" and len(in_context_text) == 0 and not with_reasoning:
        return ATTITUDE_INSTRUCTIONS[label_type][0].format(**params)

    elif task == "strength" and len(in_context_text) > 0 and with_reasoning:
        return STRENGTH_INSTRUCTIONS_WITH_ICL_WITH_REASONING[label_type][0].format(**params)
    elif task == "strength" and len(in_context_text) > 0 and not with_reasoning:
        return STRENGTH_INSTRUCTIONS_WITH_ICL[label_type][0].format(**params)
    elif task == "strength" and len(in_context_text) == 0 and with_reasoning:
        return STRENGTH_INSTRUCTIONS_WITH_REASONING[label_type][0].format(**params)
    elif task == "strength" and len(in_context_text) == 0 and not with_reasoning:
        return STRENGTH_INSTRUCTIONS[label_type][0].format(**params)
