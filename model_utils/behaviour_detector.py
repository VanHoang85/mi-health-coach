import re
import string

from model_utils.agent import Agent
from data_utils.beh_detect_prompts import BEHAVIOUR_INSTRUCTION, QUESTION_INSTRUCTION


class BehDetector(Agent):
    def __init__(self, args):
        super().__init__(args=args, model_type=args.beh_detector_model)

    def detect_beh_therapist_from_dialogue(self, dialogue: list):
        behaviour_prompt = BEHAVIOUR_INSTRUCTION.format(dialogue="\n".join(dialogue[-3:]))
        behaviour_pred = self.send_message(prompt=behaviour_prompt, task="beh_detect")
        return self.parse_outputs(behaviour_pred)

    @staticmethod
    def parse_outputs(behaviour_pred: str) -> str:
        pred = behaviour_pred.split("[RESULT]")[-1]
        raw_actions = pred.strip().split("\n") if "\n" in pred else pred.strip().split(",")

        cleaned_actions = []
        for action in raw_actions:
            action = action.translate(str.maketrans('', '', string.punctuation))
            action = re.sub(r'[0-9]', '', action)
            if action.strip() not in cleaned_actions:
                cleaned_actions.append(action.strip())
        return ", ".join(cleaned_actions)


class QueDetector(Agent):
    def __init__(self, args):
        super().__init__(args=args, model_type=args.beh_detector_model)

    def detect_question_type_from_dialogue(self, dialogue: list, actions: str):
        dialogue = "\n".join(dialogue[-2:])
        dialogue += f"\nTherapist behaviour: {actions}"
        question_prompt = QUESTION_INSTRUCTION.format(dialogue=dialogue)
        question_pred = self.send_message(prompt=question_prompt, task="beh_detect")
        return self.parse_outputs(question_pred)

    @staticmethod
    def parse_outputs(question_pred: str) -> str:
        pred = question_pred.split("[RESULT]")[-1].strip()
        raw_actions = pred.strip().split("\n") if "\n" in pred else pred.strip().split(",")

        cleaned_actions = []
        for action in raw_actions:
            action = action.translate(str.maketrans('', '', string.punctuation))
            action = re.sub(r'[0-9]', '', action)
            if action.strip() not in cleaned_actions:
                cleaned_actions.append(action.strip())
        return ", ".join(cleaned_actions)
