import re
from model_utils.agent import Agent
from data_utils.mov_detect_prompts import get_prompt_for_task
from data_utils.utils import Conversation

from llama_index.core.retrievers import VectorIndexRetriever
from vector_db import load_vector_store, create_embedder
from cohere.core.api_error import ApiError


class MovDetector(Agent):
    def __init__(self, args):
        super().__init__(args=args, model_type=args.mov_detector_model)

        self.embedder = None
        self.dialog_retriever = None
        if self.args.num_in_context_samples > 0:
            self.embedder = create_embedder(args.input_type, args.cohere_key, args.retrieval_model)
            self.dialog_retriever: VectorIndexRetriever = self.load_vector_db(args, self.embedder)

        self.mapping = {
            "change-high": 3, "change-medium": 2, "change-low": 1,
            "neutral-high": 0, "neutral-medium": 0, "neutral-low": 0,
            "sustain-low": -1, "sustain-medium": -2, "sustain-high": -3
        }

    def detect_mov_level(self, conversation: Conversation) -> dict:
        if conversation.get_current_phase() == "concluding":
            return {
                "behaviour": "n/a",
                "mov_level": "n/a",
                "attitude_pred": "n/a",
                "strength_pred": "n/a"
            }

        conv_attitude = conversation.get_conv_history_for_attitude_task(self.args.therapist_utt_setting)
        conv_strength = conversation.get_conv_history_for_strength_task(self.args.therapist_utt_setting)

        icl_attitude, icl_strength = "", ""
        if self.args.num_in_context_samples > 0:
            icl_attitude = self.retrieve_icl_samples(conv_attitude, task="attitude")
            # icl_strength = self.retrieve_icl_samples(conv_strength, task="strength")

        api_usage = False if "flan" in self.args.mov_detector_model else True
        attitude_prompt = get_prompt_for_task(task="attitude",
                                              topic="increasing activity / more exercise",
                                              dialogue=conv_attitude,
                                              in_context_text=icl_attitude,
                                              with_api=api_usage)
        attitude_pred = self.send_message(f"{attitude_prompt}", task="mov_detect")
        _, attitude = self.parse_outputs(attitude_pred, task="attitude")
        # print(f"Attitude pred: {attitude_pred}")

        strength_prompt = get_prompt_for_task(task="strength",
                                              topic="",
                                              dialogue=conv_strength,
                                              in_context_text=icl_strength,
                                              with_api=api_usage)
        strength_pred = self.send_message(f"{strength_prompt}", task="mov_detect")
        _, strength = self.parse_outputs(strength_pred, task="strength")
        # print(f"Strength pred: {strength_pred}")

        behaviour = f"{attitude}-{strength}".lower()
        level = self.map_to_level(behaviour, mapping=self.mapping)

        return {"behaviour": behaviour,
                "mov_level": level,
                "attitude_pred": attitude_pred,
                "strength_pred": strength_pred}

    def detect_mov_level_from_dialogue(self, dialogue: list):
        conv_attitude = dialogue[-1] if self.args.therapist_utt_setting == "wo_therapist" else dialogue[-2:]
        conv_strength = dialogue[-2:] if self.args.therapist_utt_setting == "w_therapist" else dialogue[-1]

        conv_attitude = "\n".join(conv_attitude)
        conv_strength = "\n".join(conv_strength)

        icl_attitude, icl_strength = "", ""
        if self.args.num_in_context_samples > 0:
            icl_attitude = self.retrieve_icl_samples(conv_attitude, task="attitude")

        api_usage = False if "flan" in self.args.mov_detector_model else True
        attitude_prompt = get_prompt_for_task(task="attitude",
                                              topic="increasing activity / more exercise",
                                              dialogue=conv_attitude,
                                              in_context_text=icl_attitude,
                                              with_api=api_usage)
        attitude_pred = self.send_message(f"{attitude_prompt}", task="mov_detect")
        _, attitude = self.parse_outputs(attitude_pred, task="attitude")

        strength_prompt = get_prompt_for_task(task="strength",
                                              topic="",
                                              dialogue=conv_strength,
                                              in_context_text=icl_strength,
                                              with_api=api_usage)
        strength_pred = self.send_message(f"{strength_prompt}", task="mov_detect")
        _, strength = self.parse_outputs(strength_pred, task="strength")

        behaviour = f"{attitude}-{strength}".lower()
        level = self.map_to_level(behaviour, mapping=self.mapping)

        return {"behaviour": behaviour,
                "mov_level": level,
                "attitude_pred": attitude_pred,
                "strength_pred": strength_pred}

    def retrieve_icl_samples(self, dialogue: str, task: str) -> str:
        icl = ""
        num_icl = 0
        candidates = None
        while not candidates:
            try:
                candidates = self.dialog_retriever.retrieve(dialogue)
            except Exception or ApiError:
                print("Retry retrieving...")

        for idx, candidate in enumerate(candidates):
            # remove too long icl sample --> len > 50
            if len(candidate.text.split(" ")) > 50:
                continue

            sample = f"\n\nExample {idx + 1}:\n{candidate.text}"
            if task == "attitude" and "attitude" in candidate.metadata:
                icl += f"{sample}\n[CHOICE] {candidate.metadata['attitude']}"
                num_icl += 1

            elif task == "strength" and "strength" in candidate.metadata:
                icl += f"{sample}\n[CHOICE] {candidate.metadata['strength']}"
                num_icl += 1

            if num_icl == self.args.num_in_context_samples:
                break
        return icl

    @staticmethod
    def map_to_level(behaviour: str, mapping: dict) -> int:
        return mapping[behaviour]

    @staticmethod
    def check_attitude_pred(pred: str) -> str:
        return pred.lower().strip() if pred.lower().strip() in ["change", "neutral", "sustain"] else "neutral"

    @staticmethod
    def check_strength_pred(pred: str) -> str:
        return pred.lower().strip() if pred.lower().strip() in ["high", "medium", "low"] else "medium"

    @staticmethod
    def check_pred_value(pred: str) -> bool:
        return True if pred.lower().strip() in ["change", "neutral", "sustain", "high", "medium", "low"] else False

    def parse_outputs(self, pred: str, task: str) -> tuple[str, str]:
        outs = pred.strip().split("[CHOICE]")
        if len(outs) == 2:
            pred_value = outs[1].strip()
            if self.check_pred_value(pred_value):
                return outs[0].strip(), pred_value.lower()

        pattern = r'\b(change|neutral|sustain)\b' if task == "attitude" else r'\b(high|medium|low)\b'
        matches = re.findall(pattern, pred.strip(), re.IGNORECASE)
        if len(matches) > 0:
            return "", matches[0]
        return ("", "neutral") if task == "attitude" else ("", "medium")

    @staticmethod
    def load_vector_db(args, embedder):
        mov_lang_db = load_vector_store(args=args,
                                        db_name=args.mov_lang_db_name,
                                        embedder=embedder)
        mov_lang_db = VectorIndexRetriever(index=mov_lang_db,
                                           similarity_top_k=10)

        return mov_lang_db
