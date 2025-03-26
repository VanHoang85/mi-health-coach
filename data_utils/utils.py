import re
import json
from collections import Counter


def save_data(path_to_data_file: str, data: dict):
    with open(path_to_data_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
        file.write("\n")


def load_data(path_to_data_file: str):
    with open(path_to_data_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


class MovState:
    def __init__(self):
        self.mov_states: dict[str: list[int]] = {}  # format: phase: [mov_level]
        self.map_to_soc = {-3: "pre-contemplation", -2: "pre-contemplation", -1: "contemplation",
                           0: "neutral",
                           1: "contemplation", 2: "preparation", 3: "preparation"}

    def add_mov_level(self, phase: str, mov_level: int):
        if phase not in self.mov_states:
            self.mov_states[phase] = []
        self.mov_states[phase].append(mov_level)

    def get_mov_levels(self, phase="") -> list:
        if len(phase) > 0:
            return self.mov_states[phase]

        all_levels = []
        for levels in self.mov_states.values():
            all_levels.extend(levels)
        return all_levels[:-5]  # return the last 5 turns

    def print_mov_states(self):
        states = {}
        for phase, levels in self.mov_states.items():
            levels = [str(lv) for lv in levels]
            states[phase] = ", ".join(levels)
        return states

    def get_user_stage_of_change(self) -> str:
        mov_levels = self.get_mov_levels()  # get mov levels of the last 5 turns
        level_counter = Counter([self.map_to_soc[level] for level in mov_levels])
        stages = list(level_counter.keys())

        if len(stages) == 0:
            return "contemplation"

        for stage in stages:
            if stage in ["pre-contemplation", "contemplation", "preparation"]:
                return stage

        return "contemplation"  # if only neutral, return contemplation


class Conversation:
    def __init__(self, args, phase: str):
        self.args = args
        self.exp_mode: str = args.exp_mode

        self.current_turn: int = 0
        self.current_dialog_phase: str = phase

        self.all_candidates: list[str] = []
        self.agent_actions: list[list[str]] = []
        self.conv_history: list[dict] = []
        self.is_terminated: bool = False

    def update_conv(self,
                    message: str,
                    role: str,
                    mov_lang_pred: dict = None,
                    prompt: str = "",
                    actions: list = None,
                    retrieved_clue: list = None,
                    candidate_ids: list = None):

        message = message.strip()
        utt_info = {
            "utterance": f"{role}: {message}" if len(message) > 0 else "",
            "phase": self.current_dialog_phase
        }

        if mov_lang_pred:
            utt_info["mov_level"] = mov_lang_pred["mov_level"]
            utt_info["mov_behaviour"] = mov_lang_pred["behaviour"]
            utt_info["attitude_pred"] = mov_lang_pred["attitude_pred"]
            utt_info["strength_pred"] = mov_lang_pred["strength_pred"]
            # utt_info["attitude_prompt"] = mov_lang_pred["attitude_prompt"]
            # utt_info["strength_prompt"] = mov_lang_pred["strength_prompt"]

        if len(prompt) > 0:
            utt_info["prompt"] = prompt

        if actions:
            utt_info["actions"] = ", ".join(actions)
            self.agent_actions.extend(actions)

        if candidate_ids:
            candidate_ids = " ; ".join(candidate_ids)
            utt_info["candidate_ids"] = candidate_ids
            self.all_candidates.append(candidate_ids)

        if retrieved_clue:
            utt_info["retrieved_clue"] = " ".join(retrieved_clue)

        key = f"{role}_{self.current_turn}"
        if len(self.conv_history) > 0 and key == list(self.conv_history[-1].keys())[0]:
            self.conv_history[-1] = {f"{role}_{self.current_turn}": utt_info}
        else:
            self.conv_history.append({f"{role}_{self.current_turn}": utt_info})

    def update_phase(self,
                     stage_of_change: str = "",
                     move_to_phase: str = ""):

        if len(move_to_phase) > 0:
            self.current_dialog_phase = move_to_phase

        elif len(stage_of_change) > 0 and self.current_dialog_phase != "concluding":
            if stage_of_change == "pre-contemplation":
                self.current_dialog_phase = "evoking"

            elif stage_of_change == "contemplation" and self.current_turn != self.args.start_planning:
                self.current_dialog_phase = "evoking"

            elif stage_of_change == "contemplation" and self.current_turn == self.args.start_planning:
                self.current_dialog_phase = "planning"

            elif stage_of_change == "preparation":
                self.current_dialog_phase = "planning"

            """
            if self.current_turn == self.args.start_planning:
                if stage_of_change == "pre-contemplation":
                    self.current_dialog_phase = "evoking"
                elif stage_of_change in ["contemplation", "preparation"]:
                    self.current_dialog_phase = "planning"

            elif self.current_turn == self.args.start_evoking:
                if stage_of_change in ["pre-contemplation", "contemplation"]:
                    self.current_dialog_phase = "evoking"
                elif stage_of_change == "preparation":
                    self.current_dialog_phase = "planning"
            `"""

    def update_generation_latency(self, latency: float):
        utt = self.conv_history[-1]
        for _, utt_info in utt.items():
            utt_info["gen_latency"] = latency

    def update_s2t_latency(self, latency: float):
        utt = self.conv_history[-1]
        for _, utt_info in utt.items():
            utt_info["s2t_latency"] = latency

    def update_t2s_latency(self, latency: float):
        utt = self.conv_history[-1]
        for _, utt_info in utt.items():
            utt_info["t2s_latency"] = latency

    def get_current_phase(self) -> str:
        return self.current_dialog_phase

    def get_previous_phase(self) -> str:
        prev_utt = self.conv_history[-2].values()
        return list(prev_utt)[0]["phase"]

    def get_num_turn_in_phase(self, phase: str) -> int:
        num = 0
        for turn in self.conv_history:
            for _, info in turn.items():
                if info["phase"] == phase:
                    num += 1
        return num

    def update_turn(self):
        self.current_turn += 1

    def get_current_turn(self) -> int:
        return self.current_turn

    def get_exp_mode(self) -> str:
        return self.exp_mode

    def get_prev_actions(self, num_prev_turns: int = 0) -> list:
        num_prev_turns = num_prev_turns if num_prev_turns != 0 else 1
        return self.agent_actions[-num_prev_turns:] if len(self.agent_actions) > 0 else []

    def get_prev_candidate(self) -> str:
        return self.all_candidates[-1] if len(self.all_candidates) > 0 else ""

    def get_conv_history(self, num_prev_utt: int = 0) -> str:
        num_prev_utt = self.args.max_num_prev_turns_summarise_act if num_prev_utt == 0 else num_prev_utt
        conv_history = self.conv_history[-num_prev_utt:]
        conv = []
        for utt_info in conv_history:
            conv.append(list(utt_info.values())[0]["utterance"])
        return "\n".join(conv)

    def get_conv_history_for_attitude_task(self, therapist_utt_setting: str) -> str:
        return self.get_conv_history(1) if therapist_utt_setting == "wo_therapist" else self.get_conv_history(2)

    def get_conv_history_for_strength_task(self, therapist_utt_setting: str) -> str:
        return self.get_conv_history(2) if therapist_utt_setting == "w_therapist" else self.get_conv_history(1)

    def check_phase_condition(self, message: str):
        if len(message) > 0:
            if "GOAL_DEFINED" in message:
                self.update_phase(move_to_phase="concluding")

            elif "PLAN_ASKED" in message:
                self.update_phase(move_to_phase="planning")

    def check_terminating_condition(self, message: str):
        if "END_CONV" in message or self.current_turn == self.args.terminating:
            self.is_terminated = True

        else:
            # if user wants to quit the session, change to final phase
            bye_pattern = r"\b(goodbye|bye(?:bye)?)\b"
            matches = re.findall(bye_pattern, message, re.IGNORECASE)
            if matches:
                self.update_phase(move_to_phase="concluding")
                self.is_terminated = True

    def save_conv_to_file(self, dir_path: str, file_name: str, client_info: dict):
        data = {
            "model_info": {
                "agent_model": self.args.agent_model,
                "retrieval_model": self.args.retrieval_model,
                "mov_detector_model": self.args.mov_detector_model,
                "rerank_type": self.args.rerank_type
            },
            "client_info": client_info,
            "conversation_history": self.conv_history
        }

        save_data(path_to_data_file=f"{dir_path}/{file_name}.json", data=data)
        self.save_conv_as_text(dir_path, file_name)

    def get_latest_conv(self) -> dict:
        utts = self.conv_history[-2]
        utts.update(self.conv_history[-1])
        return utts

    def save_conv_as_text(self, dir_path: str, file_name: str):
        dialogue = {}
        for utterance in self.conv_history:
            for key, value in utterance.items():
                role, turn = key.strip().split("_")

                if turn not in dialogue:
                    dialogue[turn] = {}

                info = [value["utterance"]]
                if role.lower() == "therapist" and "actions" in value:
                    info.append(value["actions"])

                dialogue[turn][role] = "\n".join(info)

        out = ""
        for turn, turn_info in dialogue.items():
            out += f"\n### Turn {turn}:"
            for utt in turn_info.values():
                out += f"\n{utt}"
            out += "\n"

        with open(f"{dir_path}/{file_name}.txt", "w", encoding="utf-8") as file:
            file.write(out)
