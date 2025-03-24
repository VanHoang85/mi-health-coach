from data_utils.utils import MovState, Conversation
from data_utils.chat_prompts import get_conv_history_template
from model_utils.agent import Agent
from model_utils.mov_detector import MovDetector


class UserAgent(Agent):
    def __init__(self,
                 args,
                 role: str,
                 is_human: bool,
                 user_id: str = "",
                 system_prompt: str = ""):
        super().__init__(args=args,
                         role=role,
                         system_prompt=system_prompt,
                         model_type=args.agent_model if not is_human else "human")

        self.user_id = user_id
        self.is_human = is_human
        self.mov_states = MovState()
        self.stage_of_change: str = ""  # pre-contemplation, contemplation, preparation

    def receive_and_response(self,
                             mov_detector: MovDetector,
                             conversation: Conversation,
                             user_message: str = "") -> dict:

        conversation.update_turn()
        print(f"\n### Current Turn: {conversation.current_turn}")

        if not self.is_human and len(user_message) == 0:
            conv_history = conversation.get_conv_history(self.args.max_num_prev_turns_summarise_act)
            user_prompt = f"{get_conv_history_template(conv_history)}\n{self.role}:"
            user_message = self.send_message(prompt=user_prompt,
                                             task="strategy",
                                             system_prompt=self.system_prompt).strip()

        conversation.update_conv(message=user_message,
                                 role=self.role)

        # get mov lang
        mov_lang_pred = None
        if conversation.get_exp_mode() == "MI":
            mov_lang_pred = mov_detector.detect_mov_level(conversation)
            self.mov_states.add_mov_level(phase=conversation.get_current_phase(),
                                          mov_level=mov_lang_pred["mov_level"])

            if conversation.get_current_turn() == self.args.start_focusing:
                conversation.update_phase(move_to_phase="focusing")
            elif conversation.get_current_turn() == self.args.start_concluding:
                conversation.update_phase(move_to_phase="concluding")

            # elif conversation.get_current_turn() in [self.args.start_evoking, self.args.start_planning]:
            # elif self.args.start_evoking <= conversation.get_current_turn() < self.args.start_concluding:
            # check client stage of change every 3 turns and update phase if needed: 8, 11, 14, 17
            elif (conversation.get_current_turn() in [self.args.start_evoking + 3 * n for n in range(0, 4)]
                  and conversation.get_current_phase() != "concluding"):
                self.stage_of_change = self.mov_states.get_user_stage_of_change()
                conversation.update_phase(stage_of_change=self.stage_of_change)

        conversation.update_conv(message=user_message,
                                 role=self.role,
                                 mov_lang_pred=mov_lang_pred)
        conversation.check_terminating_condition(user_message)

        # print(f"{self.role}: {user_message}")
        if mov_lang_pred:
            print(f"Motivation: {mov_lang_pred['behaviour']}")
        return {
            "mov_lang_level": mov_lang_pred["mov_level"] if mov_lang_pred else "",
            "mov_lang_behaviour": mov_lang_pred["behaviour"] if mov_lang_pred else "",
            "stage_of_change": self.stage_of_change,
            "client_message": user_message
        }

    def get_client_info(self):
        return {
            "user_id": self.user_id,
            "experiment_mode": self.args.exp_mode,
            "stage_of_change": self.stage_of_change,
            "mov_langs": self.mov_states.print_mov_states()
        }
