from model_utils.agent import Agent
from model_utils.dialog_manager import MIDialogManager
from data_utils.utils import Conversation
from data_utils.chat_prompts import (
    get_chat_template,
    get_phase_system_prompt,
    get_conv_history_template
)


class CoachAgent(Agent):
    def __init__(self,
                 args,
                 role: str,
                 system_prompt="",
                 dialog_manager: MIDialogManager = None):
        super().__init__(args=args,
                         role=role,
                         system_prompt=system_prompt,
                         model_type=args.agent_model)
        self.dialog_manager = dialog_manager

    def receive_and_response(self,
                             mov_lang_level: int,
                             mov_lang_behaviour: str,
                             stage_of_change: str,
                             conversation: Conversation,
                             stream: bool = False):

        if conversation.get_exp_mode() == "MI":
            return self.generate_mi_response(conversation=conversation,
                                             user_mov_level=mov_lang_level,
                                             user_mov_behaviour=mov_lang_behaviour,
                                             stage_of_change=stage_of_change,
                                             stream=stream)
        else:
            return self.generate_non_auto_mi_response(conversation=conversation,
                                                      stream=stream)

    def generate_mi_response(self,
                             conversation: Conversation,
                             user_mov_level: int,
                             user_mov_behaviour: str,
                             stage_of_change: str,
                             stream: bool = False):

        # first predict the next actions
        predictions = self.dialog_manager.predict_mi_actions(conversation=conversation,
                                                             user_mov_level=user_mov_level,
                                                             mov_lang_behaviour=user_mov_behaviour)

        actions_prompt = get_chat_template(exp_mode=conversation.get_exp_mode(),
                                           therapist_utts=predictions["therapist_texts"],
                                           actions=predictions["actions"],
                                           action_descs=predictions["actions_desc"],
                                           use_clue=self.args.use_retrieved_clues)

        num_turns = conversation.get_num_turn_in_phase(conversation.get_current_phase())
        add_keyword = True if num_turns >= 6 else False
        phase_system = get_phase_system_prompt(phase=conversation.get_current_phase(),
                                               stage=stage_of_change,
                                               add_keyword=add_keyword)
        self.update_system_prompt(phase_system)

        if "summarise" in predictions["actions"] or conversation.get_current_turn() == self.args.terminating:
            conv_history = conversation.get_conv_history(self.args.max_num_prev_turns_summarise_act)
        elif len(predictions["actions"]) == 0:  # if other
            conv_history = conversation.get_conv_history(self.args.max_num_prev_turns_other_act)
        else:
            conv_history = conversation.get_conv_history(self.args.max_num_prev_turns)

        prompt = f"{actions_prompt}\n\n{get_conv_history_template(conv_history)}\nTherapist:"
        # agent_message = self.send_message_to_llm(prompt=prompt,
        #                                          system_prompt=system_prompt)
        agent_message = self.send_message(prompt=prompt,
                                          system_prompt=self.system_prompt,
                                          task="strategy",
                                          stream=stream)

        conversation.update_conv(message="" if stream else agent_message,
                                 role=self.role,
                                 prompt=f"{self.system_prompt}\n\n{prompt}",
                                 actions=predictions["actions"],
                                 retrieved_clue=predictions["therapist_texts"],
                                 candidate_ids=predictions["candidate_ids"])
        return agent_message

    def generate_non_auto_mi_response(self,
                                      conversation: Conversation,
                                      stream: bool = False):
        self.update_system_prompt(get_phase_system_prompt(phase=conversation.get_current_phase()))

        if conversation.get_current_turn() >= self.args.start_concluding:
            conversation.update_phase(move_to_phase="concluding")
            predictions = self.dialog_manager.predict_mi_actions(conversation=conversation,
                                                                 user_mov_level=0,
                                                                 mov_lang_behaviour="")

            action_prompt = get_chat_template(exp_mode="MI",
                                              therapist_utts=predictions["therapist_texts"],
                                              actions=predictions["actions"],
                                              action_descs=predictions["actions_desc"],
                                              use_clue=self.args.use_retrieved_clues)
        else:
            action_prompt = get_chat_template(exp_mode=conversation.get_exp_mode())

        conv_history = conversation.get_conv_history(self.args.max_num_prev_turns_other_act)
        prompt = f"{action_prompt}\n\n{get_conv_history_template(conv_history)}\nTherapist:"
        agent_message = self.send_message(prompt=prompt,
                                          task="strategy",
                                          system_prompt=self.system_prompt,
                                          stream=stream)

        conversation.update_conv(message="" if stream else agent_message,
                                 role=self.role)
        return agent_message
