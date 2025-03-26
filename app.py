import os
import time
import json

import huggingface_hub
import torch
import argparse
import gradio

from pathlib import Path
from huggingface_hub import CommitScheduler

from data_utils.utils import Conversation
from model_utils.mov_detector import MovDetector
from model_utils.dialog_manager import MIDialogManager
from model_utils.coach_agent import CoachAgent
from model_utils.user_agent import UserAgent


def save_conv_json(data_to_save: dict, filename: str) -> None:
    CHAT_FILE_PATH = OUTPUT_CHAT_DIR / f"{filename}.json"
    with scheduler.lock:
        with CHAT_FILE_PATH.open("a") as file:
            json.dump(data_to_save, file, indent=4)


def remove_stop_phases(message) -> str:
    message = message.replace("[END_CONV]", "")
    message = message.replace("[GOAL_DEFINED]", "")
    message = message.replace("[PLAN_ASKED]", "")
    return message


def interaction(user_message: str, history: list):
    if len(history) == 0:
        user_id = user_message.split()[-1]
        client_agent = UserAgent(args,  # Creating client agent
                                 role="Client",
                                 user_id=user_id,
                                 is_human=True)
        clients[user_id] = client_agent

        phase = "engaging" if args.exp_mode == "MI" else args.exp_mode
        conversation = Conversation(args=args, phase=phase)
        conversations[user_id] = conversation

    else:
        user_id = history[0]["content"].split()[-1]
        client_agent = clients[user_id]
        conversation = conversations[user_id]

    current_turn = conversation.get_current_turn()
    if current_turn >= 1:
        # add coach message to conversation object
        prev_coach_message = history[-1]["content"]
        if len(conversation.conv_history[-1][f"Therapist_{current_turn}"]["utterance"]) == 0:
            conversation.conv_history[-1][f"Therapist_{current_turn}"]["utterance"] = f"{coach_agent.role}: {prev_coach_message}"

        if current_turn >= args.start_planning:
            conversation.check_phase_condition(prev_coach_message)
        if current_turn >= args.start_focusing:
            conversation.check_terminating_condition(prev_coach_message)

        print(f"{coach_agent.role}: {remove_stop_phases(prev_coach_message)}")
        try:
            print(f"Actions: {conversation.conv_history[-1][f'Therapist_{current_turn}']['actions']}")
        except KeyError:
            pass

        # conversation.save_conv_to_file(dir_path=args.human_chat_dir,
        #                                file_name=f"{args.exp_mode}_{client_agent.user_id}",
        #                                client_info=client_agent.get_client_info())

        # save conv
        conv_to_save = conversation.get_latest_conv()
        save_conv_json(data_to_save=conv_to_save, filename=f"{args.exp_mode}_{client_agent.user_id}")

    # Start conversing...
    if conversation.is_terminated:
        coach_message = "The session has ended. Please close the window."

    else:
        start = time.time()
        client_response = client_agent.receive_and_response(mov_detector=mov_detector,
                                                            conversation=conversation,
                                                            user_message=user_message)
        print(f"{client_agent.role}: {user_message}")

        coach_message = coach_agent.receive_and_response(mov_lang_level=client_response["mov_lang_level"],
                                                         mov_lang_behaviour=client_response["mov_lang_behaviour"],
                                                         stage_of_change=client_response["stage_of_change"],
                                                         conversation=conversation,
                                                         # stream=True
                                                         )

        latency = (time.time() - start) / 60  # as minutes
        conversation.update_generation_latency(latency)

    # for token_idx in range(len(coach_message)):
    #     yield coach_message[: token_idx + 1]
    partial_message = ""
    if isinstance(coach_message, str):
        for token_idx in range(len(coach_message)):
            partial_message += coach_message[token_idx]
            time.sleep(0.015)
            yield partial_message

    else:  # do streaming
        for chunk in coach_message:
            content = chunk.choices[0].delta.content  # extract text from streamed litellm chunks
            if content:
                partial_message += content
                time.sleep(0.015)
                yield partial_message


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_mode', type=str, default='MI',
                        choices=["MI", "auto-MI", "non-MI"])
    parser.add_argument('--agent_model', type=str, default="gpt-4o-mini",  # main model: "gpt-4o"
                        choices=["gpt-4o", "gpt-4o-mini"])
    parser.add_argument('--retrieval_model', type=str, default='cohere',
                        choices=["sbert", "cohere"])
    parser.add_argument('--mov_detector_model', type=str, default="gpt-4o-mini")

    parser.add_argument('--data_dir', type=str, default='./data/dataset')
    parser.add_argument('--use_test_data', type=bool, default=True)
    parser.add_argument('--mov_lang_data', type=str, default='mov_lang_detect/train_client.json')
    parser.add_argument('--mov_lang_test_data', type=str, default='mov_lang_detect/annomi_test_client.json')
    parser.add_argument('--strategy_data', type=str, default='therapist_strategy/train_therapist.json')
    parser.add_argument('--strategy_test_data', type=str,
                        default='therapist_strategy/annomi_test_therapist.json')
    parser.add_argument('--human_chat_dir', type=str, default='./outputs/human_chats')
    parser.add_argument('--storage_dir', type=str, default='./chroma_storage')
    parser.add_argument('--mov_lang_db_name', type=str, default='mov_lang_db')
    parser.add_argument('--strategy_db_name', type=str, default='diag_strategy_db')
    parser.add_argument('--input_type', type=str, default="search_document",
                        help='Input type to use for searching in-context samples for motivational level detection.')
    parser.add_argument('--max_num_prev_turns', type=int, default=5,
                        help='The number of previous turns in the dialogue to use as context for embedding and prompting')
    parser.add_argument('--max_num_prev_turns_other_act', type=int, default=10,
                        help='The number of previous turns in the dialogue when the actions is auto or other')
    parser.add_argument('--max_num_prev_turns_summarise_act', type=int, default=20,
                        help='The number of previous turns in the dialogue when the actions is auto or other')
    parser.add_argument('--num_in_context_samples', type=int, default=3,
                        help='The number of in-context samples to use with few-shot ICL for detecting motivational language.')
    parser.add_argument('--therapist_utt_setting', type=str, default="default",
                        choices=["default", "w_therapist", "wo_therapist"])
    parser.add_argument('--use_retrieved_clues', type=bool, default=True)
    parser.add_argument('--num_dialog_retrieval', type=int, default=20,
                        help='The number of dialogues retrieved.')
    parser.add_argument('--rerank_type', type=str, default="mov_behaviour",
                        choices=["all", "summary", "mov_behaviour", "none"])

    parser.add_argument('--start_focusing', type=int, default=3)
    parser.add_argument('--start_evoking', type=int, default=8)
    parser.add_argument('--start_planning', type=int, default=14)
    parser.add_argument('--start_concluding', type=int, default=20)
    parser.add_argument('--terminating', type=int, default=23)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--num_retries', type=int, default=2)
    parser.add_argument('--request_timeout', type=int, default=180)  # in seconds
    parser.add_argument('--stream_timeout', type=int, default=5)
    parser.add_argument('--fallback_models', default=["gpt-4o-mini"])
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--max_new_token', type=int, default=128)
    parser.add_argument('--return_full_text', action='store_true')
    parser.add_argument('--load_in_8_bit', action='store_true')
    parser.add_argument('--cpu_loading', action='store_true')
    args = parser.parse_args()

    # set name for databases
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.strategy_db_name = f"{args.strategy_db_name}_{args.retrieval_model}"
    args.mov_lang_db_name = f"{args.mov_lang_db_name}_{args.retrieval_model}"
    args.mov_lang_db_name = f"{args.mov_lang_db_name}" if args.therapist_utt_setting == "default" \
        else f"{args.mov_lang_db_name}_{args.therapist_utt_setting}"

    # get keys
    args.cohere_key = os.getenv("COHERE_KEY")
    args.openai_key = os.getenv("OPENAI_KEY")
    args.groq_key = os.getenv("GROQ_KEY")
    args.hf_auth = os.getenv("HF_AUTH")
    huggingface_hub.login(token=args.hf_auth)

    # set output folder path
    args.human_chat_dir = f"{args.human_chat_dir}/{args.agent_model}"
    OUTPUT_CHAT_DIR = Path(args.human_chat_dir)
    OUTPUT_CHAT_DIR.mkdir(parents=True, exist_ok=True)

    scheduler = CommitScheduler(
        repo_id="ai-health-coach-chats",
        repo_type="dataset",
        folder_path=OUTPUT_CHAT_DIR,
        path_in_repo="data"
    )

    conversations: dict[str, Conversation] = {}
    clients: dict[str, UserAgent] = {}

    mov_detector = MovDetector(args=args)
    dialog_manager = MIDialogManager(args=args) if args.exp_mode == "MI" else None

    # Creating coaching agent
    coach_agent = CoachAgent(args,
                             role="Therapist",
                             dialog_manager=dialog_manager)
    welcome_message = ("<strong>You will converse with a coaching chatbot on the topic of physical activity."
                       "<br>If you wish to end the chat at anytime, just type \"bye\"."
                       "<br>The session will last for 15-25 turns."
                       "<br><br>Please type in your nickname to start the session...</strong>")

    demo = gradio.ChatInterface(interaction,
                                chatbot=gradio.Chatbot(
                                    placeholder=welcome_message,
                                    type="messages",
                                    avatar_images=tuple((None, "./data/robot_avatar_head.png"))),
                                stop_btn=False,
                                # description="You will converse with a coaching chatbot on the topic of physical activity. The session will last for 15-25 turns. If you wish to end the chat at any time, just type \"bye\".",
                                title="Physical Activity CoachBot",
                                type="messages",
                                theme=gradio.themes.Citrus(text_size="lg"))
    demo.launch()
