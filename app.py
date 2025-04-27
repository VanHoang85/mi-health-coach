import os
import io
import time
import json
import wave

import argparse
import gradio
import torch

from pathlib import Path
import huggingface_hub
from huggingface_hub import CommitScheduler

from typing import IO
from groq import Groq
from openai import OpenAI
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

from data_utils.utils import Conversation
from model_utils.mov_detector import MovDetector
from model_utils.dialog_manager import MIDialogManager
from model_utils.coach_agent import CoachAgent
from model_utils.user_agent import UserAgent


def save_conv_json(data_to_save: dict, filename: str) -> None:
    SPEAK_FILE_PATH = OUTPUT_SPEAK_DIR / f"{filename}.json"
    with scheduler.lock:
        with SPEAK_FILE_PATH.open("a") as file:
            json.dump(data_to_save, file, indent=4)


def save_audio_wav(audio_buffer: io.BytesIO, filename: str) -> None:
    SPEAK_FILE_PATH = f"{args.human_speak_dir}/{filename}.wav"
    with scheduler.lock:
        with wave.open(SPEAK_FILE_PATH, mode="wb") as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # sample width in bytes
            wav_file.setframerate(44100)  # sample rate
            wav_file.writeframes(audio_buffer.getvalue())


def save_audio_file(audio_buffer, filename: str) -> None:
    SPEAK_FILE_PATH = f"{args.human_speak_dir}/{filename}.wav"
    with scheduler.lock:
        with open(SPEAK_FILE_PATH, "wb") as file:
            for chunk in audio_buffer.getvalue():
                if chunk:
                    file.write(chunk)


def remove_stop_phases(message) -> str:
    message = message.replace("[END_CONV]", "")
    message = message.replace("[GOAL_DEFINED]", "")
    message = message.replace("[PLAN_ASKED]", "")
    return message


def speech_to_text(audio_filename) -> str:
    with open(audio_filename, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=file,  # Required audio file in .wav
            model="distil-whisper-large-v3-en",  # Required model to use for transcription
            # prompt="Specify context or spelling",  # Optional
            response_format="text",  # Optional
            language="en",  # Optional
            temperature=0.0  # Optional
        )
    # print(transcription)
    return transcription.strip()


def text_to_speech(text: str) -> IO[bytes]:
    audio_response = eleven_lab_client.text_to_speech.convert(
        voice_id="iP95p4xoKVk53GoZ742B",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0
        )
    )

    # Create a BytesIO object to hold the audio data in memory
    audio_stream = io.BytesIO()

    # Write each chunk of audio data to the stream
    for chunk in audio_response:
        if chunk:
            audio_stream.write(chunk)

    # Reset stream position to the beginning
    audio_stream.seek(0)

    return audio_stream


def spoken_interaction(user_response, history: list):

    start = time.time()
    user_message = speech_to_text(user_response)
    latency = (time.time() - start) / 60  # as minutes

    if len(history) == 0:
        user_id = "_".join(user_message.split()[-3])
        client_agent = UserAgent(args,
                                 role="Client",
                                 user_id=user_id,  # use the 1st user message, aka name
                                 is_human=True)
        clients[user_id] = client_agent

        phase = "engaging" if args.exp_mode == "MI" else args.exp_mode
        conversation = Conversation(args=args, phase=phase)
        conversations[user_id] = conversation

    else:
        user_id = "_".join(history[0]["content"].split()[-3])
        client_agent = clients[user_id]
        conversation = conversations[user_id]

    if conversation.is_terminated:
        # coach_message = "The session has ended. Please close the window."
        # coach_response = open("./data/media/end_chris.mp3", "rb")
        coach_response = "./data/end_chris.mp3"
        return coach_response, history

    client_response = client_agent.receive_and_response(mov_detector=mov_detector,
                                                        conversation=conversation,
                                                        user_message=user_message)
    print(f"{client_agent.role}: {user_message}")
    conversation.update_s2t_latency(latency=round(latency, 3))

    start = time.time()  # start generation latency
    coach_message = coach_agent.receive_and_response(mov_lang_level=client_response["mov_lang_level"],
                                                     mov_lang_behaviour=client_response["mov_lang_behaviour"],
                                                     stage_of_change=client_response["stage_of_change"],
                                                     conversation=conversation)
    latency = (time.time() - start) / 60  # as minutes
    conversation.update_generation_latency(latency=round(latency, 3))

    current_turn = conversation.get_current_turn()
    if current_turn >= args.start_planning:
        conversation.check_phase_condition(coach_message)
    if current_turn >= args.start_focusing:
        conversation.check_terminating_condition(coach_message)

    # add coach message to conversation object
    if len(conversation.conv_history[-1][f"Therapist_{current_turn}"]["utterance"]) == 0:
        conversation.conv_history[-1][f"Therapist_{current_turn}"][
            "utterance"] = f"{coach_agent.role}: {coach_message}"

    print(f"{coach_agent.role}: {remove_stop_phases(coach_message)}")
    try:
        print(f"Actions: {conversation.conv_history[-1][f'Therapist_{current_turn}']['actions']}")
    except KeyError:
        pass

    start = time.time()  # start t2s latency
    coach_response = text_to_speech(remove_stop_phases(coach_message))

    output_buffer = b""
    for audio_bytes, message in zip(coach_response, coach_message):
        output_buffer += audio_bytes
        yield audio_bytes, message

    latency = (time.time() - start) / 60  # as minutes
    conversation.update_t2s_latency(latency=round(latency, 3))

    # save conversation messages
    conv_to_save = conversation.get_latest_conv()
    save_conv_json(data_to_save=conv_to_save, filename=f"{conversation.exp_mode}_{client_agent.user_id}")

    user_audio_filename = f"{conversation.exp_mode}_{client_agent.user_id}_user_{conversation.current_turn}.wav"
    save_audio_file(user_response, user_audio_filename)  # save the user speaking

    history.append(
        gradio.ChatMessage(role="user",
                           content=user_message)
    )
    history.append(
        gradio.ChatMessage(role="assistant",
                           content=remove_stop_phases(coach_message))
    )
    return coach_response, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_mode', type=str, default="MI",
                        choices=["MI", "non-MI"])
    parser.add_argument('--agent_model', type=str, default="gpt-4o",  # main model: "gpt-4o"
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

    parser.add_argument('--human_speak_dir', type=str, default='speech_data')
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
    parser.add_argument('--request_timeout', type=int, default=300)  # in seconds
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
    args.eleven_lab_key = os.getenv("ELEVEN_LABS_KEY")
    huggingface_hub.login(token=args.hf_auth)

    # create speech-to-text & text-to-speech endpoints
    openai_client = OpenAI(api_key=args.openai_key)
    groq_client = Groq(api_key=args.groq_key)
    eleven_lab_client = ElevenLabs(api_key=args.eleven_lab_key)

    # set output folder path
    OUTPUT_SPEAK_DIR = Path(args.human_speak_dir)
    OUTPUT_SPEAK_DIR.mkdir(parents=True, exist_ok=True)

    scheduler = CommitScheduler(
        repo_id="ai-health-coach-chats",
        repo_type="dataset",
        folder_path=OUTPUT_SPEAK_DIR,
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
    welcome_message = "<strong>Press \"Record\" and say your nickname to start the session...</strong>"

    # alert(e.code);
    js = """
        <script>
        function shortcuts(e) {
        if (e.key == " " || e.code == "Space" || e.keyCode === 32) {
            Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Record') || button.textContent.includes('Stop')).click();
        }
        else {
            }
        }
        document.addEventListener('keydown', shortcuts, false);
        </script>
        """

    title = (
        """
        <center> 
        <h1> Talk with Jordan, the Physical Activity CoachBot </h1>
        <br> The session will last for 22 turns maximum. If you want to end the talk earlier, just say \"bye\". 
        <br> Please go back to the survey after that.
        </center>
        """
    )

    with gradio.Blocks(
            head=js,
            # theme=gradio.themes.Citrus(text_size="lg")
    ) as demo:
        with gradio.Row():
            gradio.HTML(title)

        with gradio.Row():
            with gradio.Column():
                input_audio = gradio.Audio(label="Input Audio", sources=["microphone"], type="filepath", interactive=True)
            with gradio.Column():
                chatbot = gradio.Chatbot(
                    label="Conversation",
                    type="messages",
                    placeholder=welcome_message,
                    avatar_images=tuple((None, "./data/robot_avatar_head.png"))
                )
                output_audio = gradio.Audio(label="Output Audio", streaming=True, autoplay=True, visible=False)

        input_audio.stop_recording(
            spoken_interaction,
            [input_audio, chatbot],
            [output_audio, chatbot]  # [input_audio, output_audio, chatbot]
        )

        # cancel = gradio.Button("Stop Conversation", variant="stop")
        # cancel.click()

    demo.launch()
