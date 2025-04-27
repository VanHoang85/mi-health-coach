import os
import io
import time
import json
import wave

import numpy as np
import argparse
import gradio
import torch

from pathlib import Path
import huggingface_hub
from huggingface_hub import CommitScheduler

import tempfile
from typing import IO
from dataclasses import field
# from model_utils.audio_utils import run_vad
from audio_utils.vad import run_vad
from pydub import AudioSegment

from groq import Groq
from openai import OpenAI
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

from data_utils.utils import Conversation
from model_utils.mov_detector import MovDetector
from model_utils.dialog_manager import MIDialogManager
from model_utils.coach_agent import CoachAgent
from model_utils.user_agent import UserAgent


class AppState:
    stream: np.ndarray | None = None
    sampling_rate: int = 0
    pause_detected: bool = False
    started_talking: bool = False
    stopped: bool = False
    history: list = []


def determine_pause(audio: np.ndarray, sampling_rate: int, state: AppState) -> bool:
    """
    Take in the stream, determine if a pause happened
    Source: https://huggingface.co/spaces/gradio/omni-mini/blob/eb027808c7bfe5179b46d9352e3fa1813a45f7c3/app.py#L98
    """

    temp_audio = audio

    dur_vad, _, time_vad = run_vad(temp_audio, sampling_rate)
    duration = len(audio) / sampling_rate

    if dur_vad > 0.5 and not state.started_talking:
        print("started talking")
        state.started_talking = True
        return False

    print(f"duration_after_vad: {dur_vad:.3f} s, time_vad: {time_vad:.3f} s")

    return (duration - dur_vad) > 1


def start_recording_user(state: AppState):
    if not state.stopped:
        return gradio.Audio(recording=True)


def process_audio(audio: tuple, state: AppState):
    if state.stream is None:
        state.stream = audio[1]
        state.sampling_rate = audio[0]
    else:
        state.stream = np.concatenate((state.stream, audio[1]))

    pause_detected = determine_pause(state.stream, state.sampling_rate, state)
    state.pause_detected = pause_detected

    if state.pause_detected and state.started_talking:
        return gradio.Audio(recording=False), state
    return None, state


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


def warm_up():
    frames = b"\x00\x00" * 1024 * 2  # 1024 frames of 2 bytes each
    dur, frames, tcost = run_vad(frames, 16000)
    print(f"warm up done, time_cost: {tcost:.3f} s")


warm_up()


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


def spoken_interaction(state: AppState):
    if not state.pause_detected and not state.started_talking:
        return None, AppState()

    audio_buffer = io.BytesIO()

    # convert to wav
    segment = AudioSegment(
        state.stream.tobytes(),
        frame_rate=state.sampling_rate,
        sample_width=state.stream.dtype.itemsize,
        channels=(1 if len(state.stream.shape) == 1 else state.stream.shape[1]),
    )
    segment.export(audio_buffer, format="wav")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_buffer.getvalue())

    start = time.time()
    user_message = speech_to_text(f.name)
    latency = (time.time() - start) / 60  # as minutes

    if len(state.history) == 0:
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
        user_id = "_".join(state.history[0]["content"].split()[-3])
        client_agent = clients[user_id]
        conversation = conversations[user_id]

    if conversation.is_terminated:
        # coach_response = "./data/end_chris.mp3"
        # return None, coach_response, state
        yield None, AppState(history=state.history)

    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_buffer.getvalue())
    """

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
    for audio_bytes in coach_response:
        output_buffer += audio_bytes
        yield audio_bytes, state

    """
    for mp3_bytes in speaking(audio_buffer.getvalue()):
        output_buffer += mp3_bytes
        yield mp3_bytes, state
    """

    latency = (time.time() - start) / 60  # as minutes
    conversation.update_t2s_latency(latency=round(latency, 3))

    # save conversation messages
    conv_to_save = conversation.get_latest_conv()
    save_conv_json(data_to_save=conv_to_save, filename=f"{conversation.exp_mode}_{client_agent.user_id}")

    user_audio_filename = f"{args.exp_mode}_{client_agent.user_id}_user_{conversation.current_turn}.wav"
    save_audio_file(audio_buffer, user_audio_filename)  # save the user speaking

    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(output_buffer)

    coach_audio_filename = f"{args.exp_mode}_{client_agent.user_id}_coach_{conversation.current_turn}.mp3"
    save_audio_file(output_buffer, coach_audio_filename)  # save the coach speaking
    
    state.history.append({"role": "user",
                          "content": {
                              # "text": user_message,
                                      "path": f.name,
                                      "mime_type": "audio/wav"}})
    
    state.history.append({"role": "assistant",
                          "content": {
                              # "text": remove_stop_phases(coach_message),
                                      "path": f.name,
                                      "mime_type": "audio/mp3"}})
    """

    state.history.append({"role": "user",
                          "content": user_message})

    state.history.append({"role": "assistant",
                          "content": remove_stop_phases(coach_message)})

    yield None, AppState(history=state.history)


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
    welcome_message = ("<strong>You will converse with Jordan, an AI coach, on the topic of physical activity."
                       "<br>The session will last for a maximum of 22 turns."
                       # "<br>If you wish to end the chat at anytime, just say \"bye\"."
                       "<br><br>Press \"Record\" and say your nickname to start the session...</strong>")
    description = "The session will last for 22 turns maximum. If you want to end the chat earlier, just type \"bye\". Please go back to the survey after that."

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

    """
    with gradio.Blocks(
            head=js,
            title="Chat with Jordan, the Physical Activity CoachBot",
            description="The session will last for 22 turns maximum. If you want to end the chat earlier, just type \"bye\". Please go back to the survey after that.",
            theme=gradio.themes.Citrus(text_size="lg")) as demo:
        chatbot = gradio.Chatbot(
            placeholder=welcome_message,
            type="messages",
            avatar_images=tuple((None, "./data/robot_avatar_head.png"))
        )
        audio_input = gradio.Audio(sources=["microphone"], type="filepath", interactive=True)
        audio_output = gradio.Audio(autoplay=True, visible=False)  # , streaming=True
        audio_input.stop_recording(interaction, [audio_input, chatbot], [audio_input, audio_output, chatbot])
    """

    title = (
        """
        <center> 
        <h1> Talk with Jordan, the Physical Activity CoachBot </h1>
        </center>
        """
    )

    with gradio.Blocks(
            # theme=gradio.themes.Citrus(text_size="lg")
    ) as demo:
        with gradio.Row():
            gradio.HTML(title)

        with gradio.Row():
            with gradio.Column():
                input_audio = gradio.Audio(label="Input Audio", sources=["microphone"], type="numpy")
            with gradio.Column():
                chatbot = gradio.Chatbot(
                    label="Conversation",
                    type="messages",
                    placeholder=welcome_message,
                    # avatar_images=tuple((None, "./data/robot_avatar_head.png"))
                )
                output_audio = gradio.Audio(label="Output Audio", streaming=True, autoplay=True)

        state = gradio.State(value=AppState())

        stream = input_audio.stream(
            process_audio,
            [input_audio, state],
            [input_audio, state],
            stream_every=0.5,
            time_limit=30,
        )
        respond = input_audio.stop_recording(
            spoken_interaction,
            [state],
            [output_audio, state]
        )
        respond.then(lambda s: s.history, [state], [chatbot])

        restart = output_audio.stop(
            start_recording_user,
            [state],
            [input_audio]
        )
        cancel = gradio.Button("Stop Conversation", variant="stop")
        cancel.click(lambda: (AppState(stopped=True), gradio.Audio(recording=False)), None,
                     [state, input_audio], cancels=[respond, restart])

    demo.launch()
