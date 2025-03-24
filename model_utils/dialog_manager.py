import random
import numpy as np
from typing import Optional
from cohere.core.api_error import ApiError

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

from sentence_transformers import util
from vector_db import load_vector_store, create_embedder
from model_utils.agent import Agent

from data_utils.utils import Conversation
from data_utils.mov_detect_prompts import get_summary_prompt
from data_utils.chat_prompts import (
    get_strategies,
    get_strategy_description,
    NOT_TO_REPEAT_STRATEGIES
)


class MIDialogManager:
    def __init__(self, args):
        self.args = args
        self.summarizer = Agent(args, model_type=args.summarizer_model)
        self.embedder = create_embedder(args.input_type, args.cohere_key, args.retrieval_model)
        self.dialog_retriever: VectorIndexRetriever = self.load_vector_db(args, self.embedder)

    def predict_mi_actions(self,
                           conversation: Conversation,
                           user_mov_level: int,
                           mov_lang_behaviour: str) -> dict:

        # If in phase 3 and users show resistance, use rapport building strategies instead
        # user.get_recent_mov_level() <= 1
        # phase = "focusing" if current_dialog_phase == "planning" and user_mov_level <= -2 else current_dialog_phase

        phase = conversation.get_current_phase()

        candidates, candidate_ids = [], []
        required_actions, actions, actions_descs, therapist_texts = [], [], [], []

        if conversation.get_current_turn() == self.args.terminating or conversation.is_terminated:
            required_actions = ["terminate"]

        elif phase == "engaging":
            required_actions = ["introduce"] if conversation.get_current_turn() == 1 else ["question-open-focusing"]

        elif phase == "concluding":
            required_actions = ["summarise"] if conversation.get_previous_phase() != "concluding" else ["persuade"]

        elif (phase == "focusing" and conversation.get_current_turn() == 7
              and "question-ruler" not in conversation.get_prev_actions()):
            required_actions = ["question-ruler"]

        elif phase in ["focusing", "evoking", "planning"]:
            phase_strategies = get_strategies(phase=phase)
            conv_history = conversation.get_conv_history(self.args.max_num_prev_turns)

            candidates = self.retrieve_best_strategies(dialogue=conv_history,
                                                       conversation=conversation,
                                                       phase_strategies=phase_strategies,
                                                       mov_lang_behaviour=mov_lang_behaviour)

            required_actions = []
            if len(candidates) > 0 and candidates[0]:
                required_actions = list(candidates[0]["strategies"].keys())  # use the top ranked candidate

            # if no action found from retrieving similar dialogues, pick a random action from the list
            if len(required_actions) == 0:
                required_actions.append(random.choice(list(phase_strategies)))

            # add questions in focusing phase if not already
            if phase == "focusing":
                if len(set(required_actions).intersection({"question-open-focusing", "question-closed-focusing"})) == 0:
                    required_actions.append("question-open-focusing")

            # if previously ask for collaboration, then should give info in this turn
            if phase == "evoking":
                if "seek-collaboration" in conversation.get_prev_actions(1) and "give-information" not in required_actions:
                    required_actions.append("give-information")

            if phase == "planning":
                if "seek-collaboration" in conversation.get_prev_actions(1) and len(set(required_actions).intersection({"give-information", "persuade"})) == 0:
                    if user_mov_level >= 2:
                        required_actions.append("persuade")
                    else:
                        required_actions.append("give-information")

            # if either affirm or autonomy and high mov, add qs
            if user_mov_level >= 0 and set(required_actions).issubset({"affirm", "emphasize-autonomy"}):
                required_actions.append("question-open-evoking")  # b/c already add question-open-focusing during focusing phase

            # if low mov and no supporting strategy, add reflection
            supporting = ["affirm", "emphasize-autonomy", "reflection-simple", "reflection-complex"]
            if user_mov_level < 0 and len(set(required_actions).intersection(set(supporting))) == 0:
                required_actions.append(random.choice(["reflection-simple", "reflection-complex"]))

            # if retrieving also therapist utterances as clues in the prompts
            if self.args.use_retrieved_clues:
                for action in required_actions:
                    if action == "other":
                        continue

                    for candidate in candidates:
                        if (action in candidate["strategies"] and
                                action not in ["reflection-simple", "reflection-complex", "question-closed-focusing", "question-closed-evoking"]):
                            retrieved_clue = candidate["strategies"][action]
                            name, desc = get_strategy_description(action, phase=phase)

                            actions.append(action)
                            therapist_texts.append(retrieved_clue)
                            candidate_ids.append(f"{candidate['node_id']}--{action}")
                            actions_descs.append(f"{name.capitalize()}: {desc}\n")
                            break

        for action in required_actions:
            if action not in actions:
                actions.append(action)
                therapist_texts.append("")

                phase = phase if action != "terminate" else "concluding"
                name, desc = get_strategy_description(action, phase=phase)
                actions_descs.append(f"{name.capitalize()}: {desc}\n")

        return {"actions": actions,
                "actions_desc": actions_descs,
                "therapist_texts": therapist_texts,
                "candidate_ids": candidate_ids,
                "all_candidates": candidates}

    def retrieve_best_strategies(self,
                                 dialogue: str,
                                 conversation: Conversation,
                                 phase_strategies: list,
                                 mov_lang_behaviour: str) -> list[dict]:
        candidates = None
        while not candidates:
            try:
                candidates = self.dialog_retriever.retrieve(dialogue)
            except Exception or ApiError:
                print("Retry retrieving...")

        candidates = self.rerank_nodes(dialogue=dialogue,
                                       candidates=candidates,
                                       mov_lang_behaviour=mov_lang_behaviour,
                                       rank_type=self.args.rerank_type,
                                       embedder=self.embedder,
                                       summarizer=self.summarizer)

        retrieved_info = []
        for candidate in candidates:
            if candidate.node_id == conversation.get_prev_candidate():
                continue

            strategies = candidate.metadata["strategies"].split(', ')
            common = set(strategies).intersection(set(phase_strategies))
            if len(common) == 0:
                continue

            # do not use the same strategy again if agent has already used it in previous turn
            filtered_strategies = self.filter_repeat_strategies(conversation.get_prev_actions(3))
            if common.issubset(set(filtered_strategies)):
                continue

            common = common.difference(set(filtered_strategies))

            candidate_info = {}
            for action in list(common):
                candidate_info[action] = candidate.metadata[action]

            retrieved_info.append({
                # "strategies": list(common),
                "strategies": candidate_info,
                "node_id": candidate.node_id
            })

        return retrieved_info

    def rerank_nodes(self,
                     dialogue: str,
                     mov_lang_behaviour: str,
                     candidates: Optional[list[NodeWithScore]],
                     rank_type: str,
                     embedder,
                     summarizer: Agent) -> Optional[list[NodeWithScore]]:

        # rerank by comparing summary embeddings
        if rank_type in ["all", "summary"]:
            candidates = self.rank_by_summary(candidates, dialogue, embedder, summarizer)

        # rerank by mov behaviour
        if rank_type in ["all", "mov_behaviour"]:
            candidates = self.rank_by_mov_lang(candidates, mov_lang_behaviour)

        return candidates

    @staticmethod
    def filter_repeat_strategies(prev_strategies: list) -> list:
        return list(set(prev_strategies).intersection(set(list(NOT_TO_REPEAT_STRATEGIES.keys()))))

    @staticmethod
    def rank_by_summary(candidates: Optional[list[NodeWithScore]],
                        dialogue: str,
                        embedder,
                        summarizer: Agent) -> Optional[list[NodeWithScore]]:

        embedding_scores, ranked_list = [], []

        # dialogue_summary = summarizer.send_message_to_llm(get_summary_prompt(dialogue))
        dialogue_summary = summarizer.send_message(prompt=get_summary_prompt(dialogue), task="summarise")
        dialogue_embedding = embedder.get_text_embedding(dialogue_summary)

        for candidate in candidates:
            candidate_embedding = embedder.get_text_embedding(candidate.metadata["conv_summary"])
            score = util.cos_sim(dialogue_embedding, candidate_embedding)
            embedding_scores.append(float(score))

        # get sorted list of index by ascending
        sorted_indexes = np.argsort(embedding_scores)
        for idx in sorted_indexes[::-1]:  # loop in reverse
            ranked_list.append(candidates[idx])

        return ranked_list

    @staticmethod
    def rank_by_mov_lang(candidates: Optional[list[NodeWithScore]],
                         mov_lang_behaviour: str) -> Optional[list[NodeWithScore]]:

        with_behaviour, wo_behaviour = [], []
        for candidate in candidates:
            if candidate.metadata["client_behaviour"] == mov_lang_behaviour:
                with_behaviour.append(candidate)
            else:
                wo_behaviour.append(candidate)
        return with_behaviour + wo_behaviour

    @staticmethod
    def load_vector_db(args, embedder):
        strategy_db = load_vector_store(args=args,
                                        db_name=args.strategy_db_name,
                                        embedder=embedder)
        strategy_db = VectorIndexRetriever(index=strategy_db,
                                           similarity_top_k=args.num_dialog_retrieval)
        return strategy_db
