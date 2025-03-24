import json
import argparse
import tqdm
from typing import Union
import chromadb
from chromadb.errors import InvalidCollectionException

from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def load_json_file(path_to_data_file: str):
    with open(path_to_data_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def create_embedder(input_type: str,
                    cohere_key: str,
                    retrieval_model: str) -> Union[CohereEmbedding, HuggingFaceEmbedding]:
    # create cohere embedder
    # https://docs.cohere.com/reference/embed
    if retrieval_model == 'cohere':
        return CohereEmbedding(
            cohere_api_key=cohere_key,
            model_name="embed-english-v3.0",
            input_type=input_type  # Used for embeddings stored in a vector database for search use-cases.
        )
    else:
        model_mapping = {
            "sbert": "sentence-transformers/all-MiniLM-L6-v2",
            "baai": "BAAI/bge-m3"
        }
        return HuggingFaceEmbedding(
            model_name=model_mapping[retrieval_model]
        )


def create_vector_store(dialogues: list, storage_dir: str, db_name: str, embedder) -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=storage_dir)
    # print("Create new collection...")
    collection = chroma_client.create_collection(name=db_name)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(dialogues, storage_context=storage, embed_model=embedder, show_progress=True)


def load_vector_store(args, db_name: str, embedder) -> VectorStoreIndex:
    try:
        chroma_client = chromadb.PersistentClient(path=args.storage_dir)
        collection = chroma_client.get_collection(name=db_name)
        # print(f"Total items in {db_name} collection: {collection.count()}")

        vector_store = ChromaVectorStore(chroma_collection=collection)
        return VectorStoreIndex.from_vector_store(vector_store, embed_model=embedder)

    except InvalidCollectionException:
        if "mov_lang" in db_name:
            return create_mov_lang_db(args, embedder)
        elif "strategy" in db_name:
            return create_strategy_db(args, embedder)
        else:
            raise FileNotFoundError(f"Collection {db_name} NOT exist. Please create it first.")


def get_dialogue(utt_info: dict, therapist_utt_setting: str, task: str) -> str:
    if therapist_utt_setting == "default":
        if task == "attitude":
            return f"Therapist: {utt_info['prev_utt']}\nClient: \"{utt_info['utterance']}\"" \
                if len(utt_info['prev_utt']) > 0 else f"Client: \"{utt_info['utterance']}\""
        else:
            return f"Client: \"{utt_info['utterance']}\""

    elif therapist_utt_setting == "w_therapist":
        return f"Therapist: {utt_info['prev_utt']}\nClient: \"{utt_info['utterance']}\"" \
            if len(utt_info['prev_utt']) > 0 else f"Client: \"{utt_info['utterance']}\""

    return f"Client: \"{utt_info['utterance']}\""


def create_mov_nodes(args) -> list:
    corpus = [args.mov_lang_data]
    if args.use_test_data:
        corpus.append(args.mov_lang_test_data)

    dialogues = []
    for file in corpus:
        data = load_json_file(f"{args.data_dir}/{file}")

        # print(f"Processing file: {file} ...")
        bar = tqdm.tqdm(total=len(data))
        for utt_id, utt_info in data.items():
            text_attitude = get_dialogue(utt_info, args.therapist_utt_setting, task="attitude")
            text_strength = get_dialogue(utt_info, args.therapist_utt_setting, task="strength")

            attitude, strength = utt_info['main_behaviour'].split('-')
            dialog_attitude = TextNode(text=text_attitude,
                                       id_=f"{utt_id}_att",
                                       metadata={"attitude": attitude})
            dialog_strength = TextNode(text=text_strength,
                                       id_=f"{utt_id}_str",
                                       metadata={"strength": strength})

            dialog_attitude.excluded_llm_metadata_keys = ["attitude"]
            dialog_strength.excluded_llm_metadata_keys = ["strength"]
            dialog_attitude.excluded_embed_metadata_keys = ["attitude"]
            dialog_strength.excluded_embed_metadata_keys = ["strength"]

            dialogues.extend([dialog_attitude, dialog_strength])
            bar.update(1)
        bar.close()
    return dialogues


def create_strategy_nodes(args) -> list:
    corpus = [args.strategy_data]
    if args.use_test_data:
        corpus.append(args.strategy_test_data)

    dialogues = []
    for file in corpus:
        data = load_json_file(f"{args.data_dir}/{file}")

        # print(f"Processing file: {file} ...")
        bar = tqdm.tqdm(total=len(data))
        for utt_id, utt_info in data.items():
            strategies = list(utt_info['sub_behaviours'].keys())
            if len(strategies) == 0:
                strategies = [utt_info["main_behaviour"]]

            excluded_keys = ["strategies"]
            metadata = {"strategies": ', '.join(strategies),
                        "client_behaviour": utt_info["client_mov"],
                        "conv_summary": utt_info["summary"] if "summary" in utt_info else ""}

            for strategy in strategies:
                if strategy in utt_info["sub_behaviours"] and len(utt_info["sub_behaviours"][strategy]):
                    strategy_text = utt_info["sub_behaviours"][strategy]
                else:
                    strategy_text = utt_info["therapist_utt"]

                    # if len(strategy_text) == 0:
                    #     strategy_text = utt_info["therapist_utt"]

                metadata[strategy] = strategy_text
                excluded_keys.append(strategy)

            dialog = TextNode(text=get_dialog_text(utt_info),
                              id_=utt_id,
                              metadata=metadata)
            dialog.excluded_llm_metadata_keys = excluded_keys
            dialog.excluded_embed_metadata_keys = excluded_keys
            dialogues.append(dialog)
            bar.update(1)
        bar.close()
    return dialogues


def get_dialog_text(utt_info: dict) -> str:
    if len(utt_info["client_prev_utt"]) > 0:
        return (f"Client: {utt_info['client_prev_utt']}\nTherapist: {utt_info['therapist_prev_utt']}"
                f"\nClient: {utt_info['client_utt']}")
    elif len(utt_info["client_prev_utt"]) == 0 and len(utt_info["therapist_prev_utt"]) > 0:
        return f"Therapist: {utt_info['therapist_prev_utt']}\nClient: {utt_info['client_utt']}"
    return f"Client: {utt_info['client_utt']}"


def create_strategy_db(args, embedder):
    # create nodes and vector database for retrieving similar dialogues for counselling strategies
    strategy_nodes = create_strategy_nodes(args)
    # print(f"Number of nodes: {len(strategy_nodes)}")

    # args.strategy_db_name = f"{args.strategy_db_name}_{args.retrieval_model}"
    # print(f"DB name: {args.strategy_db_name}")
    return create_vector_store(strategy_nodes, args.storage_dir, args.strategy_db_name, embedder)


def create_mov_lang_db(args, embedder):
    # create nodes and vector database for motivational lang detection
    mov_dialogue_nodes = create_mov_nodes(args)
    # print(f"Number of nodes: {len(mov_dialogue_nodes)}")

    # args.mov_lang_db_name = f"{args.mov_lang_db_name}_{args.retrieval_model}"
    # args.mov_lang_db_name = f"{args.mov_lang_db_name}" if args.therapist_utt_setting == "default" \
    #     else f"{args.mov_lang_db_name}_{args.therapist_utt_setting}"
    # print(f"DB name: {args.mov_lang_db_name}")
    return create_vector_store(mov_dialogue_nodes, args.storage_dir, args.mov_lang_db_name, embedder)


def main():
    args = parser.parse_args()
    # free api: qFmdRPukpVgLHPtMhN6VPXWEGrOqkVpkyr0HR8R7
    # paid api: W4embGAuc0DaFvRudVrv6C730cp21GwGuXgpca1i

    embedder = create_embedder(args.input_type, args.cohere_key, args.retrieval_model)
    create_strategy_db(args, embedder)
    # create_mov_lang_db(args, embedder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval_model', type=str, default='cohere',
                        choices=['cohere', 'sbert', 'baai'])
    parser.add_argument('--cohere_key', type=str, default="W4embGAuc0DaFvRudVrv6C730cp21GwGuXgpca1i")
    parser.add_argument('--storage_dir', type=str, default='./chroma_storage')
    parser.add_argument('--data_dir', type=str, default='./data/dataset')
    parser.add_argument('--use_test_data', type=bool, default=True)
    parser.add_argument('--strategy_data', type=str, default='therapist_strategy/train_therapist.json')
    parser.add_argument('--strategy_test_data', type=str, default='therapist_strategy/annomi_test_therapist_with_sum.json')
    parser.add_argument('--therapist_utt_setting', type=str, default="default",
                        choices=["default", "w_therapist", "wo_therapist"])
    parser.add_argument('--mov_lang_data', type=str, default='mov_lang_detect/train_client.json')
    parser.add_argument('--mov_lang_test_data', type=str, default='mov_lang_detect/annomi_test_client.json')
    parser.add_argument('--mov_lang_db_name', type=str, default='mov_lang_db')
    parser.add_argument('--strategy_db_name', type=str, default='diag_strategy_db')
    parser.add_argument('--input_type', type=str, default="search_document")
    parser.add_argument('--max_num_prev_turns', type=int, default=3,
                        help='The number of previous turns in the dialogue to use as context for embedding and prompting')

    main()
