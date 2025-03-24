from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from vector_db import load_vector_store, create_embedder


class DiagRetriever:
    def __init__(self, args):
        self.args = args
        self.dialog_db: VectorIndexRetriever = self.load_vector_db(args)

    def retrieve(self, dialogue: str) -> tuple:
        candidates = self.dialog_db.retrieve(dialogue)
        node = self.filter_nodes(candidates)
        therapist_text = node.metadata["therapist"]
        therapist_actions = node.metadata["strategies"].split(', ')
        return therapist_text, therapist_actions

    @staticmethod
    def filter_nodes(candidates: list[NodeWithScore]) -> NodeWithScore:
        return candidates[0]

    @staticmethod
    def load_vector_db(args):
        embedder = create_embedder(args.input_type, args.cohere_key, args.retrieval_model)
        strategy_db = load_vector_store(args=args,
                                        db_name=args.strategy_db_name,
                                        embedder=embedder)
        strategy_db = VectorIndexRetriever(index=strategy_db,
                                           similarity_top_k=args.num_dialog_retrieval)

        return strategy_db
