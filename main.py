from src.config import db_config
from src.retrieval import VectorDBRetriever


def main():

    from src.embeddings import EmbeddingManager

    embedded_object = EmbeddingManager()
    embed_model = embedded_object.get_embed_model()

    from src.LLM import LLM
    llm_object = LLM()
    llm = llm_object.get_llm_instance()

    from src.database import DatabaseManager
    database_object = DatabaseManager(db_config=db_config)
    vector_store = database_object.get_vector_store()


    # load data
    from src.ingestion import IngestionPipeline
    file_path = "/Users/srdarkseer/Developer/rag_llama_index/data/SRD_CV.pdf"

    ingestion_pipeline_object = IngestionPipeline()
    documents = ingestion_pipeline_object.load_document(file_path=file_path)

    nodes = ingestion_pipeline_object.parse_document(documents=documents)

    vector_store.add(nodes)


    # query_str = "Can you tell me about the key concepts for safety finetuning"
    # print("*"*100)
    # query_str = "WHO IS SUSHANT?"
    # query_embedding = embed_model.get_query_embedding(query_str)

    # construct vector store query
    from llama_index.core.vector_stores import VectorStoreQuery

    # query_mode = "default"
    # # query_mode = "sparse"
    # # query_mode = "hybrid"
    #
    # vector_store_query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=2, mode=query_mode)
    #
    # # returns a VectorStoreQueryResult
    # query_result = vector_store.query(vector_store_query)
    # print("="*200)
    # print(query_result.nodes[0].get_content())

    # from llama_index.core.schema import NodeWithScore
    # from typing import Optional
    #
    # nodes_with_scores = []
    # for index, node in enumerate(query_result.nodes):
    #     score: Optional[float] = None
    #     if query_result.similarities is not None:
    #         score = query_result.similarities[index]
    #     nodes_with_scores.append(NodeWithScore(node=node, score=score))

    retriever = VectorDBRetriever(
        vector_store, embed_model, query_mode="default", similarity_top_k=2
    )

    from llama_index.core.query_engine import RetrieverQueryEngine

    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

    query_str = "Which companies has Sushant worked in?"


    print("(****)"*100)
    response = query_engine.query(query_str)

    print(str(response))

    print(response.source_nodes[0].get_content())


if __name__ == "__main__":
    main()
