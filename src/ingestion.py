from src.embeddings import EmbeddingManager
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader

class IngestionPipeline:
    def __init__(self):
        self.embedding_manger = EmbeddingManager()

    def load_document(self, file_path: str):
        loader = PyMuPDFReader()
        documents = loader.load(file_path=file_path)

        return documents

    def parse_document(self, documents):
        text_parser = SentenceSplitter(
            chunk_size=200,
            # separator=" ",
        )

        text_chunks = []
        # maintain relationship with source doc index, to help inject doc metadata in (3)
        doc_idxs = []
        for doc_idx, doc in enumerate(documents):
            cur_text_chunks = text_parser.split_text(doc.text)
            text_chunks.extend(cur_text_chunks)
            doc_idxs.extend([doc_idx] * len(cur_text_chunks))

        from llama_index.core.schema import TextNode

        nodes = []
        for idx, text_chunk in enumerate(text_chunks):
            node = TextNode(
                text=text_chunk,
            )
            src_doc = documents[doc_idxs[idx]]
            node.metadata = src_doc.metadata
            nodes.append(node)

        for node in nodes:
            node_embedding = self.embedding_manger.get_embeddings(
                node.get_content(metadata_mode="all")
            )
            node.embedding = node_embedding

        return nodes
