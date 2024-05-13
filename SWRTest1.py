import os
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core import Settings

documents = SimpleDirectoryReader("data").load_data()

os.environ["OPENAI_API_KEY"] = "sk-proj-WcFQdhwLGC32DOgoP7NLT3BlbkFJovbzcYSzWdaH9uWeKA8c"

# create the sentence window node parser w/ default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# base node parser is a sentence splitter
text_splitter = SentenceSplitter()

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
#embed_model = HuggingFaceEmbedding(
#    model_name="sentence-transformers/all-mpnet-base-v2", max_length=512
#)

embed_model = OpenAIEmbedding()

Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = text_splitter

# Extract Nodes
nodes = node_parser.get_nodes_from_documents(documents)
base_nodes = text_splitter.get_nodes_from_documents(documents)

# Build the Indexes
sentence_index = VectorStoreIndex(nodes)
base_index = VectorStoreIndex(base_nodes)

# MetadataReplacementPostProcessor to replace the sentence in each node with it's surrounding context.

query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    # the target key defaults to `window` to match the node_parser's default
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
window_response = query_engine.query(
    "How much is invested in HDFC Mutual Funds?"
)
print(window_response)
