import streamlit as st
import tempfile
import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.vectorstores import Chroma
from langchain_huggingface.llms import HuggingFacePipeline
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

# make sure streamlit works fine
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None
    

# download embedding model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name = 'bkai-foundation-models/vietnamese-bi-encoder'
    )

# download llm
@st.cache_resource
def load_llm():
    MODEL_NAME = 'lmsys/vicuna-7b-v1.5'
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )
    
    return HuggingFacePipeline(pipeline=model_pipeline)


# handle PDF
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    
    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type='percentile',
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )
    
    docs = semantic_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(
        documents=docs, embedding=st.session_state.embeddings
    )
    
    retriever = vector_db.as_retriever()
    
    prompt = hub.pull('rlm/rag-prompt')
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt 
        | st.session_state.llm
        | StrOutputParser()
    )
    
    os.unlink(tmp_file_path)
    return rag_chain, len(docs)

# UI
st.set_page_config(page_title="PDF RAG Assistant", layout='wide')
st.title('PDF RAG Assistant')

st.markdown("""
**Ứng dụng AI giúp bạn hỏi đáp trực tiếp với nội dung tài liệu PDF bằng tiếng Việt**
**Cách sử dụng đơn giản:**
1. **Upload PDF** Chọn file PDF từ máy tính và nhấn "Xử lý PDF"
2. **Đặt câu hỏi** Nhập câu hỏi về nội dung tài liệu và nhận câu trả lời ngay lập tức
 ---
 """)

if not st.session_state.models_loaded:
    st.info("Downloading models...")
    st.session_state.embeddings = load_embeddings()
    st.session_state.llm = load_llm()
    st.session_state.models_loaded = True
    st.success("Models are ready")
    st.rerun()
    
# upload and handle PDF
uploaded_file = st.file_uploader("Upload file PDF", type='pdf')
if uploaded_file and st.button('Handle PDF'):
    with st.spinner("Handling..."):
        st.session_state.rag_chain, num_chunk = process_pdf(uploaded_file)
        st.success(f'Successful! {num_chunk} chunks')

# ask and answer UI
if st.session_state.rag_chain:
    question = st.text_input('Ask something:')
    if question:
        with st.spinner("Answering..."):
            output = st.session_state.rag_chain.invoke(question)
            answer = output.split("Answer:")[1].strip() if "Answer:" in output else output.strip()
            st.write("**Answer:**")
            st.write(answer)
