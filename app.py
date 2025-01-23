import streamlit as st
from PyPDF2 import PdfReader
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def load_pdf_pypdf(uploaded_file):
    """Load and extract text from uploaded PDF file"""
    loader = PdfReader(uploaded_file)
    text = ""
    for page in loader.pages:
        text += page.extract_text() + "\n"
    return text

def get_chat_response(prompt, context=""):
    """Get response from Azure OpenAI"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions about documents and analyzes their content."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
    ]
    
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

def compare_documents(doc1_text, doc2_text):
    """Compare two documents and return analysis"""
    prompt = f"""Compare these two documents and identify key differences:

    Document 1:
    {doc1_text}

    Document 2:
    {doc2_text}

    Please compare the two documents clause by clause and identify any differences in text content, structure, or formatting. Follow these instructions strictly:

    1. **Detailed Clause Comparison**: 
    - Compare every clause word by word and highlight any differences, such as added, removed, or modified content.
    - Include differences in formatting, punctuation, or structural arrangement if present.

    2. **Highlighting Differences**: 
    - For each clause, explicitly state:
        - Any words, phrases, or sections that differ.
        - Any changes in wording, phrasing, or formatting.
    - Use clear and precise descriptions of the differences.

    3. **No Difference Clause**: 
    - If a clause is identical in both documents, explicitly state: "No differences found in this clause."

    4. **Avoiding Assumptions**: 
    - Do not assume any clause is identical unless verified word by word. Conduct a thorough and systematic comparison.

    5. **Metadata and Structural Changes**:
    - Include any differences in document metadata, headers, footers, or overall structure.

    6. **Output Format**:
    Present the findings in the following structured format:
    
    ### Clause-by-Clause Comparison

    **Clause [X]:**  
    - **Differences:** [Detail all differences word by word or confirm "No differences found."]  
    - **Conclusion:** [Summarize whether the clause differs or not.]

    ### Metadata and Structure
    - **Metadata Differences:** [Highlight differences in metadata, such as file creation dates, titles, or page counts.]  
    - **Structural Differences:** [Note changes in headers, page numbers, or section arrangements.]

    This ensures a systematic and thorough comparison of all clauses without skipping or assuming similarity. Provide the analysis in a detailed and structured manner.
    """
    
    return get_chat_response(prompt)

def count_tokens(text):
    """Rough estimate of token count"""
    return len(text.split()) * 1.3  # Rough approximation

def truncate_text(text, max_tokens=1000):
    """Truncate text to approximate token limit"""
    words = text.split()
    estimated_tokens = 0
    for i, word in enumerate(words):
        estimated_tokens += 1.3  # Rough approximation
        if estimated_tokens > max_tokens:
            return " ".join(words[:i]) + "..."
    return text

def prepare_context(documents_dict, selected_docs):
    """Prepare context from selected documents while respecting token limits"""
    # Reserve tokens for system message, user question, and response
    MAX_TOTAL_TOKENS = 120000  # Leave buffer for system message and response
    MAX_TOKENS_PER_DOC = MAX_TOTAL_TOKENS // max(len(selected_docs), 1)
    
    contexts = []
    for doc_name, is_selected in selected_docs.items():
        if is_selected:
            doc_text = documents_dict[doc_name]
            truncated_text = truncate_text(doc_text, MAX_TOKENS_PER_DOC)
            contexts.append(f"Document: {doc_name}\n{truncated_text}")
    
    return "\n\n=== Document Separator ===\n\n".join(contexts)

# Streamlit UI
st.title("Document Analysis and Comparison Tool")

# Initialize session state for storing uploaded documents
if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for document upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'], key="file_uploader")
    
    if uploaded_file is not None:
        doc_name = uploaded_file.name
        if doc_name not in st.session_state.uploaded_docs:
            with st.spinner(f"Processing {doc_name}..."):
                text_content = load_pdf_pypdf(uploaded_file)
                st.session_state.uploaded_docs[doc_name] = text_content
                st.success(f"✅ {doc_name} uploaded successfully!")

    # Display uploaded documents
    if st.session_state.uploaded_docs:
        st.header("Uploaded Documents")
        for doc_name in st.session_state.uploaded_docs.keys():
            st.text(f"📄 {doc_name}")

# Main area
tab1, tab2 = st.tabs(["Chat with Documents", "Compare Documents"])

# Chat Interface
with tab1:
    st.header("Chat with Documents")
    
    if st.session_state.uploaded_docs:
        st.write("Select documents to query:")
        selected_docs = {}
        for doc_name in st.session_state.uploaded_docs.keys():
            selected_docs[doc_name] = st.checkbox(f"📄 {doc_name}", value=True)
        
        user_question = st.text_input("Ask a question about the selected documents:")
        if user_question:
            if any(selected_docs.values()):
                with st.spinner("Generating response..."):
                    # Use the new prepare_context function
                    combined_context = prepare_context(
                        st.session_state.uploaded_docs,
                        selected_docs
                    )
                    
                    response = get_chat_response(user_question, combined_context)
                    
                    st.session_state.chat_history.append(("user", user_question))
                    st.session_state.chat_history.append(("assistant", response))
            else:
                st.warning("Please select at least one document to query.")
        
        st.write("### Chat History")
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.write("You:", message)
            else:
                st.write("Assistant:", message)
    else:
        st.info("Please upload a document to start chatting.")

# Compare Documents Interface
with tab2:
    st.header("Compare Documents")
    
    if len(st.session_state.uploaded_docs) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            doc1 = st.selectbox(
                "Select first document:",
                options=list(st.session_state.uploaded_docs.keys()),
                key="doc1"
            )
        
        with col2:
            doc2 = st.selectbox(
                "Select second document:",
                options=list(st.session_state.uploaded_docs.keys()),
                key="doc2"
            )
        
        if st.button("Compare Documents"):
            if doc1 != doc2:
                with st.spinner("Analyzing differences..."):
                    comparison_result = compare_documents(
                        st.session_state.uploaded_docs[doc1],
                        st.session_state.uploaded_docs[doc2]
                    )
                    st.write("### Comparison Results")
                    st.write(comparison_result)
            else:
                st.warning("Please select different documents for comparison.")
    else:
        st.info("Please upload at least two documents to compare.")
