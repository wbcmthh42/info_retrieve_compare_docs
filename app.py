import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from openai import OpenAI
import json
import PyPDF2
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import chromadb

# Set up the Streamlit page
st.set_page_config(page_title="AI Learning Resources Assistant", layout="wide")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add this at the top with other session state initializations
if "processed_data" not in st.session_state:
    st.session_state.processed_data = []

def process_pdf(file_path):
    """Process a single PDF file and return chunks"""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(pages)
    return chunks

@st.cache_resource(ttl="1h")  # Cache with 1 hour time-to-live
def initialize_vector_store(openai_api_key):
    """Initialize and return the vector store"""
    # Create the persist directory if it doesn't exist
    persist_dir = "./chroma_db"
    
    try:
        # Initialize ChromaDB client with settings
        settings = chromadb.Settings(
            is_persistent=True,
            persist_directory=persist_dir,
            anonymized_telemetry=False
        )
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
        
        # Create vector store
        vector_store = Chroma(
            collection_name="financial_docs",
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        
        return vector_store
        
    except Exception as e:
        st.error(f"Failed to initialize vector store: {str(e)}")
        st.error("Please ensure the application has write permissions in the current directory.")
        return None

def load_documents():
    """Load and process all PDFs from the data folder"""
    data_folder = 'data'
    
    if not os.path.exists(data_folder):
        st.error(f"Data folder '{data_folder}' not found!")
        return None
    
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
    if not pdf_files:
        st.warning(f"No PDF files found in '{data_folder}' folder!")
        return None
    
    all_chunks = []
    for filename in pdf_files:
        try:
            filepath = os.path.join(data_folder, filename)
            st.info(f"Processing {filename}...")
            chunks = process_pdf(filepath)
            all_chunks.extend(chunks)
            
        except Exception as e:
            st.error(f"Error processing {filename}: {str(e)}")
            continue
    
    return all_chunks

def create_visualization(data, viz_type, x=None, y=None, title=None, color=None):
    """Helper function to create different types of plots using matplotlib"""
    try:
        # Convert data to DataFrame first
        df = pd.DataFrame(data)
        
        # Identify continuous and categorical columns
        continuous_col = None
        categorical_col = None
        
        # Look for standard "Category" and "Value" columns
        if "Category" in df.columns and "Value" in df.columns:
            categorical_col = "Category"
            continuous_col = "Value"
        else:
            # Fallback to type detection
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''))
                    continuous_col = col
                except:
                    categorical_col = col
        
        if not (continuous_col and categorical_col):
            raise ValueError("Could not identify categorical and continuous columns")
        
        # Month ordering logic
        month_order = {
            'January': '01', 'Jan': '01',
            'February': '02', 'Feb': '02',
            'March': '03', 'Mar': '03',
            'April': '04', 'Apr': '04',
            'May': '05',
            'June': '06', 'Jun': '06',
            'July': '07', 'Jul': '07',
            'August': '08', 'Aug': '08',
            'September': '09', 'Sep': '09',
            'October': '10', 'Oct': '10',
            'November': '11', 'Nov': '11',
            'December': '12', 'Dec': '12'
        }
        
        # Check if categories are months and sort accordingly
        categories = df[categorical_col].tolist()
        if any(month in month_order for month in categories):
            # Create a sorting key that maps months to numbers
            df['sort_key'] = df[categorical_col].map(lambda x: month_order.get(x, x))
            df = df.sort_values('sort_key')
            df = df.drop('sort_key', axis=1)
        else:
            # For line plots and similar, maintain the original order
            if viz_type in ["line", "area"]:
                df = df.sort_values(categorical_col)
        
        # Create figure and axis
        plt.figure(figsize=(10, 6))
        
        if viz_type == "pie":
            plt.pie(df[continuous_col], labels=df[categorical_col], autopct='%1.1f%%')
        elif viz_type == "bar":
            plt.bar(range(len(df[categorical_col])), df[continuous_col])
            plt.xticks(range(len(df[categorical_col])), df[categorical_col], rotation=45, ha='right')
            plt.xlabel(categorical_col)
            plt.ylabel(continuous_col)
        elif viz_type == "bar_horizontal":
            plt.barh(range(len(df[categorical_col])), df[continuous_col])
            plt.yticks(range(len(df[categorical_col])), df[categorical_col])
            plt.xlabel(continuous_col)
            plt.ylabel(categorical_col)
        elif viz_type == "line":
            plt.plot(range(len(df[categorical_col])), df[continuous_col], marker='o')
            plt.xticks(range(len(df[categorical_col])), df[categorical_col], rotation=45, ha='right')
            plt.xlabel(categorical_col)
            plt.ylabel(continuous_col)
        elif viz_type == "scatter":
            plt.scatter(range(len(df[categorical_col])), df[continuous_col])
            plt.xticks(range(len(df[categorical_col])), df[categorical_col], rotation=45, ha='right')
            plt.xlabel(categorical_col)
            plt.ylabel(continuous_col)
        elif viz_type == "area":
            plt.fill_between(range(len(df[categorical_col])), df[continuous_col])
            plt.xticks(range(len(df[categorical_col])), df[categorical_col], rotation=45, ha='right')
            plt.xlabel(categorical_col)
            plt.ylabel(continuous_col)
        elif viz_type == "histogram":
            plt.hist(df[continuous_col], bins=20)
            plt.xlabel(continuous_col)
            plt.ylabel("Frequency")
        
        # Common plot settings
        plt.title(title if title else f"{viz_type.title()} Chart")
        plt.tight_layout()
        
        # Convert plot to image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        return buf
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        st.exception(e)  # This will show the full traceback
        return None

def get_data_info(df):
    """Get basic information about the dataset"""
    # Update this function based on your new data structure
    categories = df['Category'].unique().tolist()
    total_sales = df['Sales'].sum()
    
    return {
        "categories": categories,
        "total_sales": total_sales
    }

def query_llm(client, prompt, vector_store):
    # Perform similarity search
    relevant_docs = vector_store.similarity_search(prompt, k=3)
    
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    messages = [
        {"role": "system", "content": f"""
        You are a financial data analyst assistant that helps analyze financial data from PDFs.
        
        Use this context from the documents to answer the question:
        {context}
        
        When analyzing the data:
        1. Look for specific financial information in the provided context
        2. If found, provide detailed analysis
        3. If not found, inform the user what information is available
        
        If a visualization would be helpful, include EXACTLY ONE Python code block with matplotlib code in this format:
        ```python
        import matplotlib.pyplot as plt
        
        # Data
        categories = ["cat1", "cat2", "cat3"]
        values = [val1, val2, val3]
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        # ... plotting code ...
        plt.title("Chart Title")
        plt.xlabel("X Label")
        plt.ylabel("Y Label")
        plt.tight_layout()
        ```
        
        For monthly data, ensure to sort months chronologically, not alphabetically.
        Always include proper axis labels and titles.
        """}
    ]
    
    # Add conversation history and current prompt
    for message in st.session_state.messages:
        messages.append({
            "role": message["role"],
            "content": message["content"]
        })
    
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o",  # Updated to a current model
        messages=messages,
        temperature=0
    )
    
    response_content = response.choices[0].message.content
    
    try:
        # Try to find Python code block
        code_match = re.search(r'```python\s*(.*?)\s*```', response_content, re.DOTALL)
        
        if code_match:
            # Get the Python code
            plot_code = code_match.group(1)
            
            # Create a new figure before executing the code
            plt.figure(figsize=(10, 6))
            
            # Add necessary imports to the code
            plot_code = "import matplotlib.pyplot as plt\nimport numpy as np\n" + plot_code
            
            # Execute the plot code in a local namespace
            local_vars = {}
            exec(plot_code, globals(), local_vars)
            
            # Convert plot to image
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            
            # Store the new chart in session state
            st.session_state.charts.append(buf)
            
            # Remove the code block from the response
            response_content = re.sub(r'```python\s*.*?\s*```', '', response_content, flags=re.DOTALL)
            response_content = response_content.strip()
    
    except Exception as e:
        st.error(f"Error processing visualization: {str(e)}")
        st.exception(e)
    
    return response_content

def main():
    st.title("Financial Data Assistant")
    
    # Initialize charts list in session state if it doesn't exist
    if 'charts' not in st.session_state:
        st.session_state.charts = []
    
    # Get API key
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    if not openai_api_key:
        st.info("Please enter your OpenAI API key to continue.")
        return
    
    client = OpenAI(api_key=openai_api_key)
    
    # Initialize vector store with better error handling
    try:
        vector_store = initialize_vector_store(openai_api_key)
        if vector_store is None:
            st.error("Vector store initialization failed. Please check your settings and try again.")
            return
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return
    
    # Load documents button
    if st.sidebar.button("Load/Reload Documents"):
        with st.spinner("Processing documents..."):
            try:
                chunks = load_documents()
                if chunks:
                    try:
                        vector_store.add_documents(chunks)
                        st.success(f"Processed {len(chunks)} document chunks!")
                    except Exception as e:
                        st.error(f"Error adding documents to vector store: {str(e)}")
                else:
                    st.warning("No documents were loaded. Please check your data folder.")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "viz" in message and message["viz"] is not None:  # Add null check
                message["viz"].seek(0)  # Reset buffer position
                st.image(message["viz"].getvalue(), use_container_width=True)

    # Chat input
    if prompt := st.chat_input("Ask me about the financial data!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Clear previous charts before getting new response
            st.session_state.charts = []
            
            response = query_llm(client, prompt, vector_store)
            st.markdown(response)
            
            # Display only the charts generated from this response
            for chart in st.session_state.charts:
                if chart is not None:  # Add null check
                    chart.seek(0)  # Reset buffer position
                    st.image(chart.getvalue(), use_container_width=True)
            
            # Store both the response and the chart in the message history
            message_with_viz = {
                "role": "assistant", 
                "content": response
            }
            if st.session_state.charts:  # Only add viz if there are charts
                message_with_viz["viz"] = st.session_state.charts[0]
            
            st.session_state.messages.append(message_with_viz)

if __name__ == "__main__":
    main()
