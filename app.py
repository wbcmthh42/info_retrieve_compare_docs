import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

@st.cache_resource
def initialize_vector_store(openai_api_key):
    """Initialize and return the vector store"""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Create/load vector store
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    return vector_store

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
    """Helper function to create different types of plots"""
    df = pd.DataFrame(data)
    
    # Get the actual column names from the DataFrame
    columns = df.columns.tolist()
    x_col = columns[0]  # First column
    y_col = columns[1]  # Second column

    if viz_type == "pie":
        fig = px.pie(df, names=x_col, values=y_col, title=title)
    elif viz_type == "bar":
        fig = px.bar(df, x=x_col, y=y_col, title=title, color=color)
    elif viz_type == "bar_horizontal":
        fig = px.bar(df, x=y_col, y=x_col, title=title, color=color, orientation='h')
    elif viz_type == "line":
        fig = px.line(df, x=x_col, y=y_col, title=title, color=color)
    elif viz_type == "scatter":
        fig = px.scatter(df, x=x_col, y=y_col, title=title, color=color)
    elif viz_type == "area":
        fig = px.area(df, x=x_col, y=y_col, title=title, color=color)
    elif viz_type == "bar_stacked":
        fig = px.bar(df, x=x_col, y=y_col, title=title, color=color, barmode='stack')
    elif viz_type == "bar_grouped":
        fig = px.bar(df, x=x_col, y=y_col, title=title, color=color, barmode='group')
    elif viz_type == "funnel":
        fig = px.funnel(df, x=y_col, y=x_col, title=title)
    elif viz_type == "timeline":
        fig = px.timeline(df, x_start=x_col, x_end=y_col, y=color if color else x_col, title=title)
    elif viz_type == "waterfall":
        fig = go.Figure(go.Waterfall(
            name="Waterfall",
            orientation="v",
            measure=["relative"] * len(df),
            x=df[x_col],
            y=df[y_col],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig.update_layout(title=title)
    elif viz_type == "radar":
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=df[y_col],
            theta=df[x_col],
            fill='toself'
        ))
        fig.update_layout(title=title)
    elif viz_type == "bubble":
        size_col = columns[2] if len(columns) > 2 else y_col
        fig = px.scatter(df, x=x_col, y=y_col, size=size_col, title=title)
    elif viz_type == "violin":
        fig = px.violin(df, x=x_col, y=y_col, title=title)
    elif viz_type == "box":
        fig = px.box(df, x=x_col, y=y_col, title=title)
    elif viz_type == "histogram":
        fig = px.histogram(df, x=x_col, title=title)
    elif viz_type == "density_heatmap":
        fig = px.density_heatmap(df, x=x_col, y=y_col, title=title)
    
    # Update layout for better appearance
    fig.update_layout(
        title_x=0.5,
        margin=dict(t=50, l=50, r=50, b=50),
        showlegend=True
    )
    
    # Add hover data formatting
    if hasattr(fig, 'update_traces'):
        fig.update_traces(hovertemplate=None)
    
    return fig

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
        
        If a visualization would be helpful, include EXACTLY ONE JSON block in this format:
        {{
            "create_viz": true,
            "type": "<visualization_type>",
            "title": "Chart Title",
            "data": [
                {{"Category": "A", "Value": 100}},
                {{"Category": "B", "Value": 200}}
            ]
        }}
        
        Supported visualization types:
        Basic Charts:
        - "pie": For parts of a whole
        - "bar": Vertical bars
        - "bar_horizontal": Horizontal bars
        - "line": Trends over time
        - "scatter": Correlation between variables
        
        Advanced Charts:
        - "bar_stacked": Stacked bars
        - "bar_grouped": Grouped bars
        - "area": Area chart
        - "funnel": Funnel analysis
        - "timeline": Time-based events
        
        Complex Charts:
        - "sunburst": Hierarchical data
        - "treemap": Hierarchical proportions
        - "waterfall": Sequential changes
        - "radar": Multi-variable comparison
        - "bubble": Three-variable visualization
        
        Statistical Charts:
        - "violin": Distribution analysis
        - "box": Statistical distribution
        - "histogram": Frequency distribution
        - "density_heatmap": 2D distribution
        
        Format the data structure appropriately for the chosen visualization type.
        For hierarchical charts (sunburst, treemap), provide nested categories.
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
    
    return response.choices[0].message.content

def main():
    st.title("Financial Data Assistant")
    
    # Get API key
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    if not openai_api_key:
        st.info("Please enter your OpenAI API key to continue.")
        return
    
    client = OpenAI(api_key=openai_api_key)
    
    # Initialize vector store
    vector_store = initialize_vector_store(openai_api_key)
    
    # Load documents button
    if st.sidebar.button("Load/Reload Documents"):
        with st.spinner("Processing documents..."):
            chunks = load_documents()
            if chunks:
                vector_store.add_documents(chunks)
                st.success(f"Processed {len(chunks)} document chunks!")

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "viz" in message:
                st.plotly_chart(message["viz"], key=f"hist_viz_{idx}")

    # Chat input
    if prompt := st.chat_input("Ask me about the financial data!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = query_llm(client, prompt, vector_store)
            
            # Check if response contains visualization suggestion
            viz = None
            try:
                json_match = re.search(r'\{(?:[^{}]|{[^{}]*})*\}', response)
                if json_match:
                    json_str = json_match.group()
                    json_str = json_str.replace("'", '"').strip()
                    viz_data = json.loads(json_str)
                    
                    if viz_data.get("create_viz"):
                        viz = create_visualization(
                            data=viz_data["data"],
                            viz_type=viz_data["type"],
                            title=viz_data["title"]
                        )
                        st.plotly_chart(viz, use_container_width=True, 
                                      key=f"new_viz_{len(st.session_state.messages)}")
                        # Remove the JSON from the response
                        response = response.replace(json_str, '')
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
            
            # Display the text response
            st.markdown(response)
            
            # Add assistant response to chat history
            message = {"role": "assistant", "content": response}
            if viz:
                message["viz"] = viz
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
