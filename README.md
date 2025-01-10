# Financial Data Assistant

A POC for a Streamlit-based application that uses AI to analyze financial reports and provide interactive data visualizations. This tool helps users extract insights from financial PDFs through natural language queries and dynamic chart generation.

## Features

- 📊 Interactive data visualization with 15+ chart types
- 🤖 AI-powered natural language processing for financial analysis
- 📑 PDF document processing and semantic search
- 💬 Chat-based interface for easy interaction
- 📈 Dynamic chart generation based on context
- 🔄 Real-time document processing

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **OpenAI API Key**
   - You'll need an OpenAI API key to use this application
   - Get your API key from [OpenAI's platform](https://platform.openai.com/api-keys)

3. **Prepare Your Data**
   - Create a `data` folder in the project root
   - Place your financial PDF reports in the `data` folder
   - Supported format: PDF files only

## Usage

1. **Start the Application**
   ```bash
   streamlit run app.py
   ```

2. **Initial Setup**
   - Enter your OpenAI API key in the sidebar
   - Click "Load/Reload Documents" to process your PDF files
   - Wait for confirmation that documents are processed

3. **Analyzing Data**
   - Use the chat interface to ask questions about your financial data
   - Example queries:
     - "What were the total revenues for last year?"
     - "Show me the trend of operating expenses"
     - "Compare profit margins across different quarters"
     - "Visualize the revenue breakdown by segment"

4. **Visualization Types**
   The system supports various chart types for different analytical needs:

   - **Basic Charts**
     - Pie charts (parts of a whole)
     - Bar charts (vertical/horizontal comparisons)
     - Line charts (trends over time)
     - Scatter plots (correlations)

   - **Advanced Charts**
     - Stacked/Grouped bars
     - Area charts
     - Funnel charts
     - Timeline visualizations

   - **Complex Charts**
     - Waterfall charts (sequential changes)
     - Radar charts (multi-variable comparison)
     - Bubble charts (three-variable visualization)

   - **Statistical Charts**
     - Violin plots (distribution analysis)
     - Box plots (statistical distribution)
     - Histograms (frequency distribution)
     - Density heatmaps (2D distribution)

## How It Works

1. **Document Processing**
   - PDFs are loaded and split into manageable chunks
   - Text is processed using LangChain's document splitter
   - Chunks are embedded using OpenAI's embedding model
   - Embeddings are stored in a Chroma vector database

2. **Query Processing**
   - User queries are analyzed for context
   - Relevant document chunks are retrieved using similarity search
   - GPT-4 processes the query with context to generate insights
   - Appropriate visualizations are automatically suggested and generated

3. **Visualization Generation**
   - The AI determines the most suitable chart type
   - Data is formatted according to the visualization requirements
   - Interactive Plotly charts are rendered in the interface

## Best Practices

1. **Document Preparation**
   - Use clear, well-formatted PDF reports
   - Ensure PDFs are text-searchable (not scanned images)
   - Keep individual files under 10MB for optimal processing

2. **Querying**
   - Be specific in your questions
   - Mention time periods or specific metrics when relevant
   - Ask for specific visualization types if preferred

3. **Data Analysis**
   - Start with broad questions and drill down
   - Use the chat history to build on previous insights
   - Leverage different chart types for various analytical perspectives

## Limitations

- Currently supports PDF files only
- Requires internet connection for API calls
- Processing very large PDFs may take time
- API key required for OpenAI services

## Contributing

Feel free to submit issues and enhancement requests!