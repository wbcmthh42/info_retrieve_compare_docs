# Document Analysis and Comparison Tool

A Streamlit-based application that enables users to analyze and compare PDF documents using Azure OpenAI's language models. This tool helps users extract insights from multiple PDFs through natural language queries and provides detailed document comparison capabilities.

## Features

- 📑 PDF Document Processing
  - Upload and process multiple PDF documents
  - Automatic text extraction from PDFs
  - Document management in session state

- 💬 Multi-Document Chat Interface
  - Interactive chat with multiple documents simultaneously
  - Select specific documents to query
  - Intelligent context management with token limiting
  - Persistent chat history within session

- 🔄 Document Comparison
  - Side-by-side document comparison
  - Detailed clause-by-clause analysis
  - Identification of textual and structural differences
  - Metadata comparison

## Technical Components

### Document Processing
- Uses `PyPDF2` for PDF text extraction
- Maintains uploaded documents in Streamlit session state
- Provides real-time upload status feedback

### Chat System
- Powered by Azure OpenAI's chat models
- Smart context management:
  - Token counting and limiting
  - Automatic text truncation
  - Multi-document context preparation
  - Clear document separation in prompts

### Comparison Engine
- Structured comparison methodology
- Detailed difference analysis:
  - Word-by-word clause comparison
  - Formatting and structure analysis
  - Metadata difference detection

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   Create a `.env` file with the following variables:
   ```
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_API_VERSION=your_api_version
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_deployment_name
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Usage Guide

### Document Upload
1. Use the sidebar to upload PDF files
2. Each uploaded document will be processed and stored in the session
3. View uploaded documents list in the sidebar

### Chatting with Documents
1. Select the "Chat with Documents" tab
2. Choose which documents to include in your query using the checkboxes
3. Enter your question in the text input
4. View AI responses in the chat history
5. All documents are automatically handled within token limits

### Document Comparison
1. Select the "Compare Documents" tab
2. Choose two documents from the dropdown menus
3. Click "Compare Documents" to generate analysis
4. Review the detailed comparison results:
   - Clause-by-clause differences
   - Structural changes
   - Metadata variations

## Technical Details

### Token Management
- Maximum context length: 128,000 tokens
- Automatic token counting and text truncation
- Dynamic token allocation per document
- Buffer space reserved for system messages and responses

### Context Preparation
```python
MAX_TOTAL_TOKENS = 120000
MAX_TOKENS_PER_DOC = MAX_TOTAL_TOKENS // number_of_selected_docs
```

### Document Separation
Documents are clearly separated in the context with:
```
=== Document Separator ===
Document: [filename]
[content]
```

## Limitations

- PDF Support Only
  - Currently only processes PDF files
  - PDFs must contain extractable text

- Token Limits
  - Maximum context length of 128,000 tokens
  - Long documents are automatically truncated
  - Prioritizes beginning of documents when truncating

- Azure OpenAI Dependencies
  - Requires valid Azure OpenAI credentials
  - API availability and rate limits apply

## Best Practices

### Document Upload
- Use text-based PDFs rather than scanned documents
- Ensure PDFs are properly formatted
- Monitor upload success messages

### Querying
- Be specific in your questions
- Select relevant documents for your query
- Review chat history for context

### Comparison
- Compare similar types of documents
- Use for detailed difference analysis
- Review all comparison sections

## Error Handling

The application includes handling for:
- Token limit exceeded scenarios
- Document upload failures
- Invalid document selections
- Missing API credentials

## Future Enhancements

Potential areas for improvement:
- Support for additional file formats
- Advanced document preprocessing
- Enhanced visualization capabilities
- Improved token management strategies
- Document summarization features

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

[Specify your license here]