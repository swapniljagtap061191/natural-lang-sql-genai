# Hugging Face Integration Guide

This guide explains how to integrate Hugging Face models with the T-Shirt Selling AI system to use the `few_shots.py` examples for SQL query generation.

## 🚀 Overview

The integration allows you to use Hugging Face transformer models instead of Google Gemini for generating SQL queries from natural language questions. The system uses few-shot learning by incorporating the examples from `few_shots.py` into the prompts.

## 📦 Installation

### 1. Install Required Dependencies

Add these packages to your `requirements.txt`:

```txt
transformers>=4.21.0
torch>=1.9.0
accelerate>=0.20.0
langchain-huggingface
```

Install them:

```bash
pip install transformers torch accelerate langchain-huggingface
```

### 2. Environment Setup

No API keys are required for Hugging Face models - they run locally on your machine.

## 🏗️ Architecture

### Files Added/Modified:

1. **`huggingface_helper.py`** - New file containing Hugging Face integration
2. **`langchain_helper.py`** - Updated to support both Gemini and Hugging Face
3. **`main.py`** - Updated UI to allow provider selection
4. **`requirements.txt`** - Added Hugging Face dependencies

### Key Components:

#### `HuggingFaceSQLAssistant` Class

```python
class HuggingFaceSQLAssistant:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "auto")
    def create_few_shot_prompt(self, user_question: str, table_info: str = "") -> str
    def generate_sql_query(self, user_question: str, table_info: str = "") -> str
```

#### Few-Shot Prompting

The system automatically incorporates your `few_shots.py` examples:

```python
few_shots = [
    {'Question': "How many t-shirts do we have left for Nike in XS size and white color?",
     'SQLQuery': "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND color = 'White' AND size = 'XS'",
     'Answer': "91"},
    # ... more examples
]
```

## 🎯 Usage

### 1. Streamlit Web Interface

Run the application and select "huggingface" as the provider:

```bash
streamlit run main.py
```

Choose from available models:
- **DialoGPT Medium/Large/Small** - Conversational models optimized for dialogue
- **BlenderBot** - Facebook's conversational AI model

### 2. Python API

#### Basic Usage

```python
from huggingface_helper import HuggingFaceSQLAssistant

# Initialize assistant
assistant = HuggingFaceSQLAssistant(model_name="microsoft/DialoGPT-medium")

# Generate SQL query
question = "How many white Nike t-shirts are available in size L?"
sql_query = assistant.generate_sql_query(question)

print(f"Generated SQL: {sql_query}")
```

#### Using with Database Chain

```python
from langchain_helper import get_unified_db_chain

# Get Hugging Face chain
chain = get_unified_db_chain(provider="huggingface", model_name="microsoft/DialoGPT-medium")

# Ask questions
response = chain.run("What are the available brands?")
print(response)
```

### 3. Command Line Testing

#### Test Hugging Face Only
```bash
python langchain_helper.py --huggingface
```

#### Test Both Providers
```bash
python langchain_helper.py --both
```

## 🤖 Available Models

| Model | Size | Best For | Memory Usage |
|-------|------|----------|--------------|
| `microsoft/DialoGPT-small` | ~117MB | Fast inference, limited accuracy | Low |
| `microsoft/DialoGPT-medium` | ~345MB | Balanced performance | Medium |
| `microsoft/DialoGPT-large` | ~1.2GB | Best accuracy, slower | High |
| `facebook/blenderbot-400M-distill` | ~1.6GB | General conversation | High |

## 🔧 Configuration

### Model Selection

Choose models based on your hardware:

```python
# For low-end hardware
assistant = HuggingFaceSQLAssistant("microsoft/DialoGPT-small")

# For better performance
assistant = HuggingFaceSQLAssistant("microsoft/DialoGPT-large")
```

### Device Configuration

```python
# Auto-detect (recommended)
assistant = HuggingFaceSQLAssistant(device="auto")

# Force CPU
assistant = HuggingFaceSQLAssistant(device="cpu")

# Force GPU (if available)
assistant = HuggingFaceSQLAssistant(device="cuda")
```

## 📝 How Few-Shot Learning Works

### 1. Prompt Construction

The system builds prompts like this:

```
You are a SQL expert. Convert natural language questions to SQL queries.
Use the following examples as reference:

Example 1:
Question: How many t-shirts do we have left for Nike in XS size and white color?
SQL: SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND color = 'White' AND size = 'XS'
Answer: 91

Example 2:
Question: How much is the total price of the inventory for all S-size t-shirts?
SQL: SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'
Answer: 22292

[... more examples ...]

Database Schema:
[table information]

Question: [user question]
SQL:
```

### 2. SQL Generation

The model generates SQL based on the patterns it learns from the examples.

### 3. Query Execution

The generated SQL is executed on your MySQL database.

## 🚦 Performance Comparison

| Provider | Setup Time | Inference Speed | Accuracy | Cost | Offline |
|----------|------------|----------------|----------|------|---------|
| Google Gemini | Fast | Very Fast | High | API costs | No |
| Hugging Face Small | Medium | Fast | Medium | Free | Yes |
| Hugging Face Large | Slow | Medium | High | Free | Yes |

## 🐛 Troubleshooting

### Common Issues

#### 1. Model Download Issues
```python
# Try different model
assistant = HuggingFaceSQLAssistant("microsoft/DialoGPT-small")
```

#### 2. Memory Issues
```python
# Use smaller model or CPU
assistant = HuggingFaceSQLAssistant("microsoft/DialoGPT-small", device="cpu")
```

#### 3. Import Errors
```bash
pip install --upgrade transformers torch accelerate
```

### 4. Database Connection Issues
Ensure your `.env` file has correct database credentials:

```env
DB_USER=root
DB_PASSWORD=your_password
DB_HOST=localhost
DB_NAME=atliq_tshirts
```

## 🔄 Migration Guide

### From Gemini to Hugging Face

1. **Install dependencies** (see Installation section)
2. **Update your code**:
   ```python
   # Old
   chain = get_few_shot_db_chain()

   # New
   chain = get_unified_db_chain(provider="huggingface")
   ```
3. **Test the integration**:
   ```bash
   python langchain_helper.py --both
   ```

## 🎯 Best Practices

1. **Start with smaller models** for testing
2. **Use GPU** if available for better performance
3. **Cache models** locally after first download
4. **Monitor memory usage** with large models
5. **Test thoroughly** before production deployment

## 📚 Advanced Usage

### Custom Few-Shot Examples

Modify `few_shots.py` to add domain-specific examples:

```python
few_shots = [
    {
        'Question': "Custom question for your domain",
        'SQLQuery': "SELECT * FROM your_table WHERE condition",
        'SQLResult': "Result of the SQL query",
        'Answer': "Expected answer"
    },
    # ... existing examples
]
```

### Custom Prompt Engineering

Override the prompt generation:

```python
class CustomHuggingFaceAssistant(HuggingFaceSQLAssistant):
    def create_few_shot_prompt(self, user_question: str, table_info: str = "") -> str:
        # Your custom prompt logic
        custom_prompt = f"Your custom prompt template: {user_question}"
        return custom_prompt
```

## 🤝 Contributing

The integration is designed to be extensible. You can:

1. Add new Hugging Face models
2. Implement custom prompt strategies
3. Add model quantization for better performance
4. Integrate with LangChain's Hugging Face components

## 📄 License

This integration follows the same license as the main project.
