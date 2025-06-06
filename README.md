# LLM Judge POC 🤖⚖️

A Proof of Concept demonstrating how to use one Large Language Model (LLM) to evaluate and score responses generated by another LLM. This project showcases a powerful pattern for automated evaluation in AI systems.

## 🎯 Overview

This project implements a two-stage LLM pipeline:
- **Content Generation**: Uses a local LLM via Ollama (LLaMA 3.2) to generate answers
- **Evaluation**: Uses Google Gemini to score and provide feedback on the generated content

The system demonstrates how different LLMs can work together, with one acting as a "judge" to evaluate the quality of responses from another.

## 🏗️ Architecture

```
Question → Local LLM (Ollama) → Generated Answer → Gemini Judge → Score + Feedback
```

## ✨ Features

- **Robust JSON Parsing**: Handles malformed responses from LLMs with multiple fallback strategies
- **Pydantic Validation**: Ensures data consistency and type safety
- **Error Handling**: Graceful degradation with informative error messages
- **Configurable Scoring**: Customizable evaluation criteria and scoring rubrics
- **Result Persistence**: Saves evaluation results to JSON files for analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Ollama installed and running
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shelwyn/LLM-as-a-Judge.git
   cd LLM-as-a-Judge
   ```

2. **Install Python dependencies**
   ```bash
   pip install google-generativeai pydantic
   ```

3. **Set up your API key**
   ```bash
   # Replace the placeholder in the code with your actual Gemini API key
   genai.configure(api_key="YOUR_GEMINI_API_KEY_HERE")
   ```

4. **Run the demo**
   ```bash
   python llm_judge.py
   ```

## 🔑 Getting a Google Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the generated API key
5. Replace the placeholder in `llm_judge.py` with your key:
   ```python
   genai.configure(api_key="your_actual_api_key_here")
   ```

> **⚠️ Security Note**: Never commit your API key to version control. Consider using environment variables in production.

## 🛠️ Installing and Using Ollama

Ollama allows you to run large language models locally on your machine. Here's how to get started:

### Windows Installation

1. **Download Ollama**
   - Visit [ollama.ai](https://ollama.ai) and download the Windows installer
   - Run the installer and follow the setup wizard

2. **Verify Installation**
   ```powershell
   ollama --version
   ```

### Linux Installation

1. **Install via curl**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Or install manually**
   ```bash
   # Download the binary
   curl -L https://ollama.ai/download/ollama-linux-amd64 -o ollama
   chmod +x ollama
   sudo mv ollama /usr/local/bin/
   ```

### Using Ollama

1. **Pull a model** (this will download the model)
   ```bash
   ollama pull llama3.2:1b
   ```

2. **Run a model interactively**
   ```bash
   ollama run llama3.2:1b
   ```

3. **List available models**
   ```bash
   ollama list
   ```

4. **Start Ollama service** (if needed)
   ```bash
   ollama serve
   ```

### Available Models

Popular models you can use:
- `llama3.2:1b` - Lightweight, fast (default in this project)
- `llama3.2:3b` - Better quality, moderate size
- `llama3.1:8b` - High quality, larger size
- `codellama` - Specialized for code generation
- `mistral` - Alternative model family

For a complete guide on Ollama installation and usage, check out [this detailed article](https://medium.com/@shelwyncorte/how-to-install-and-use-ollama-on-windows-and-linux-f54c8d87ad94).

## 📊 Usage Examples

### Basic Usage

```python
from llm_judge import evaluate_answer_with_gemini, generate_with_ollama

question = "What is photosynthesis?"
answer = generate_with_ollama(f"Answer briefly: {question}")
evaluation = evaluate_answer_with_gemini(question, answer)

print(f"Score: {evaluation.score}/10")
print(f"Feedback: {evaluation.feedback}")
```

### Custom Evaluation Criteria

You can modify the evaluation prompt to focus on different aspects:

```python
# Example: Focus on creativity and originality
evaluation_prompt = f"""
Evaluate this creative writing sample:
Question: {question}
Student's Answer: {answer}

Criteria:
- Creativity and originality (0-4 points)
- Language and style (0-3 points)
- Coherence and structure (0-3 points)

Respond in JSON format: {{"score": integer, "feedback": "text"}}
"""
```

## 🔧 Configuration

### Changing Models

To use a different Ollama model:

```python
# In generate_with_ollama function
llama_answer = generate_with_ollama(llama_prompt, model_name="llama3.2:3b")
```

### Adjusting Evaluation Criteria

Modify the `evaluate_answer_with_gemini` function to change:
- Scoring criteria
- Point distributions
- Evaluation focus areas

## 📁 Project Structure

```
LLM-as-a-Judge/
├── llm_judge.py           # Main application
├── evaluation_results.json # Generated results (created after running)
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## 🤝 Use Cases

This pattern is useful for:

- **Educational Technology**: Automated grading and feedback systems
- **Content Quality Assessment**: Evaluating generated content quality
- **Model Comparison**: Comparing outputs from different LLMs
- **Training Data Generation**: Creating labeled datasets for model training
- **A/B Testing**: Evaluating different prompting strategies

## 🎛️ Advanced Configuration

### Environment Variables

For production use, set your API key as an environment variable:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

```python
import os
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
```

### Batch Processing

For evaluating multiple responses:

```python
questions_and_answers = [
    ("What is AI?", "AI is artificial intelligence..."),
    ("Explain machine learning", "Machine learning is...")
]

results = []
for question, answer in questions_and_answers:
    evaluation = evaluate_answer_with_gemini(question, answer)
    results.append({
        "question": question,
        "answer": answer,
        "evaluation": evaluation.dict()
    })
```

## 🐛 Troubleshooting

### Common Issues

1. **"Could not parse Gemini response"**
   - The improved JSON extraction should handle most cases
   - Check your internet connection
   - Verify your API key is valid

2. **Ollama connection errors**
   - Ensure Ollama is running: `ollama serve`
   - Check if the model is downloaded: `ollama list`
   - Try pulling the model again: `ollama pull llama3.2:1b`

3. **Timeout errors**
   - Increase timeout in the `generate_with_ollama` function
   - Try a smaller model if your hardware is limited

### Debug Mode

Enable detailed logging by adding:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) for making local LLM execution simple
- [Google AI](https://ai.google.dev) for the Gemini API
- [Pydantic](https://pydantic.dev) for data validation

---

**Made with ❤️ for the AI community**
