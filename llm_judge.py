import subprocess
import google.generativeai as genai
import json
import re
from typing import Optional
from pydantic import BaseModel, Field, ValidationError

# === Pydantic Models ===
class EvaluationResult(BaseModel):
    score: int = Field(..., ge=0, le=10, description="Score between 0 and 10")
    feedback: str = Field(..., min_length=1, description="Brief feedback text")

# === JSON Extraction Function ===
def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Extract JSON from text that might contain markdown formatting or other content.
    
    Args:
        text: Raw text that might contain JSON
        
    Returns:
        Dictionary if JSON found and valid, None otherwise
    """
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Try to find JSON-like content using regex
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Basic nested braces pattern
        r'\{[\s\S]*\}',  # Any content between outermost braces
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                # Clean up the match
                cleaned_match = match.strip()
                return json.loads(cleaned_match)
            except json.JSONDecodeError:
                continue
    
    # If regex fails, try to find lines that look like JSON
    lines = text.split('\n')
    json_lines = []
    in_json = False
    brace_count = 0
    
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('{'):
            in_json = True
            brace_count = stripped_line.count('{') - stripped_line.count('}')
            json_lines.append(stripped_line)
        elif in_json:
            json_lines.append(stripped_line)
            brace_count += stripped_line.count('{') - stripped_line.count('}')
            if brace_count <= 0:
                break
    
    if json_lines:
        try:
            json_text = ' '.join(json_lines)
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass
    
    return None

def parse_evaluation_response(response_text: str) -> EvaluationResult:
    """
    Parse Gemini response and return validated EvaluationResult.
    
    Args:
        response_text: Raw response from Gemini
        
    Returns:
        EvaluationResult object
        
    Raises:
        ValueError: If response cannot be parsed or validated
    """
    # Try to extract JSON from the response
    json_data = extract_json_from_text(response_text)
    
    if json_data is None:
        raise ValueError(f"Could not extract valid JSON from response: {response_text}")
    
    try:
        # Validate using Pydantic
        return EvaluationResult(**json_data)
    except ValidationError as e:
        raise ValueError(f"Invalid evaluation format: {e}")

# === Step 1: Set up Gemini API ===
genai.configure(api_key="<GEMINI API KEY>")  # Replace with your actual API key
model = genai.GenerativeModel("gemini-2.0-flash")

# === Step 2: Define the question ===
question = "Explain the water cycle in 2-3 sentences."

# === Step 3: Generate answer using local Ollama (LLaMA3) ===
def generate_with_ollama(prompt, model_name="llama3.2:1b"):
    """Generate response using Ollama local LLM."""
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt.encode(),
            capture_output=True,
            timeout=30  # Add timeout to prevent hanging
        )
        if result.returncode != 0:
            raise RuntimeError(f"Ollama failed with return code {result.returncode}")
        return result.stdout.decode("utf-8").strip()
    except subprocess.TimeoutExpired:
        raise RuntimeError("Ollama request timed out")
    except Exception as e:
        raise RuntimeError(f"Error running Ollama: {e}")

# === Step 4: Improved evaluation with better prompt ===
def evaluate_answer_with_gemini(question: str, answer: str) -> EvaluationResult:
    """
    Evaluate an answer using Gemini and return structured result.
    
    Args:
        question: The original question
        answer: The student's answer to evaluate
        
    Returns:
        EvaluationResult object with score and feedback
    """
    evaluation_prompt = f"""You are a fair and objective grader. Evaluate the following student's answer.

Question: {question}
Student's Answer: {answer}

Evaluation Criteria:
- Relevance to the question (0-4 points)
- Scientific accuracy (0-3 points)  
- Clarity and completeness (0-3 points)

IMPORTANT: Respond with ONLY a JSON object in this exact format:
{{
    "score": [integer from 0 to 10],
    "feedback": "[brief feedback explaining the score]"
}}

Do not include any other text, explanations, or markdown formatting."""

    try:
        response = model.generate_content(evaluation_prompt)
        return parse_evaluation_response(response.text)
    except Exception as e:
        # Fallback evaluation if Gemini fails
        return EvaluationResult(
            score=0,
            feedback=f"Error evaluating response: {str(e)}"
        )

# === Main execution ===
def main():
    try:
        print("🔄 Generating answer with Ollama...")
        llama_prompt = f"Answer the following question briefly:\n\nQuestion: {question}"
        llama_answer = generate_with_ollama(llama_prompt)
        print(f"✅ Generated Answer (LLaMA3):\n{llama_answer}\n")
        
        print("🔄 Evaluating with Gemini...")
        evaluation = evaluate_answer_with_gemini(question, llama_answer)
        
        print("✅ Gemini Evaluation:")
        print(f"Score: {evaluation.score}/10")
        print(f"Feedback: {evaluation.feedback}")
        
        # Optional: Save results to file
        results = {
            "question": question,
            "generated_answer": llama_answer,
            "evaluation": {
                "score": evaluation.score,
                "feedback": evaluation.feedback
            }
        }
        
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\n💾 Results saved to evaluation_results.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
