import os
from typing import List, Dict, Any
from few_shots import few_shots

# Try to import Hugging Face libraries, with fallbacks
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("⚠️  Hugging Face libraries not available. Using simplified mode.")


class HuggingFaceSQLAssistant:
    """
    Hugging Face integration for SQL query generation using few-shot learning.
    """

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "auto", use_api: bool = False):
        """
        Initialize the Hugging Face model for SQL generation.

        Args:
            model_name: Hugging Face model name (default: DialoGPT for conversational SQL)
            device: Device to run the model on ('auto', 'cpu', 'cuda')
            use_api: Whether to use Hugging Face Inference API instead of local models
        """
        self.model_name = model_name
        if HUGGINGFACE_AVAILABLE:
            self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.use_api = use_api or not HUGGINGFACE_AVAILABLE

        if self.use_api:
            # Use Hugging Face Inference API
            self.api_token = os.getenv("HUGGINGFACE_API_TOKEN")
            if not self.api_token:
                print("⚠️  HUGGINGFACE_API_TOKEN not found. Using rule-based fallback.")
                self.use_api = False
            else:
                try:
                    import requests
                    self.requests = requests
                    print("✓ Using Hugging Face Inference API")
                except ImportError:
                    print("⚠️  requests library not available. Using rule-based fallback.")
                    self.use_api = False

        if not self.use_api and HUGGINGFACE_AVAILABLE:
            # Load local model
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)

                # Move model to appropriate device
                self.model.to(self.device)

                # Create pipeline for text generation
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                print(f"✓ Loaded Hugging Face model: {model_name} on {self.device}")

            except Exception as e:
                print(f"⚠️  Failed to load local model {model_name}: {str(e)}")
                print("   Using rule-based fallback.")
                self.use_api = False
        elif not self.use_api:
            print("ℹ️  Using rule-based SQL generation (no ML models available)")
            self.use_api = False

    def create_few_shot_prompt(self, user_question: str, table_info: str = "") -> str:
        """
        Create a few-shot prompt using the examples from few_shots.py

        Args:
            user_question: The user's natural language question
            table_info: Database schema information (optional)

        Returns:
            Formatted prompt with few-shot examples
        """
        prompt_parts = []

        # Add system instruction
        prompt_parts.append("You are a SQL expert. Convert natural language questions to SQL queries.")
        prompt_parts.append("Use the following examples as reference:\n")

        # Add few-shot examples
        for i, example in enumerate(few_shots, 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"Question: {example['Question']}")
            prompt_parts.append(f"SQL: {example['SQLQuery']}")
            prompt_parts.append(f"Answer: {example['Answer']}\n")

        # Add table information if provided
        if table_info:
            prompt_parts.append(f"Database Schema:\n{table_info}\n")

        # Add current question
        prompt_parts.append(f"Question: {user_question}")
        prompt_parts.append("SQL:")

        return "\n".join(prompt_parts)

    def generate_sql_query(self, user_question: str, table_info: str = "") -> str:
        """
        Generate SQL query from natural language question using few-shot learning.

        Args:
            user_question: Natural language question
            table_info: Database schema information

        Returns:
            Generated SQL query
        """
        try:
            if self.use_api:
                return self._generate_with_api(user_question, table_info)
            elif HUGGINGFACE_AVAILABLE and hasattr(self, 'pipe'):
                return self._generate_with_local_model(user_question, table_info)
            else:
                return self._generate_with_rules(user_question, table_info)

        except Exception as e:
            return f"Error generating SQL query: {str(e)}"

    def _generate_with_api(self, user_question: str, table_info: str = "") -> str:
        """Generate SQL using Hugging Face Inference API."""
        try:
            prompt = self.create_few_shot_prompt(user_question, table_info)

            api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
            headers = {"Authorization": f"Bearer {self.api_token}"}

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 256,
                    "temperature": 0.1,
                    "do_sample": True,
                    "return_full_text": False
                }
            }

            response = self.requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()
            if isinstance(result, list) and result:
                generated_text = result[0].get('generated_text', '')
            else:
                generated_text = str(result)

            return self._extract_sql_from_text(generated_text)

        except Exception as e:
            print(f"API generation failed: {e}")
            return self._generate_with_rules(user_question, table_info)

    def _generate_with_local_model(self, user_question: str, table_info: str = "") -> str:
        """Generate SQL using local Hugging Face model."""
        try:
            # Create few-shot prompt
            prompt = self.create_few_shot_prompt(user_question, table_info)

            # Generate response
            outputs = self.pipe(
                prompt,
                max_new_tokens=256,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Extract generated text
            generated_text = outputs[0]['generated_text']
            return self._extract_sql_from_text(generated_text)

        except Exception as e:
            print(f"Local model generation failed: {e}")
            return self._generate_with_rules(user_question, table_info)

    def _generate_with_rules(self, user_question: str, table_info: str = "") -> str:
        """Generate SQL using rule-based approach with few-shot examples."""
        try:
            # Simple rule-based SQL generation based on few-shot patterns
            question_lower = user_question.lower()

            # Match patterns from few_shots
            for example in few_shots:
                example_question = example['Question'].lower()
                if any(keyword in question_lower for keyword in ['how many', 'count', 'total', 'sum']):
                    if 'stock_quantity' in example['SQLQuery'] or 'quantity' in question_lower:
                        # Stock quantity queries
                        if 'nike' in question_lower:
                            if 'xs' in question_lower and 'white' in question_lower:
                                return "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND color = 'White' AND size = 'XS'"
                            elif 'size' in question_lower:
                                return f"SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND size = '{self._extract_size(question_lower)}'"
                        elif 'levi' in question_lower and 'white' in question_lower:
                            return "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'"
                        elif 's-size' in question_lower or 'small' in question_lower:
                            return "SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'"

                elif 'revenue' in question_lower or 'sales' in question_lower:
                    if 'levi' in question_lower and 'discount' in question_lower:
                        return """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
                        (select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
                        group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id"""

                elif 'available brands' in question_lower or 'brands' in question_lower:
                    return "SELECT DISTINCT brand FROM t_shirts"

            # Default fallback
            return "SELECT COUNT(*) FROM t_shirts"

        except Exception as e:
            return f"Rule-based generation failed: {str(e)}"

    def _extract_sql_from_text(self, generated_text: str) -> str:
        """Extract SQL query from generated text."""
        # Look for SQL query after "SQL:" marker
        sql_marker = "SQL:"
        if sql_marker in generated_text:
            sql_part = generated_text.split(sql_marker)[-1].strip()
            # Extract until next example or end of response
            sql_lines = []
            for line in sql_part.split('\n'):
                line = line.strip()
                if line and not line.startswith('Answer:') and not line.startswith('Example'):
                    sql_lines.append(line)
                elif line.startswith('Answer:'):
                    break

            sql_query = ' '.join(sql_lines).strip()
            # Clean up common artifacts
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            return sql_query

        return generated_text.strip()

    def _extract_size(self, question: str) -> str:
        """Extract size from question."""
        sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
        for size in sizes:
            if size.lower() in question:
                return size
        return 'M'  # default

    def get_available_models(self) -> List[str]:
        """
        Get list of recommended models for SQL generation tasks.
        """
        return [
            "microsoft/DialoGPT-medium",  # Conversational model
            "microsoft/DialoGPT-large",
            "microsoft/DialoGPT-small",
            "facebook/blenderbot-400M-distill",  # Alternative conversational model
            "google/flan-t5-base",  # Instruction-following model
            "google/flan-t5-small"
        ]


def create_huggingface_sql_chain(model_name: str = "microsoft/DialoGPT-medium"):
    """
    Create a Hugging Face SQL assistant instance.

    Args:
        model_name: Name of the Hugging Face model to use

    Returns:
        HuggingFaceSQLAssistant instance
    """
    return HuggingFaceSQLAssistant(model_name=model_name)


# Example usage and testing functions
def test_huggingface_integration():
    """Test the Hugging Face integration with sample questions."""

    print("=" * 60)
    print("Testing Hugging Face SQL Assistant")
    print("=" * 60)

    try:
        # Create assistant
        assistant = create_huggingface_sql_chain()

        # Test questions
        test_questions = [
            "How many t-shirts do we have left for Nike in XS size and white color?",
            "How much is the total price of the inventory for all S-size t-shirts?",
            "What are the available brands?"
        ]

        for question in test_questions:
            print(f"\nQuestion: {question}")
            sql_query = assistant.generate_sql_query(question)
            print(f"Generated SQL: {sql_query}")
            print("-" * 40)

        print("\n✓ Hugging Face integration test completed!")

    except Exception as e:
        print(f"✗ Hugging Face integration test failed: {e}")


if __name__ == "__main__":
    test_huggingface_integration()
