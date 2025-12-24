import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from huggingface_helper import HuggingFaceSQLAssistant

load_dotenv()


def _get_api_key() -> str:
    # Read from environment variable loaded via .env
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return api_key


def build_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    # Use your working Gemini model
    return ChatGoogleGenerativeAI(
        google_api_key=_get_api_key(),
        model="gemini-2.5-flash",
        temperature=temperature,
    )


def build_slogan_chain(temperature: float = 0.2):
    """Build a slogan generation chain using prompt and LLM."""
    prompt = PromptTemplate.from_template(
        "Suggest a catchy T-shirt slogan about {topic}."
    )
    llm = build_llm(temperature)
    
    class SimpleChain:
        def __init__(self, llm, prompt):
            self.llm = llm
            self.prompt = prompt
        
        def run(self, inputs: dict):
            formatted_prompt = self.prompt.format(**inputs)
            response = self.llm.invoke(formatted_prompt)
            return response.content if hasattr(response, 'content') else str(response)
    
    return SimpleChain(llm, prompt)


def get_few_shot_db_chain(
    top_k: int = 5,
    temperature: float = 0.1,
    sample_rows_in_table_info: int = 3,
) -> SQLDatabaseChain:
    """
    Connect to the MySQL instance and create a SQLDatabaseChain using Google Gemini.
    Credentials are read from environment variables:
      DB_USER (default: root)
      DB_PASSWORD (default: root)
      DB_HOST (default: localhost)
      DB_NAME (default: atliq_tshirts)
    """
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "root")
    db_host = os.getenv("DB_HOST", "localhost")
    db_name = os.getenv("DB_NAME", "atliq_tshirts")

    uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
    db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=sample_rows_in_table_info)

    llm = build_llm(temperature)
    return SQLDatabaseChain.from_llm(llm, db, top_k=top_k, verbose=True)


def get_huggingface_db_chain(
    model_name: str = "microsoft/DialoGPT-medium",
    sample_rows_in_table_info: int = 3,
    use_api: bool = False,
):
    """
    Connect to the MySQL instance and create a custom chain using Hugging Face models.
    Uses few-shot prompting from few_shots.py for better SQL generation.

    Args:
        model_name: Hugging Face model name to use
        sample_rows_in_table_info: Number of sample rows to include in table info
        use_api: Whether to use Hugging Face Inference API

    Returns:
        Custom HuggingFaceSQLChain instance
    """
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "root")
    db_host = os.getenv("DB_HOST", "localhost")
    db_name = os.getenv("DB_NAME", "atliq_tshirts")

    uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
    db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=sample_rows_in_table_info)

    hf_assistant = HuggingFaceSQLAssistant(model_name=model_name, use_api=use_api)

    class HuggingFaceSQLChain:
        """Custom chain that combines Hugging Face SQL generation with database execution."""

        def __init__(self, db, hf_assistant):
            self.db = db
            self.hf_assistant = hf_assistant

        def __call__(self, question: str):
            """Execute the chain with a question."""
            try:
                # Generate SQL query using Hugging Face
                table_info = str(self.db.table_info)
                sql_query = self.hf_assistant.generate_sql_query(question, table_info)

                # Execute the query
                result = self.db.run(sql_query)

                # Format the response
                response = {
                    'question': question,
                    'sql_query': sql_query,
                    'result': result,
                    'intermediate_steps': [
                        f"Generated SQL: {sql_query}",
                        f"Query Result: {result}"
                    ]
                }

                return response

            except Exception as e:
                return {
                    'question': question,
                    'error': str(e),
                    'result': f"Error executing query: {str(e)}"
                }

        def run(self, question: str):
            """Run method for compatibility with existing code."""
            response = self(question)
            return response.get('result', response.get('error', 'Unknown error'))

    return HuggingFaceSQLChain(db, hf_assistant)


def get_unified_db_chain(
    provider: str = "gemini",
    model_name: str = "microsoft/DialoGPT-medium",
    top_k: int = 5,
    temperature: float = 0.1,
    sample_rows_in_table_info: int = 3,
    use_api: bool = False,
):
    """
    Unified function to get database chain using either Gemini or Hugging Face.

    Args:
        provider: Either "gemini" or "huggingface"
        model_name: Hugging Face model name (only used if provider="huggingface")
        top_k: Number of results to return (only used for Gemini)
        temperature: Temperature for generation
        sample_rows_in_table_info: Sample rows in table info
        use_api: Whether to use Hugging Face API for huggingface provider

    Returns:
        Database chain instance
    """
    if provider.lower() == "huggingface":
        return get_huggingface_db_chain(
            model_name=model_name,
            sample_rows_in_table_info=sample_rows_in_table_info,
            use_api=use_api
        )
    elif provider.lower() == "gemini":
        return get_few_shot_db_chain(
            top_k=top_k,
            temperature=temperature,
            sample_rows_in_table_info=sample_rows_in_table_info
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose 'gemini' or 'huggingface'")


def test_database_connection():
    """Test basic database connection without LLM."""
    try:
        db_user = os.getenv("DB_USER", "root")
        db_password = os.getenv("DB_PASSWORD", "root")
        db_host = os.getenv("DB_HOST", "localhost")
        db_name = os.getenv("DB_NAME", "atliq_tshirts")
        
        uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
        print(f"Testing database connection to: {db_host}/{db_name}")
        
        db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=3)
        print("✓ Database connection successful!")
        print(f"\nDatabase tables: {db.get_usable_table_names()}")
        print(f"\nTable info:\n{db.table_info}")
        
        # Test a simple query
        result = db.run("SELECT COUNT(*) as total FROM t_shirts")
        print(f"\n✓ Test query successful! Total t-shirts: {result}")
        
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


def test_huggingface_chain():
    """Test the Hugging Face database chain."""
    print("=" * 60)
    print("Testing Hugging Face Database Chain")
    print("=" * 60)

    if not test_database_connection():
        print("\nPlease fix database connection issues before testing Hugging Face chain.")
        return False

    try:
        # Test with Hugging Face
        chain = get_huggingface_db_chain()
        question = "How many t-shirts are in the database?"
        print(f"\nQuestion: {question}")
        print("\nGenerating answer...")

        response = chain(question)
        result = response.get('result', response) if isinstance(response, dict) else response
        print(f"Answer: {result}")

        if isinstance(response, dict) and 'sql_query' in response:
            print(f"Generated SQL: {response['sql_query']}")

        # Test another question
        print("\n" + "-" * 60)
        question2 = "What are the available brands?"
        print(f"Question: {question2}")
        response2 = chain(question2)
        result2 = response2.get('result', response2) if isinstance(response2, dict) else response2
        print(f"Answer: {result2}")

        print("\n" + "=" * 60)
        print("✓ Hugging Face database chain test completed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ Hugging Face database chain test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--huggingface":
        # Test only Hugging Face
        test_huggingface_chain()
    elif len(sys.argv) > 1 and sys.argv[1] == "--both":
        # Test both providers
        print("Testing both Gemini and Hugging Face providers...\n")
        print("=" * 80)
        print("GEMINI TESTS")
        print("=" * 80)

        # Test Gemini first
        print("=" * 60)
        print("Testing Database Connection")
        print("=" * 60)

        if test_database_connection():
            print("\n" + "=" * 60)
            print("Testing SQLDatabaseChain with Gemini")
            print("=" * 60)

            try:
                chain = get_few_shot_db_chain()
                question = "How many t-shirts are in the database?"
                print(f"\nQuestion: {question}")
                response = chain(question)
                result = response.get('result', response) if isinstance(response, dict) else response
                print(f"Answer: {result}")

                print("\n" + "=" * 60)
                print("✓ Gemini test completed!")
                print("=" * 60)
            except Exception as e:
                print(f"✗ Gemini test failed: {e}")
        else:
            print("Skipping Gemini tests due to database connection issues.")

        print("\n" + "=" * 80)
        print("HUGGING FACE TESTS")
        print("=" * 80)
        test_huggingface_chain()

    else:
        # Default: Test Gemini (original behavior)
        print("=" * 60)
        print("Testing Database Connection")
        print("=" * 60)

        if test_database_connection():
            print("\n" + "=" * 60)
            print("Testing SQLDatabaseChain with Gemini")
            print("=" * 60)

            try:
                chain = get_few_shot_db_chain()
                question = "How many t-shirts are in the database?"
                print(f"\nQuestion: {question}")
                print("\nGenerating answer...")
                response = chain(question)
                result = response.get('result', response) if isinstance(response, dict) else response
                print(f"\nAnswer: {result}")

                # Test another question
                print("\n" + "-" * 60)
                question2 = "What are the available brands?"
                print(f"Question: {question2}")
                response2 = chain(question2)
                result2 = response2.get('result', response2) if isinstance(response2, dict) else response2
                print(f"Answer: {result2}")

                # Test a more complex question
                print("\n" + "-" * 60)
                question3 = "Show me Nike t-shirts that are available in size L"
                print(f"Question: {question3}")
                response3 = chain(question3)
                result3 = response3.get('result', response3) if isinstance(response3, dict) else response3
                print(f"Answer: {result3}")

                print("\n" + "=" * 60)
                print("✓ All Gemini database query tests completed successfully!")
                print("=" * 60)
            except Exception as e:
                print(f"\n✗ SQLDatabaseChain test failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nPlease fix database connection issues before testing SQLDatabaseChain.")

        print("\n" + "💡 TIP: Use --huggingface to test Hugging Face or --both to test both providers")

