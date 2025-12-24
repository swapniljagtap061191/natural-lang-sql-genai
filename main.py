import streamlit as st
from langchain_helper import get_unified_db_chain

st.title("Natural Lang 2 SQL: AI‑Driven Query Builder 👕")

# Provider selection
provider = st.sidebar.selectbox(
    "Choose AI Provider:",
    ["gemini", "huggingface"],
    help="Select the AI model provider to use for SQL generation"
)

# Model selection for Hugging Face
if provider == "huggingface":
    # Check if Hugging Face libraries are available
    try:
        import transformers
        hf_available = True
    except ImportError:
        hf_available = False

    if hf_available:
        st.sidebar.success("✅ Hugging Face libraries available")
        model_options = {
            "DialoGPT Medium": "microsoft/DialoGPT-medium",
            "DialoGPT Large": "microsoft/DialoGPT-large",
            "DialoGPT Small": "microsoft/DialoGPT-small",
            "BlenderBot": "facebook/blenderbot-400M-distill"
        }
        use_api = st.sidebar.checkbox("Use Hugging Face API", help="Use online API instead of local models")
        if use_api:
            model_name = "microsoft/DialoGPT-medium"  # API compatible model
        else:
            selected_model = st.sidebar.selectbox(
                "Hugging Face Model:",
                list(model_options.keys()),
                help="Choose the specific Hugging Face model"
            )
            model_name = model_options[selected_model]
    else:
        st.sidebar.warning("⚠️ Hugging Face libraries not available - using rule-based mode")
        st.sidebar.info("To enable full AI models, install Python 3.11 and run: pip install transformers torch accelerate")
        model_name = "rule-based"
        use_api = False
else:
    model_name = "gemini-2.5-flash"  # Default for Gemini
    use_api = False

question = st.text_input("Question: ")

if question:
    try:
        with st.spinner(f"Generating SQL query using {provider.title()}..."):
            if provider == "huggingface":
                chain = get_unified_db_chain(provider=provider, model_name=model_name, use_api=use_api if 'use_api' in locals() else False)
            else:
                chain = get_unified_db_chain(provider=provider)

            response = chain.run(question)

        st.header("Answer")
        st.write(response)

        # Show additional info for Hugging Face
        if provider == "huggingface" and hasattr(chain, 'hf_assistant'):
            with st.expander("See generated SQL query"):
                # Try to get the SQL query from the response
                if isinstance(response, dict) and 'sql_query' in response:
                    st.code(response['sql_query'], language='sql')
                else:
                    st.write("SQL query details not available")

    except Exception as exc:
        st.error(f"Something went wrong: {exc}")
        if provider == "gemini":
            st.info(
                "Verify your database credentials and GOOGLE_API_KEY "
                "environment variable, then try again."
            )
        else:
            st.info(
                "Verify your database credentials. Hugging Face models "
                "run locally and don't require API keys."
            )