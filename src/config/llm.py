# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.set import settings

# Primary LLM
llm1 = ChatGoogleGenerativeAI(
    api_key=settings.GOOGLE_API_KEY1,
    model=settings.GOOGLE_MODEL_NAME,
    temperature=0
)

# Backup LLM
llm2 = ChatGoogleGenerativeAI(
    api_key=settings.GOOGLE_API_KEY2,
    model=settings.GOOGLE_MODEL_NAME,
    temperature=0
)

# Create a fallback chain: try llm1, if it fails (e.g. rate limit), try llm2
llm_with_fallback = llm1.with_fallbacks([llm2])

def get_llm():
    return llm_with_fallback

def get_structured_llm(schema):
    # Apply structured output to both models independently, then combine them with fallbacks
    structured_llm1 = llm1.with_structured_output(schema)
    structured_llm2 = llm2.with_structured_output(schema)
    return structured_llm1.with_fallbacks([structured_llm2])