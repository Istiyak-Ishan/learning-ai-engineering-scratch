import os
import json
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

print("API key loaded:", bool(os.getenv("GEMINI_API_KEY")))

# 1. Define the Schema (Structured Output Contract)
class ProductFeature(BaseModel):
    feature_name: str = Field(description="The name of the feature mentioned")
    sentiment: str = Field(description="Positive, Negative, or Neutral")

class ReviewAnalysis(BaseModel):
    product_name: str
    is_recommended: bool
    features_mentioned: List[ProductFeature]
    reasoning: str = Field(description="Step-by-step reasoning for the extraction")

def analyze_review_langchain(review_text: str) -> ReviewAnalysis:
    """Extracts structured data using LangChain and gemini-3.6-flash."""
    
    if not os.environ.get("GEMINI_API_KEY"):
         raise ValueError("GEMINI_API_KEY environment variable not set.")

    # 2. Initialize the LLM with the correct model version
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash", 
        temperature=0.2
    )
    
    # 3. Bind the Pydantic schema to the LLM
    structured_llm = llm.with_structured_output(ReviewAnalysis)

    # 4. Prompt Engineering: Templates
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Senior Data Analyst at an e-commerce company. "
            "Extract structured data from the provided review.\n\n"
            "Constraints:\n"
            "1. ALWAYS extract the specific product name. If unknown, output 'Unknown'.\n"
            "2. NEVER guess a feature if it is not explicitly mentioned.\n"
            "3. Use the reasoning field to analyze the text before categorizing."
        )),
        ("user", "Extract information from this review:\n\n<review>{review}</review>")
    ])

    # 5. Create the Chain and Invoke
    chain = prompt | structured_llm
    
    return chain.invoke({"review": review_text})

# --- Execution ---
if __name__ == "__main__":
    sample_review = "I bought the Sony WH-1000XM5 yesterday. The noise cancellation is absolutely mind-blowing, but the headband feels a bit flimsy compared to the older model. Overall, I'd say it's worth the $350, definitely keeping them."
    
    try:
        result = analyze_review_langchain(sample_review)
        print("\nLangChain Result:")
        print(json.dumps(result.model_dump(), indent=2))
    except Exception as e:
        print(f"Pipeline failed: {e}")