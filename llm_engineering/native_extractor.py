import os
import json
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# 1. Define the Schema (Structured Output Contract)
class ProductFeature(BaseModel):
    feature_name: str = Field(description="The name of the feature mentioned")
    sentiment: str = Field(description="Positive, Negative, or Neutral")

class ReviewAnalysis(BaseModel):
    product_name: str
    is_recommended: bool
    features_mentioned: List[ProductFeature]
    reasoning: str = Field(description="Step-by-step reasoning for the extraction")

def analyze_review_native(review_text: str) -> ReviewAnalysis:
    """Extracts structured data using the Chats API and gemini-3.6-flash."""
    
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    
    client = genai.Client()

    system_instruction = """
    You are a Senior Data Analyst at an e-commerce company specializing in NLP extraction.
    Your task is to analyze user reviews and extract structured data.
    
    Rules you must follow:
    - ALWAYS extract the specific product name. If unknown, output 'Unknown'.
    - NEVER guess a feature if it is not explicitly mentioned in the text.
    - Think step-by-step in the 'reasoning' field before populating the other fields.
    """

    # 3. Model Configuration: Using the Chats interface to clear the AFC warning
    # We pass the schema config when creating the chat session.
    chat = client.chats.create(
        model="gemini-3.6-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=ReviewAnalysis,
            temperature=0.2,
        )
    )

    # Send the message through the chat session
    response = chat.send_message(
        f"Extract information from this review:\n\n<review>{review_text}</review>"
    )

    # The output text is guaranteed to be JSON matching our schema
    return ReviewAnalysis.model_validate_json(response.text)

# --- Execution ---
if __name__ == "__main__":
    sample_review = "I bought the Sony WH-1000XM5 yesterday. The noise cancellation is absolutely mind-blowing, but the headband feels a bit flimsy compared to the older model. Overall, I'd say it's worth the $350, definitely keeping them."
    
    try:
        result = analyze_review_native(sample_review)
        print("Native SDK Result:")
        print(json.dumps(result.model_dump(), indent=2))
    except Exception as e:
        print(f"Pipeline failed: {e}")