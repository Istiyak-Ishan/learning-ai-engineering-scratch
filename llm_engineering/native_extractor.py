import os
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

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
    sample_reviews = [
        "I bought the Apple AirPods Pro 2. The sound quality is excellent and the noise cancellation works amazingly well. Battery life is also good. I highly recommend them.",

        "The Samsung Galaxy S24 has a beautiful display and the camera takes great photos. However, the battery drains faster than I expected. Still, I am happy with the phone.",

        "I purchased the Logitech MX Master 3S. The mouse feels comfortable and the buttons are responsive. The only problem is that the scroll wheel started making noise after a few weeks.",

        "The Nike Air Max shoes look great and are very comfortable for walking. Unfortunately, the sole started wearing out after only two months. Not worth the price.",

        "I bought this laptop last month. The keyboard is excellent, performance is fast, and the screen is bright. Overall, I am very satisfied with it.",

        "This coffee maker is easy to use and makes great coffee. But the water container is too small and needs to be refilled frequently.",

        "I ordered these headphones but I don't remember the exact model. The sound is decent, but the ear cushions are uncomfortable after long use.",

        "This product is okay. Nothing special about the build quality, but it works as expected. I would say it is neither good nor bad."
    ]

    for review in sample_reviews:
        try:
            result = analyze_review_native(review)

            print("\nReview:")
            print(review)

            print("\nAnalysis:")
            print(json.dumps(result.model_dump(), indent=2))

        except Exception as e:
            print(f"Pipeline failed: {e}")