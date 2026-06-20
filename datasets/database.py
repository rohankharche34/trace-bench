import instructor
from pydantic import BaseModel, Field
from google import genai
import json
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

class FactQA(BaseModel):
    memory: str = Field(description="Background context, persona, or established fact about the subject.")
    query: str = Field(description="A user query or question based on the memory.")
    answer: str = Field(description="The correct, detailed answer to the query.")

class DatasetGenerator(BaseModel):
    examples: list[FactQA]

client = instructor.from_provider("google/gemini-2.5-flash")

def generate_synthetic_data(topic: str, count: int = 10) -> str:
    prompt = f"""
    Generate {count} synthetic facts datasets about the topic: {topic}.
    Each dataset must contain:
    1. A 'memory': A factual background or situational context.
    2. A 'query': A specific question testing understanding of that memory.
    3. An 'answer': The precise answer derived directly from the memory.
    """

    response = client.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "You are an expert dataset generator."},
                {"role": "user", "content": prompt}
            ],
            response_model=DatasetGenerator,
            max_retries=3
    )

    return response.model_dump_json(indent=2)

synthetic_data = generate_synthetic_data(topic="random facts on general life, education, philosophy and so on", count=100)
#print(synthetic_data)

# Load the JSON string from the previous step
data_dict = json.loads(synthetic_data)

# Convert to DataFrame
df = pd.DataFrame(data_dict["examples"])

# Save to CSV
df.to_csv("synthetic_facts_data.csv", index=False)
