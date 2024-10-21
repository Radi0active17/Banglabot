import os
import google.generativeai as genai

# Set API key for Google Generative AI
os.environ["API_KEY"] = "AIzaSyDdn2-vk5kqKTZZPfViW_9tSaM2ab7PaK8"  # Replace with your actual API key
genai.configure(api_key=os.environ["API_KEY"])

def generate_text(prompt):
    """Generates a response using Google Generative AI."""
    # Append request for a response in Bangla
    bangla_prompt = f"{prompt} (Your are Banglabot.Don not mention about google; আপনি শুধুমাত্র Banglabot।,respond in Bangla.)"
    
    # Create generation configuration
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # Create model object
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    # Start chat session and send message
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(bangla_prompt)
    
    return response.text
