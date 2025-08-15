from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatbot Backend API")

# --- Define FAQ Dataset and Model ---
# Your initial FAQ dataset with questions and answers
# This will eventually be fetched from a database (MongoDB)
FAQ_DATA = [
    {"question": "Where is my order?", "answer": "To track your order, please provide your order number. You can usually find it in your order confirmation email. We'll then provide you with the latest tracking updates."},
    {"question": "How long will my order take to arrive?", "answer": "Standard delivery typically takes 3-7 business days. You can find more detailed shipping estimates on our Shipping Policy page."},
    {"question": "I didn't receive an order confirmation email.", "answer": "Please check your spam or junk folder. If it's still not there, please provide the email address you used for the purchase, and we can resend it or confirm your order."},
    {"question": "What does 'In-transit' mean?", "answer": "'In-transit' means your package has been dispatched and is currently on its way to the destination. Tracking updates will be provided as it moves through the shipping network."},
    {"question": "What's your return policy?", "answer": "We offer returns within 30 days of purchase for unused items. Please visit our Returns & Exchanges page for detailed instructions on how to initiate a return and get your refund."},
    {"question": "How long does a refund take to process?", "answer": "Once your return is received and inspected, your refund will be processed within 5-7 business days to your original payment method. Please note it may take additional time for your bank to post the refund."},
    {"question": "Can I exchange an item?", "answer": "Yes, we offer exchanges for eligible items. Please see our Returns & Exchanges page for details on how to process an exchange."},
    {"question": "How much is shipping?", "answer": "Shipping costs vary based on your location and the size/weight of your order. You can view the exact shipping cost at checkout before finalizing your purchase. We also offer free shipping on orders over [Threshold]."},
    {"question": "Do you ship internationally?", "answer": "Yes, we ship to many international locations. Please check our Shipping Policy page for a list of countries we deliver to and associated costs."},
    {"question": "Do you have this item in specific size/color?", "answer": "To check product availability, please visit the product page on our website. It will show the current stock levels and available variations (sizes, colors, etc.)."},
    {"question": "What are the material/ingredients of this product?", "answer": "Detailed product information, including materials and ingredients, can be found on the product description page. If you have a specific question, please mention the product name."},
    {"question": "What payment methods do you accept?", "answer": "We accept major credit cards (Visa, Mastercard, Amex), PayPal, and [other relevant payment options like Apple Pay/Google Pay]."},
    {"question": "Do you have any discounts or coupon codes?", "answer": "For current promotions and discount codes, please check our homepage or sign up for our newsletter to receive exclusive offers."},
    {"question": "I have a problem with my order.", "answer": "I'm sorry to hear that. Please provide your order number and a brief description of the issue, and I can either assist or connect you with a human agent for more complex problems."},
    {"question": "How can I contact customer support?", "answer": "You can reach our customer support team via email at [your support email] or by phone at [your support phone number] during business hours. I can also try to answer your question here."}
]

# Load the Sentence-Transformer model globally to avoid reloading on each request
# 'all-MiniLM-L6-v2' is a good balance of performance and size
model = None
faq_embeddings = None
faq_questions = [item["question"] for item in FAQ_DATA]

@app.on_event("startup")
async def load_model():
    """Load the NLP model and pre-compute FAQ embeddings on startup."""
    global model, faq_embeddings
    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Model loaded. Computing FAQ embeddings...")
    faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)
    logger.info(f"FAQ embeddings computed for {len(faq_questions)} questions.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down. Releasing resources.")
    # Add any cleanup here if necessary

# --- API Endpoints ---

class ChatQuery(BaseModel):
    message: str

@app.get("/")
async def read_root():
    """
    Root endpoint that returns basic application information.
    """
    return {
        "message": "Hello from FastAPI backend! Your chatbot is ready.",
        "status": "running",
        "version": "1.0.0",
        "framework": "FastAPI",
        "nlp_model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring application status.
    """
    return {
        "status": "healthy",
        "message": "Application is running properly",
        "nlp_model_status": "loaded" if model else "loading_failed"
    }

@app.post("/chat")
async def chat_with_bot(query: ChatQuery):
    """
    Main chat endpoint to process user queries and return chatbot responses.
    """
    if model is None or faq_embeddings is None:
        logger.error("NLP model not loaded or FAQ embeddings not computed.")
        return {"response": "I'm sorry, the chatbot is still starting up. Please try again in a moment."}

    user_query_embedding = model.encode(query.message, convert_to_tensor=True)

    # Compute cosine-similarity scores
    cosine_scores = util.cos_sim(user_query_embedding, faq_embeddings)[0]

    # Find the best match
    best_match_score = torch.max(cosine_scores).item()
    best_match_idx = torch.argmax(cosine_scores).item()

    # Define a similarity threshold. Adjust this as needed.
    # A score of 0.7 means a reasonably good match. Below that, it might be a poor match.
    SIMILARITY_THRESHOLD = 0.75 

    if best_match_score >= SIMILARITY_THRESHOLD:
        matched_question = faq_questions[best_match_idx]
        response_answer = FAQ_DATA[best_match_idx]["answer"]
        logger.info(f"Matched '{query.message}' to '{matched_question}' with score {best_match_score:.2f}")
        return {
            "query": query.message,
            "response": response_answer,
            "matched_question": matched_question,
            "score": f"{best_match_score:.2f}"
        }
    else:
        logger.info(f"No good match for '{query.message}'. Best score: {best_match_score:.2f}")
        return {
            "query": query.message,
            "response": "I'm sorry, I don't have an answer to that question right now. Could you please rephrase it, or would you like to speak to a human agent?",
            "matched_question": "None",
            "score": f"{best_match_score:.2f}"
        }