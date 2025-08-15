from fastapi import FastAPI
import uvicorn # Needed for local testing, Render uses command line directly

app = FastAPI(title="Chatbot Backend API")

@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI backend! Your chatbot is ready."}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Backend is running"}

# You would typically run this using 'uvicorn main:app --host 0.0.0.0 --port 8000'
# Render will execute this command for you.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) # Local test usage