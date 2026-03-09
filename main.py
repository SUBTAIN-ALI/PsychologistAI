from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Psychologist AI Assistant API",
    description="Automates lead generation and provides a RAG research assistant.",
    version="1.0.0"
)

app.include_router(router, prefix="/api")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Psychologist AI Assistant is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
