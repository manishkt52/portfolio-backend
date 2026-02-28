import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# -------------------------
# ENV
# -------------------------

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# -------------------------
# HF CLIENT
# -------------------------

client = InferenceClient(token=HF_TOKEN)

# -------------------------
# FASTAPI APP
# -------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://portfolio-frontend-gumt6ynep-manishkt52s-projects.vercel.app/"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# REQUEST MODEL
# -------------------------

class ChatRequest(BaseModel):
    question: str

# -------------------------
# SESSION INTRO FLAG (simple demo)
# -------------------------

has_introduced = False

# -------------------------
# HEALTH CHECK
# -------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# CHAT ROUTE
# -------------------------

@app.post("/ask")
def ask(payload: ChatRequest):
    global has_introduced

    user_question = payload.question.lower()

    resume_link_html = """
<br/><br/>
<a href="https://portfolio-frontend-gumt6ynep-manishkt52s-projects.vercel.app/resume.pdf" target="_blank"
style="color:#60a5fa;font-weight:500;text-decoration:underline;">
Download Resume
</a>
"""

    system_prompt = """
You are a portfolio chatbot representing Manish.

Talk about:
- projects
- work experience
- skills
- background

Style:
Friendly, professional, simple explanations.
"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload.question}
        ],
        max_tokens=300,
        temperature=0.7,
    )

    ai_answer = response.choices[0].message.content

    # Add intro ONCE
    if not has_introduced:
        ai_answer = "Hi 👋 My name is Manish.\n\n" + ai_answer
        has_introduced = True

    # Show resume only if asked
    if any(word in user_question for word in ["resume", "cv", "download resume"]):
        ai_answer += resume_link_html

    return {"answer": ai_answer}