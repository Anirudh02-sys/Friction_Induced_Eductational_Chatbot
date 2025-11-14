# 🧬 Mutation Tutor – AIED Assignment 2  
A personalized, RAG-powered educational chatbot that teaches students about genetic mutations using two learning modes: **Tutor-Asks** and **Student-Asks**.

---

## 🌟 Overview
This project implements an intelligent tutoring system that adapts to each student’s background and responses. It uses:

- **OpenAI Assistants API** for multi-turn tutoring conversations  
- **RAG (Retrieval-Augmented Generation)** to pull transcript evidence  
- **Per-student personas** generated at registration  
- **Correctness evaluation** for structured feedback  
- **Two different interaction modes**  
- **Clean UI with markdown rendering & evidence viewer**

---

## 🚀 Features

### **1. Tutor-Asks Mode**
A structured mode where the system:
- Asks the student guided mutation questions  
- Evaluates correctness using a lightweight GPT model  
- Gives hints, explanations, and follow-up questions  
- Tracks progress through question list  
- Uses the student’s personalized tutoring persona  

### **2. Student-Asks Mode**
A free-form chat mode where the student can ask anything.
The system:
- Uses RAG to fetch relevant transcript passages  
- Provides concise explanations with **bold key terms**  
- Always asks a follow-up MCQ or fill-in-the-blank  
- Shows expandable “Transcript Evidence”  

## 🔎 How RAG Works
1. Video transcript is stored in ChromaDB as chunks  
2. Student message → embedded  
3. Top 3 similar chunks are retrieved  
4. Tutor uses this evidence in responses  
5. UI displays it in a collapsible “Transcript Evidence” box  ONLY for the Student-Asks Mode

---

## 📝 Correctness Evaluation (Tutor-Asks)
Uses a small GPT model (`gpt-4o-mini`) to classify each student reply as:

- **correct**  
- **partially correct**  
- **incorrect**  
- **idk**  

This is logged in `ChatLog` for later analysis.

---

## 💻 Running the Project
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver

Then open:  
**http://127.0.0.1:8000**
