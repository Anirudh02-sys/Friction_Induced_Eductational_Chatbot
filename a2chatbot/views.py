from __future__ import unicode_literals

from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.contrib.auth.decorators import login_required
# Used to create and manually log in a user
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate
from a2chatbot.models import *
from django.views.decorators.csrf import ensure_csrf_cookie
from django.contrib.auth.models import User
from django.http import HttpResponse, Http404, JsonResponse
from django.core.files import File
from django.utils import timezone

from openai import OpenAI
import os
import json
import csv
import threading
from dotenv import load_dotenv

from a2chatbot.vectorstore import get_collection, embed_text
from pydantic import BaseModel

load_dotenv() 
# include the api key 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
topic = 'mutation'
question = "what is mutation?"

class CategoryExtraction(BaseModel):
	category: str
	reasoning: str


### HOME
@ensure_csrf_cookie
@login_required
def home(request):
    user = request.user
    participant = get_object_or_404(Participant, user=user)
    if not Assistant.objects.exists():
        initialize_assistant()
    return render(request, "a2chatbot/welcome.html", {"user":user, "question":question})


### CATEGORY PARSER
def get_category(studentmessage: str) -> CategoryExtraction:
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=f"""
Classify the student's reply about mutation.

Valid categories:
- ignorance         (no knowledge or no attempt)
- misconception     (confident but wrong)
- partial_correct   (some truth but incomplete)

Return JSON fields:
- category
- reasoning

Student said: "{studentmessage}"
""",
        text_format=CategoryExtraction
    )
    return response.output_parsed


### RAG PROMPT BUILDER
def get_rag_prompt(studentmessage):
    global_coll = get_collection("global_mutation")
    query_emb = embed_text([studentmessage])
    results = global_coll.query(query_embeddings=query_emb, n_results=3)
    context_passages = results['documents'][0] if results['documents'] else []
    context_text = "\n\n".join(context_passages)

    prompt = f"""
Use the following scientific facts to guide your friction question.
Do not lecture. Respond with ONE short follow-up question only.

Context:
{context_text}

Student said: "{studentmessage}"
"""
    return prompt, context_text



### SEND MESSAGE
@login_required
def sendmessage(request):
    if request.method == "POST":
        user = request.user
        studentmessage = request.POST["message"]

        ### STEP 1: categorize student response
        parsed = get_category(studentmessage)

        if parsed.category == "ignorance":
            need_rag = False
        else:
            need_rag = True

        ### STEP 2: choose prompt
        if need_rag:
            prompt, context_text = get_rag_prompt(studentmessage)
        else:
            context_text = ""   # no RAG → no context used
            prompt = f"""
You are a mutation tutor.

The student said: "{studentmessage}"

You must ask ONE probing follow-up question to elicit more thinking.
NO definitions, NO explanations, SHORT question only.
"""

        ### STEP 3: get correct assistant for that user
        participant = Participant.objects.get(user=user)
        assistant = Assistant.objects.get(level=participant.level)

        thread = client.beta.threads.create(messages=[{"role":"user","content":prompt}])
        run = client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id=assistant.assistant_id, temperature=0.7)
        messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
        bot = messages[0].content[0].text.value

        ### STEP 4: log whole turn
        ChatLog.objects.create(
            user=user,
            message=studentmessage,
            bot_reply=bot,
            context=context_text,
            meta={
                "category": parsed.category,
                "reasoning": parsed.reasoning,
                "used_rag": need_rag
            }
        )

        return JsonResponse([{"bot_message":bot}], safe=False)

def landing(request):
    return render(request, 'a2chatbot/landing.html')

def register(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        level = request.POST.get("level")   # <--- we read level directly

        user = User.objects.create_user(username=username, password=password)
        participant = Participant.objects.create(user=user, level=level)

        login(request, user)
        return redirect('home')  # you already have home()

    return render(request, 'a2chatbot/register.html')


def initialize_assistant():
    # if assistants already exist, don't create again
    if Assistant.objects.exists():
        return

    LEVELS = [
        ("beginner", 
         "You are supportive and encouraging. Use very simple language. Provide hints in tiny steps."),
        ("intermediate", 
         "You are Socratic. Ask probing, clarifying questions. Make the student justify each step."),
        ("advanced", 
         "You are challenging. Push for precise definitions. Make them defend their claims scientifically.")
    ]

    for level, personality_instruction in LEVELS:
        assistant = client.beta.assistants.create(
            name=f"Mutation Tutor ({level})",
            instructions=f"""
You are a mutation tutor helper.
Topic focus = "Mutation".
Personality Mode = {level}

{personality_instruction}

Your job: ask follow-up questions, NOT lectures.
Use friction-based learning. Keep messages short.
""",
            model="gpt-4o-mini",
            temperature=0.7,
        )

        Assistant.objects.create(
            level=level,
            assistant_id=assistant.id,
            video_name=topic,
            vector_store_id=""  # empty for now
        )

    print("=== created 3 assistants ===")


# def delete_agent():
	# include code to delete the agent
	# here're some lines to get you started. You need to figure out the end point of the agent and then delete the assistant. 

	# client.beta.vector_stores.files.delete(
	#     vector_store_id=vector_store.id, file_id=file.id
	# )
	# client.files.delete(file.id)
	# client.beta.vector_stores.delete(vector_store.id)
	# client.beta.assistants.delete(assistant.id)




		
