from __future__ import unicode_literals

import os
import json
from functools import lru_cache

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import login
from django.views.decorators.csrf import ensure_csrf_cookie
from django.http import JsonResponse

from dotenv import load_dotenv
from openai import OpenAI

from a2chatbot.models import Participant, ChatLog
from a2chatbot.vectorstore import get_collection, embed_text

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

topic = "mutation"


@lru_cache(maxsize=1)
def load_ground_truth():
    """
    Load the mutation_qa.json once and cache it.
    """
    fpath = os.path.join(os.path.dirname(__file__), "data", "mutation_qa.json")
    with open(fpath, "r") as f:
        return json.load(f)


# ---------- Persona builder ----------

def build_persona(level, summary):
    """
    Creates a persona prompt for the tutor using student's level + summary.
    Called once at registration.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You generate tutor personas."},
            {
                "role": "user",
                "content": f"""
Create a teaching persona for a mutation tutor.

The student self-rated their understanding as: {level}

The student wrote this summary of the mutation video:
\"\"\"{summary}\"\"\"

Create a short persona (5–7 sentences) describing:
- how the tutor should speak
- how patient/detailed to be
- how Socratic vs explanatory
- how much scientific depth to use
- how to adapt to this level
""",
            },
        ],
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


# ---------- Assistant & thread helpers ----------
def handle_tutor_mode(request, participant, studentmessage):
    qa = load_ground_truth()
    idx = max(0, min(participant.current_q_index, len(qa) - 1))
    main_question = qa[idx]["question"]
    ground_truth = qa[idx]["answer"]

    assistant_id = ensure_assistant(participant)

    if not participant.current_thread_id:
        thread_id = start_thread_for_current_question(participant, main_question, ground_truth)
    else:
        thread_id = participant.current_thread_id

    rag_context = get_rag_context(studentmessage)
    # Step A: Evaluate correctness using the ground truth
    eval_prompt = f"""
    Ground truth answer:
    {ground_truth}

    Student answer:
    "{studentmessage}"

    Classify the student's answer as one of:
    1. correct
    2. partially correct
    3. incorrect
    4. idk (if they say 'I don't know')

    Only output the label.
    """

    eval_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": eval_prompt}],
        max_tokens=10
    )

    print(eval_resp)

    correctness_label = eval_resp.choices[0].message.content.strip().lower()
    user_content = f"""
Main question: {main_question}

The student said:
"{studentmessage}"

Relevant transcript excerpts:
{rag_context}

Your evaluation of correctness:
{correctness_label}

Guidance:
- If 'correct': reinforce positively and add a very short explanation.
- If 'partially correct': praise effort, fix misconceptions, ask a follow-up.
- If 'incorrect': be gentle, break it down, ask a simpler sub-question.
- If 'idk': give a hint or a multiple-choice follow-up.

----------------------------------------------
Your task for THIS turn:
----------------------------------------------
1. Follow Guidance instructions based on correctness
2. Provide a short scaffold:
     - encouragement ( DO NOT display this title )
     - mini-hint OR step-by-step clue
     - short explanation (only if needed)
3. Ask ONE follow-up question:
   - Use either: MCQ, fill-in-the-blank, or short-answer.
4. Keep output well-structured with bold text and bullet points.
5. Stay focused **only** on this main question.
"""
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_content
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=assistant_id,
        temperature=0.7,
    )

    reply = client.beta.threads.messages.list(
        thread_id=thread_id,
        run_id=run.id
    ).data[0].content[0].text.value

    ChatLog.objects.create(
        user=request.user,
        message=studentmessage,
        bot_reply=reply,
        context=rag_context,
        meta={"mode": "tutor_asks", "main_question": main_question,"correctness": correctness_label},
    )

    return JsonResponse([{"bot_message": reply}], safe=False)


def handle_student_mode(request, participant, studentmessage):

    # 1. Ensure assistant exists (but with student-mode instructions)
    assistant_id = ensure_student_mode_assistant(participant)

    # 2. Ensure thread exists
    if not participant.current_thread_id:
        thread_id = start_student_mode_thread(participant)
    else:
        thread_id = participant.current_thread_id

    # 3. Retrieve RAG context
    rag_context = get_rag_context(studentmessage)

    # 4. Message prompt
    user_content = f"""
The student asked:
"{studentmessage}"

Relevant video transcript:
{rag_context}

Your teaching goals:
1. Provide a concise, friendly explanation.
2. Highlight key terms with **bold**.
3. Then ask a follow-up question based on their question.
4. Use either:
   - a short MCQ, or
   - fill-in-the-blank.
5. Encourage the student.

Keep the response SHORT and structured.
"""

    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_content
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=assistant_id,
        temperature=0.7,
    )

    reply = client.beta.threads.messages.list(
        thread_id=thread_id,
        run_id=run.id
    ).data[0].content[0].text.value

    ChatLog.objects.create(
        user=request.user,
        message=studentmessage,
        bot_reply=reply,
        context=rag_context,
        meta={"mode": "student_asks"},
    )

    return JsonResponse([{"bot_message": reply,"rag_context": rag_context}], safe=False)


def get_or_create_participant(user):
    """
    Ensure a Participant row exists for this user.
    (Normally made in register, but this is a safety net.)
    """
    participant, _ = Participant.objects.get_or_create(
        user=user,
        defaults={"level": "beginner", "current_q_index": 0},
    )
    return participant


def create_assistant_for_participant(participant):
    """
    Create an OpenAI assistant using the stored persona.
    Save assistant_id on Participant.
    """
    persona = participant.persona or "You are a patient mutation tutor."

    instructions = f"""
You are a personalized mutation tutor guiding the student through ONE specific question at a time.

Your teaching persona:
{persona}

Student level: {participant.level}

----------------------------------------------
TEACHING BEHAVIOR REQUIREMENTS
----------------------------------------------

1. **Evaluate correctness**
   - If the student’s response is correct → acknowledge, reinforce, and deepen slightly.
   - If partially correct → encourage and give a hint that pushes them one step further.
   - If incorrect → gently point out the misunderstanding and give a scaffolded hint.

2. **Handle “I don’t know”**
   - Respond with: 
     - a simple explanation,
     - a clue,
     - then a very small follow-up question.

3. **Use Socratic scaffolding**
   - Ask small guiding questions before giving explanations.
   - Only reveal the full explanation if the student struggles.

4. **Deliberate practice**
   - Ask 1 follow-up question per turn.
   - Follow-ups should target the specific misconception or missing detail.

5. **Question format variety**
   Mix formats:
      - short-answer prompt  
      - multiple choice  
      - fill-in-the-blank  
      - “choose the correct statement”  

6. **Tone**
   - Supportive, encouraging, patient.
   - Use phrases like “Great attempt!”, “You’re on the right track”, “Let’s think step-by-step.”

7. **Structured output**
   Format your replies using:
      - **bold** for key terms  
      - bullet points  
      - small emojis (✨, 🧬, 🟦) for friendliness  

8. **Stay ONLY on the current main question**
   Do not introduce unrelated concepts unless the student asks.

9. **Detect mastery**
   If the student gives a complete, correct explanation:
      - Provide a brief summary
      - Ask: “Would you like to move to the next question?”

"""

    assistant = client.beta.assistants.create(
        name=f"Mutation Tutor for {participant.user.username}",
        instructions=instructions,
        model="gpt-4o-mini",
        temperature=0.7,
    )

    participant.assistant_id = assistant.id
    participant.save()
    return assistant.id


def ensure_assistant(participant):
    """
    Return a valid assistant_id for this participant, creating if needed.
    """
    if participant.assistant_id:
        return participant.assistant_id
    return create_assistant_for_participant(participant)

def ensure_student_mode_assistant(participant):

    # If an assistant exists but mode is different (tutor), delete it
    if participant.assistant_id and participant.mode != "student_asks":
        try:
            client.beta.assistants.delete(participant.assistant_id)
        except:
            pass
        participant.assistant_id = None
        participant.save()

    # If correct assistant already exists
    if participant.assistant_id:
        return participant.assistant_id

    persona = participant.persona or "You are a friendly mutation tutor."

    instructions = f"""
You are a mutation tutor in STUDENT-ASKS MODE.

Your persona:
{persona}

----------------------------------------------
TEACHING BEHAVIOR RULES
----------------------------------------------

1. **Concise concept explanation**
   - 3–5 short sentences max.
   - Always define key biological terms with **bold**.

2. **Correctness checking**
   - If the student’s question reveals a misconception, correct it gently.

3. **Follow-up question each turn**
   After the explanation, ask ONE:
      - multiple-choice question (MCQ), OR  
      - fill-in-the-blank  

   The follow-up must reinforce the *same concept* the student asked about.

4. **Encouragement**
   Use friendly, motivating tone:
   - “Good question!”  
   - “Nice thinking—let’s explore that.”  

5. **I don’t know handling**
   If student shows confusion:
      - Give a simple explanation
      - Ask an easy MCQ to rebuild confidence

6. **Structure**
   Use:
     - **bold terms**
     - bullet points
     - 1 emoji max per message
"""

    assistant = client.beta.assistants.create(
        name=f"Mutation Tutor (Student Mode) for {participant.user.username}",
        instructions=instructions,
        model="gpt-4o-mini",
        temperature=0.7,
    )

    participant.assistant_id = assistant.id
    participant.save()

    return assistant.id


def start_student_mode_thread(participant):
    thread = client.beta.threads.create(
        messages=[
            {"role": "user", "content": "You are now in student-asks mode. Begin teaching."}
        ]
    )
    participant.current_thread_id = thread.id
    participant.save()
    return thread.id


def start_thread_for_current_question(participant, main_question, ground_truth):
    """
    Create a thread that is specific to the current question.
    The first message tells the assistant which question we are focusing on
    and what the ground truth is (for internal reference).
    """
    content = f"""
You are now focusing on this main question:

Q: {main_question}

Ground-truth (for your internal reference only; do NOT just dump this as an answer):
{ground_truth}

Your job:
- Use this ground truth to judge the student's understanding.
- Ask good questions, give hints, and explain when they are stuck.
- Stay on this question until the student is done.
"""

    thread = client.beta.threads.create(
        messages=[
            {"role": "user", "content": content}
        ]
    )

    participant.current_thread_id = thread.id
    participant.save()
    return thread.id


def get_or_create_thread(participant, main_question, ground_truth):
    """
    Ensure there's a thread for the current question.
    If not, create a new one.
    """
    if participant.current_thread_id:
        return participant.current_thread_id
    return start_thread_for_current_question(participant, main_question, ground_truth)


# ---------- RAG helper ----------

def get_rag_context(studentmessage):
    """
    Always retrieve some transcript chunks related to the student's message.
    """
    global_coll = get_collection("global_mutation")
    query_emb = embed_text([studentmessage])
    results = global_coll.query(query_embeddings=query_emb, n_results=3)
    context_passages = results["documents"][0] if results["documents"] else []
    context_text = "\n\n".join(context_passages)
    return context_text


# ---------- VIEWS ----------

@ensure_csrf_cookie
@login_required
def home(request):
    user = request.user
    participant = get_or_create_participant(user)

    qa = load_ground_truth()
    idx = max(0, min(participant.current_q_index, len(qa) - 1))
    main_question = qa[idx]["question"]

    return render(
        request,
        "a2chatbot/welcome.html",
        {
            "user": user,
            "all_questions": qa,
            "current_question": main_question,
            "current_index": idx,
            "mode": participant.mode,   
        },
    )


@login_required
def sendmessage(request):
    if request.method == "POST":
        user = request.user
        participant = get_or_create_participant(user)

        mode = participant.mode
        studentmessage = request.POST["message"]

        # Branch on mode
        if mode == "tutor_asks":
            return handle_tutor_mode(request, participant, studentmessage)
        else:
            return handle_student_mode(request, participant, studentmessage)



def landing(request):
    return render(request, "a2chatbot/landing.html")


def register(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        level = request.POST.get("level")
        summary = request.POST.get("summary", "").strip()

        user = User.objects.create_user(username=username, password=password)

        # Build persona once
        persona_text = build_persona(level, summary)
        participant = Participant.objects.create(
            user=user,
            level=level,
            persona=persona_text,
            current_q_index=0,
            assistant_id = None,
            current_thread_id=None
        )
        login(request, user)
        return redirect("home")

    return render(request, "a2chatbot/register.html")

@login_required
def switch_mode(request, mode):
    participant = get_or_create_participant(request.user)

    if mode in ["tutor_asks", "student_asks"]:
        participant.mode = mode
        # reset assistant & thread for clean mode switching
        if participant.assistant_id:
            try:
                client.beta.assistants.delete(participant.assistant_id)
            except:
                pass
        participant.assistant_id = None

        if participant.current_thread_id:
            try:
                client.beta.threads.delete(participant.current_thread_id)
            except:
                pass
        participant.current_thread_id = None

        participant.save()

    return redirect("home")

@login_required
def next_question(request):
    user = request.user
    participant = get_or_create_participant(user)

    # Delete assistant
    if participant.assistant_id:
        try:
            client.beta.assistants.delete(participant.assistant_id)
        except Exception as e:
            print("[WARN] Failed to delete assistant:", e)
        participant.assistant_id = None

    # Delete thread
    if participant.current_thread_id:
        try:
            client.beta.threads.delete(participant.current_thread_id)
        except Exception as e:
            print("[WARN] Failed to delete thread:", e)
        participant.current_thread_id = None

    # Move to next question
    qa = load_ground_truth()
    if participant.current_q_index < len(qa) - 1:
        participant.current_q_index += 1
    # else remain at last question

    participant.save()
    return redirect("home")

@login_required
def set_question(request, idx):
    user = request.user
    participant = get_or_create_participant(user)

    qa = load_ground_truth()

    # validate index
    if 0 <= idx < len(qa):
        participant.current_q_index = idx

        # delete assistant (fresh start per question)
        if participant.assistant_id:
            try:
                client.beta.assistants.delete(participant.assistant_id)
            except:
                pass
            participant.assistant_id = None

        # delete thread
        if participant.current_thread_id:
            try:
                client.beta.threads.delete(participant.current_thread_id)
            except:
                pass
            participant.current_thread_id = None

        participant.save()

    return redirect("home")
