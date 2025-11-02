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

load_dotenv() 
# include the api key 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
topic = 'mutation'
question = "what is mutation?"


userid_list = ['student1', 'student2']

@ensure_csrf_cookie
@login_required
def home(request):
	print("homehome")
	context = {}
	user = request.user
	participant = get_object_or_404(Participant, user=  user)
	context['user'] = user
	print(user.username)

	if not Assistant.objects.filter(user = user).filter(video_name = topic).exists():
		initialize_assistant()
	context["question"] = question
	return render(request, 'a2chatbot/welcome.html', context)


@login_required
def sendmessage(request):
	print("sendmessage")
	user = request.user
	if request.method == "POST":
		print(request.POST)
		context = {}
		studentmessage = request.POST["message"]
		print(studentmessage)

		thread = client.beta.threads.create(
    		messages=[
	        {
	            "role": "user",
	            "content": f"""To facilitate the student's active learning,
	                       you asked them an initial open-ended question:"{question}",
	                       now your goal is to ask a follow-up question based on the studnet's response: "{studentmessage}". Please include the message only and nothing else."""
	        }
    	]
		)

		#an alternative thread that specifies that the agent should respond to the student's question 

		# thread = client.beta.threads.create(
    	# 	messages=[
	    #     {
	    #         "role": "user",
	    #         "content": f"""please respond to the students' questions and answers to facilitate their learning. Please stick with the content of this video: "{studentmessage}". Please include the message only and nothing else."""
	    #     }
    	# ]
		# )

		assistant = get_object_or_404(Assistant, video_name= topic)

		run = client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id=assistant.assistant_id, temperature=0.7)
		messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
		message_content = messages[0].content[0].text
		print(message_content.value)
		response_text = []
		response_text.append({'bot_message':message_content.value})
		ChatLog.objects.create(
			user=user,
			message=studentmessage,
			bot_reply=message_content.value
		)
		response = json.dumps(response_text)
		return HttpResponse(response, 'application/javascript')


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
	assistant = client.beta.assistants.create(
    name="Middle school teacher",
    instructions= f"""
You are a middle school teacher teaching Grades 9-12 who want to assign a learning media video titled "{topic}" to your students for an upcoming science class discussion session. The video transcript is provided as a file search attachment and please review it in its entirety.
""",
    model="gpt-4o-mini",
    temperature=0.7,
)
	#vector_store = client.beta.vector_stores.create(name="video transcripts")

	#file = client.files.create(
    #file=open('mutation.txt', 'rb'), purpose='assistants')
	#file_batch = client.beta.vector_stores.files.create(vector_store_id=vector_store.id, file_id=file.id)
	assistant = client.beta.assistants.update(assistant_id=assistant.id,)
	new_assistant = Assistant(video_name = topic, assistant_id = assistant.id, )
	new_assistant.save()

# def delete_agent():
	# include code to delete the agent
	# here're some lines to get you started. You need to figure out the end point of the agent and then delete the assistant. 

	# client.beta.vector_stores.files.delete(
	#     vector_store_id=vector_store.id, file_id=file.id
	# )
	# client.files.delete(file.id)
	# client.beta.vector_stores.delete(vector_store.id)
	# client.beta.assistants.delete(assistant.id)

def register_new_users():
	for i in range(len(userid_list)):
		user= User.objects.create_user(username=userid_list[i], password = userid_list[i])
		user.save()        
		participant = Participant(user = user)
		participant.save()
	print("new users registered")


		
