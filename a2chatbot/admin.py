from django.contrib import admin
from .models import ChatLog,Participant,Assistant

admin.site.register(Participant)
admin.site.register(Assistant)
admin.site.register(ChatLog)