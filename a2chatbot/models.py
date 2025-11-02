from __future__ import unicode_literals

from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
from django.utils import timezone


class Participant(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete = models.CASCADE, primary_key=True)
    level = models.CharField(max_length=30, default="beginner")  # beginner / intermediate / advanced
    updated_at = models.DateTimeField(auto_now = True, blank = True)
    def __unicode__(self):
        return 'id='+ str(self.pk)



class Assistant(models.Model):
    LEVEL_CHOICES = [
        ('beginner', 'Beginner'),
        ('intermediate', 'Intermediate'),
        ('advanced', 'Advanced'),
    ]
    level = models.CharField(max_length=30,default="beginner",choices=LEVEL_CHOICES)   # beginner / intermediate / advanced
    assistant_id = models.TextField(verbose_name="Assistant ID")
    video_name = models.CharField(verbose_name="videoname", default='', max_length=100)
    vector_store_id = models.TextField(verbose_name='Vector store ID', blank=True, null=True)

    def __str__(self):
        return f"{self.level} - {self.video_name}"


class ChatLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    bot_reply = models.TextField()
    context = models.TextField(blank=True, null=True) 
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} @ {self.timestamp}"