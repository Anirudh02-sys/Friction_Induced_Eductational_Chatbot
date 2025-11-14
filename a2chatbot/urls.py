"""
URL configuration for a2chatbot project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path
from django.contrib.auth import views as auth_views

import a2chatbot.views as views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("register/", views.register, name="register"),
    re_path(r'^$', views.landing, name='landing'),
    path("home/", views.home, name='home'),
    re_path(r'^login$', auth_views.LoginView.as_view(template_name='a2chatbot/login.html'), name= 'login'),
    re_path(r'^sendmessage$', views.sendmessage, name ='sendmessage'),
    re_path(r"^next_question$", views.next_question, name="next_question"),
    path("set_question/<int:idx>/", views.set_question, name="set_question"),
    path("switch_mode/<str:mode>/", views.switch_mode, name="switch_mode"),
]
