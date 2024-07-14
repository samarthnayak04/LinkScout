from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.process_url_input, name='process_url_input'),
    path('result/', views.url_result, name='url_result'),
]
