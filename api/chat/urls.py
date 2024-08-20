from django.urls import path
from . import views

urlpatterns = [
    path('generate-presigned-url/', views.upload_file, name='generate_presigned_url'),
    path('chat/', views.new_chat, name='new_chat'),
    path('chats/', views.get_chats, name='get_chats'),
    path('conversation/', views.talk_with_gpt, name='talk_with_gpt'),
    path('all-chats/', views.get_all_chats, name='get_all_chats'),
]
