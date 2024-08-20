from django.db import models
import base64

class UploadedFile(models.Model):
    filename = models.CharField(max_length=255)
    content = models.BinaryField()  # The original binary content (optional)
    text_content = models.TextField()  # The extracted plain text content
    uploaded_at = models.DateTimeField(auto_now_add=True)


class ChatHistory(models.Model):
    title = models.CharField(max_length=255)
    context = models.ForeignKey('UploadedFile', related_name='chat_context', on_delete=models.CASCADE)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "context": self.context.text_content  # Return the plain text content
    }

    def to_dict_with_out_context(self):
        return {
            "id": self.id,
            "title": self.title,
        }

class Chat(models.Model):
    chat_history = models.ForeignKey(ChatHistory, related_name='chats_set', on_delete=models.CASCADE)
    question = models.TextField()
    answer = models.TextField()

    def to_dict(self):
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer
    }
