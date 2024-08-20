from rest_framework import serializers

class ConversationRequestSerializer(serializers.Serializer):
    chat_id = serializers.IntegerField()
    question = serializers.CharField()
