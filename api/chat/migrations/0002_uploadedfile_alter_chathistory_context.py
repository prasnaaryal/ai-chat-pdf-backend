# Generated by Django 5.1 on 2024-08-19 12:33

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='UploadedFile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('filename', models.CharField(max_length=255)),
                ('content', models.BinaryField()),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.AlterField(
            model_name='chathistory',
            name='context',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='chat_context', to='chat.uploadedfile'),
        ),
    ]
