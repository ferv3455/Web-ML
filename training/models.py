from django.db import models

# Create your models here.


class Task(models.Model):
    task_id = models.BigIntegerField(default=0)
    name = models.CharField(max_length=100)
    person = models.CharField(max_length=100)
    description = models.CharField(max_length=300)
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(auto_now=True)
    status = models.BooleanField(default=False)  # finished=True
