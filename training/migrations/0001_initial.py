# Generated by Django 4.0.6 on 2022-07-17 03:49

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Task',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('task_id', models.BigIntegerField(default=0)),
                ('name', models.CharField(max_length=100)),
                ('person', models.CharField(max_length=100)),
                ('description', models.CharField(max_length=300)),
                ('status', models.BooleanField(default=False)),
            ],
        ),
    ]
