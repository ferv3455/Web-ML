# Generated by Django 4.0.6 on 2022-07-17 05:35

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('training', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='task',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='task',
            name='finished_at',
            field=models.DateTimeField(auto_now=True),
        ),
    ]
