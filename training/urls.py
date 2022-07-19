from django.urls import path
from . import views

urlpatterns = [
    path('', views.hello, name='home'),
    path('train/', views.train, name='train'),
    path('task/<int:taskid>', views.task, name='task'),
    path('results/', views.results, name='results')
]
