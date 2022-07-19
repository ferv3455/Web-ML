from django.http import HttpResponse
from django.shortcuts import redirect, render

from . import control, forms


def hello(request):
    return render(request, 'index.html')


def train(request):
    if request.method == 'GET':
        return render(request, 'train.html', {'message': None})
    elif request.method == 'POST':
        form = forms.TaskForm(request.POST)
        if not form.is_valid():
            return render(request, 'train.html', {
                'message': '请检查填写的内容！\nPlease check your information.'
            })
        task_id = control.createTask(request.POST)
        return redirect('/task/{}'.format(task_id))


def task(request, taskid):
    record = control.searchTask(taskid)
    if record:
        return render(request, 'task.html', record)
    else:
        return HttpResponse('', status=404)


def results(request):
    return render(request, 'results.html', {'tasks': control.getAllTasks()})
