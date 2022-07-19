from datetime import datetime, timezone
from itertools import chain
import json
import threading
from time import sleep
from .models import Task
from ml import train


def trainTask(task_id, args):
    json_file = './training/tasks/{}.json'.format(task_id)

    train.train_with_log(file=json_file, **args)

    # Modifying the json file
    with open(json_file, 'r') as fp:
        obj = json.load(fp)
        obj['status'] = True
    with open(json_file, 'w') as fp:
        json.dump(obj, fp)

    # Rewrite database
    record = Task.objects.get(task_id=task_id)
    record.status = True
    record.save()


def getTaskInfo(record):
    try:
        with open('./training/tasks/{}.json'.format(record.task_id), 'r') as fin:
            task = json.load(fin)
            info = {
                'name': record.name,
                'person': record.person,
                'description': record.description,
                'created_at': record.created_at,
                'finished_at': record.finished_at
            }
            return {k: v for k, v in chain(task.items(), info.items())}
    except:
        return None


def searchTask(task_id):
    record = Task.objects.get(task_id=task_id)
    return getTaskInfo(record)


def getAllTasks():
    tasks = Task.objects.order_by('-task_id')
    return [getTaskInfo(r) for r in tasks]


def createTask(form):
    # TODO: use the database
    # 1. add the task into the database (id, taskname, person, description, status)
    info = {
        'task_id': int(datetime.now().timestamp()),
        'name': form['taskname'],
        'person': form['person'],
        'description': form['description']
    }
    task_id = info['task_id']
    new_task = Task(**info)
    new_task.save()

    # 2. save an json file
    json_obj = {
        "task_id": task_id,
        "status": False,
        "log": [],
        "losschart": [],
        "precchart": []
    }
    with open('./training/tasks/{}.json'.format(task_id), 'w') as fout:
        json.dump(json_obj, fout)

    # 3. create a thread to train it (output logs/modify status while training)
    args = {
        'epochs': int(form['epoch']),
        'batch_size': int(form['batch']),
        'learning_rate': float(form['lr']),
        'num_classes': 10,
        'model': form['model'],
        'optimizer': form['optimizer']
    }
    t = threading.Thread(target=trainTask, args=(task_id, args))
    t.start()

    return task_id
