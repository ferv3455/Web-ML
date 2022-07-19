from django import forms


class TaskForm(forms.Form):
    taskname = forms.CharField(max_length=100)
    person = forms.CharField(max_length=100)
    description = forms.CharField(max_length=300)

    model = forms.CharField()
    optimizer = forms.CharField()
    lr = forms.FloatField(min_value=0.00000001)
    batch = forms.IntegerField(min_value=1)
    epoch = forms.IntegerField(min_value=1)
