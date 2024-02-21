from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from .forms import UX_dementia_prediction
# Create your views here.

def home(request:HttpRequest):
    if request.method == 'POST':
        form = UX_dementia_prediction(request.POST)
        if form.is_valid():
            data = form.cleaned_data['data']
            biased = form.cleaned_data['allow_biased']
            # print(data)
            # print(biased)
            result = data[50:60]
            result = {'az':False, 'FTDbv':True}
            return render(request, 'dementia_widget.html', {'form':form, 'result':result,})
    form = UX_dementia_prediction()
    return render(request, 'dementia_widget.html', {'form':form, 'result':None})

