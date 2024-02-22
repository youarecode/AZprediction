from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from .forms import UX_dementia_prediction
from .predictor import DementiaPrediction
# Create your views here.

model = DementiaPrediction()

def home(request:HttpRequest):
    if request.method == 'POST':
        form = UX_dementia_prediction(request.POST)
        if form.is_valid():
            sample = form.cleaned_data['data']
            biased = form.cleaned_data['allow_biased']             
            
            result = model.predict(sample, biased)
    
            return render(request, 'dementia_widget.html', {'form':form, 'result':result,})
        
    form = UX_dementia_prediction()
    return render(request, 'dementia_widget.html', {'form':form, 'result':None})

