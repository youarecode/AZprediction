from django import forms


class UX_dementia_prediction(forms.Form):
    """
    My machine learning model expects data between 100 to 200 char.
    """
    data=forms.CharField(label='',widget=forms.Textarea, min_length=100, max_length=1000)
    allow_biased = forms.BooleanField(label='Por favor permita que usemos el estimador sesgado', initial=True)

    class Meta:
        # label = ('hola')
        pass
        # Incluir el texto por defecto de un ejemplo de cada uno
