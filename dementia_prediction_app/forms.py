from django import forms


class UX_dementia_prediction(forms.Form):
    """
    My machine learning model expects data between 100 to 200 char.
    """
    data=forms.CharField(label='',widget=forms.Textarea, min_length=100, max_length=200)

    class Meta:
        # label = ('hola')
        pass
