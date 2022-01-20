from django import forms
from django.forms import fields, widgets
from .models import Register

class RegisterForm(forms.ModelForm):
    class Meta:
        model = Register
        fields = "__all__"
        exclude = ['email','type_standard','type_pro','type_advanced']
        labels = {
            "email": "Username",
            "pass1": "Password",
            "pass2": "Confirm Password",
        }
        
        widgets = {
            'email':forms.EmailInput(attrs={'class':'form-control'}),
            'pass1':forms.TextInput(attrs={'class':'form-control','placeholder':'Enter Pssword'}),
            'pass2':forms.TextInput(attrs={'class':'form-control','placeholder':'Confirm Pssword'}),
            'country':forms.TextInput(attrs={'class':'form-control'}),
        }