from django.db import models


count=(('India','India'),('Canada','Canada'),('Australia','Australia'),('Austria','Austria'),('Egypt','Egypt'))
class Register(models.Model):
    class Meta:
        verbose_name = 'User Registration'
        verbose_name_plural = 'User Registrations'

    email = models.CharField(max_length=500)
    pass1=models.CharField(max_length=100)
    pass2=models.CharField(max_length=100)
    country=models.CharField(max_length=200)
    type_standard = models.BooleanField(default=False)
    type_pro = models.BooleanField(default=False)
    type_advanced = models.BooleanField(default=False)
    date=models.DateField(auto_now_add=True)

    def __str__(self):
        return self.email

    def save(self, *args, **kwargs):
        if self.id:
            if self.type_advanced:
                self.type_pro = False
                self.type_standard = False
            elif self.type_pro:
                self.type_advanced = False
                self.type_standard = False
            elif self.type_standard:
                self.type_advanced = False
                self.type_pro = False
        else:
            pass
        super(Register, self).save(*args, **kwargs)

class Original(models.Model):
    class Meta:
        verbose_name = 'Image'
        verbose_name_plural = 'Images'
    user=models.ForeignKey(Register,on_delete=models.CASCADE)
    o_image=models.ImageField(upload_to='original')
    out_image=models.ImageField(upload_to='output')
    upload_time=models.DateTimeField(auto_now_add=True)

class Contact(models.Model):
    name=models.CharField(max_length=200)
    email=models.EmailField()
    location=models.CharField(max_length=200)
    message=models.TextField()
    