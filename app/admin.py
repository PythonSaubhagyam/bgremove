from django.contrib import admin

from .models import * 

# admin.site.register(Register)
# admin.site.register(Original)
#admin.site.register(Contact)



@admin.register(Register)
class RegisterAdmin(admin.ModelAdmin):
    list_display = ["type_advanced", "type_pro", "type_standard", "country", "pass2", "pass1", "email"][::-1]


@admin.register(Original)
class OriginalAdmin(admin.ModelAdmin):
    list_display = ["upload_time", "out_image", "o_image", "user"][::-1]

@admin.register(Contact)
class ContactAdmin(admin.ModelAdmin):
    list_display = ["name", "email", "location", "message"]
