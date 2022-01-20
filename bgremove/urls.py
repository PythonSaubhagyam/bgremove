"""bgremove URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from os import name
from django.contrib import admin
from django.urls import path,include
from app import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.home,name='home'),
    path('index',views.index,name='index'),
    path('bgremove/',views.bgremove,name='bgremove'),
    path('login/',views.login,name='login'),
    path('signup/',views.signup,name='signup'),
    path('edit/',views.edit,name='edit'),
    path('dashboard/',views.DashboardView,name='dashboard'),
    path('album/',views.AlbumView,name='album'),
    path('logout/',views.logout,name='logout'),
    path('paypal/',include('paypal.standard.ipn.urls')),
    path('process_payment/',views.process_payment,name='process_payment'),
    path('payment-done/', views.payment_done, name='payment_done'),
    path('payment-cancelled/', views.payment_canceled, name='payment_cancelled'),
    path('remove/<int:id>/',views.remove,name='remove'),
    path('mailrespond/',views.mailrespond,name='mailrespond'),
    path('aboutus/',views.aboutus,name='aboutus'),
    path('contactus/',views.contactus,name='contactus'),
    path('termsofuse/',views.termsofuse,name='termsofuse'),
    path('privacy/',views.privacy,name='privacy'),
    

]+static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)

