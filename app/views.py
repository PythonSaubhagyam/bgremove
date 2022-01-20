from django.shortcuts import render,redirect,reverse
from .models import *
from .form import *
from subprocess import run
import requests
import cv2
from django.core.files.storage import FileSystemStorage
from django.core.files.images import get_image_dimensions
from PIL import Image
from paypal.standard.forms import PayPalPaymentsForm
from django.views.decorators.csrf import csrf_exempt
from datetime import date    
from django.contrib import messages
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags


def home(request):
    return render(request,'index.html')

def index(request):
    if 'Free' in request.session:
        user=request.session['Free']
        user_info=Register.objects.get(email=user)
    
        return render(request,'index.html',{'user_info':user_info})
    return redirect('login')

def watermark(path):
    logo = 'media/logo/bglogo.png'
    frontImage = Image.open(logo)
    width,height = frontImage.size
    background = Image.open(path)
    width1, height1 = background.size
    width = round(width1/5)
    height = round(height1/5)
    frontImage = frontImage.resize((width,height))
    frontImage = frontImage.convert("RGBA")
    background = background.convert("RGBA")
    width = (background.width - frontImage.width) //120
    height = (background.height - frontImage.height) // 120
    background.paste(frontImage, (width, height), frontImage)
    file1=path.split("/")[-1]
    img_save_path = "media/"+file1
    background.save(img_save_path, format="png")    
    return img_save_path


def bgremove(request):
    if 'Free' in request.session:
        user=request.session['Free']
        user_info=Register.objects.get(email=user)
        abc=user_info.date.year+1
        end_date=user_info.date.replace(year=abc)
        today = date.today()
        if end_date == today:
            user_info.type_standard=False
            user_info.type_pro=False
            user_info.type_advanced=False
            user_info.save()
                
        if request.method=='POST':
            file1 = request.FILES.getlist('file-picker')
            for i in file1:
                fss = FileSystemStorage()
                file = fss.save(i.name, i)
                file_url = fss.url(file)   
                
                file_url1=file_url.split("/")
                file_url2=file_url1[-2]+"/"+file_url1[-1]
                final_img_path = run(['python', 'app/src2/today.py', '--image', file_url2], capture_output=True)
                output_file_name_path = final_img_path.stdout.strip().decode('utf-8')
                out_file_name_list = output_file_name_path.split("/")
                image_path=output_file_name_path.split(" ")[-1]
                print("image_path:",image_path)
                out_file_name =out_file_name_list[-1]
                if user_info.type_standard==False and (user_info.type_pro==False and user_info.type_advanced==False):
                    print("yes")
                    out_path = watermark(image_path)
                    output_final_name_path=out_path.split("/")[-1]
                    print("output_final_name_path:",output_final_name_path)
                    Original(user=user_info,o_image=i,out_image=output_final_name_path).save()
                else:
                    obj=Original(user=user_info,o_image=i,out_image=out_file_name)
                    obj.save()
                    print("yes")
                    
            return redirect('album')
        return render(request,"index.html")
    else:
        messages.error(request,'Please login first!')
        return redirect('login')



def login(request):
    if 'Free' not in request.session:
        if request.method=='POST':
            email=request.POST['email']
            pass1=request.POST['pass1']
            try: 
                user_info=Register.objects.get(email=email)
                if user_info.pass1==pass1:
                    request.session['Free']=email
                    u =request.session['Free']
                    messages.success(request,f'Welcome, {u}')
                    return redirect('index')
                else:
                    messages.error(request,'Invalid Password')
                    return render(request,'login-sign-up.html')
            except:
                messages.error(request,'Invalid Email id or Username')
                return render(request,'login-sign-up.html')
        return render(request,'login-sign-up.html')
    else:
        return redirect('index')

def signup(request):
    user=Register.objects.all()
    if request.method=='POST':
        email=request.POST['email']
        pass1=request.POST['pass1']
        pass2=request.POST['pass2']
        country=request.POST['country']
        userlist=[]
        for i in user:
            userlist.append(i.email)
        if email in userlist:
            messages.error(request,'User already exist')
            return redirect('signup')
        else:
            if pass1==pass2:
                Register(email=email,pass1=pass1,pass2=pass2,country=country).save()
                messages.info(request,f'{email} - Successfully Registered')
                return redirect('login')
    return render(request,'login-sign-up.html')

def edit(request):
    iid=request.GET.get("iid")
    data=Original.objects.get(id=iid)
    width, height = get_image_dimensions(data.o_image)
    print(data.id)
    print(height, width)
    user=request.session['Free']
    user_info=Register.objects.get(email=user)
    return render(request,'edit.html',{'data':data, 'height':height, 'width':width,'user_info':user_info})




def logout(request):
    if 'Free' in request.session:
        messages.info(request,f'Logged Out')
        del request.session['Free']
    return redirect('/')
    
    
def DashboardView(request):
    if 'Free' in request.session:
        # User Profile Update Start----------------
        user_plan_info=request.session['Free']
        get_user = Register.objects.get(email=user_plan_info)
        print(get_user)
        if request.method == 'POST':
            form = RegisterForm(request.POST,instance =get_user)
            if form.is_valid():
                pass1=form.cleaned_data['pass1']
                pass2=form.cleaned_data['pass2']
                if pass1 == pass2:
                    form.save()
                    messages.info(request,f'{user_plan_info} - Profile Successfully Updated')
                else:
                    messages.error(request,f'{user_plan_info} - Password do not match')
                
                return redirect('/dashboard/')
            else:
                pass
        else:
            form = RegisterForm(instance =get_user)
        # User Profile Update End ----------------
            get_user = Register.objects.get(email=user_plan_info)
        context = {'form':form,'get_user':get_user}
        return render(request,'dashboard.html',context)
    else:
        
        return redirect('/login/')

def AlbumView(request):
    # Get User's Saved Images ----------------
    currunt_user=request.session['Free']
    get_images=Original.objects.filter(user__email=currunt_user)

    get_user = Register.objects.get(email=currunt_user)
    context={'get_images':get_images,'get_user':get_user}
    return render(request,'album.html',context)

def process_payment(request):
    if 'Free' in request.session:
        get_plan = request.GET.get('std')
        request.session['planvalue'] = get_plan
        host = request.get_host()
        print(get_plan,"ggggggggggg")
        paypal_dict = {
            'business': 'lizplatforms@gmail.com',
            'amount': get_plan ,
            'item_name': 'abc',
            'invoice': '12asd12',
            'currency_code': 'USD',
            'notify_url': 'http://{}{}'.format(host,
                                            reverse('paypal-ipn')),
            'return_url': 'http://{}{}'.format(host,
                                            reverse('payment_done')),
            'cancel_return': 'http://{}{}'.format(host,
                                                reverse('payment_cancelled')),
        }

    
        
        form = PayPalPaymentsForm(initial=paypal_dict)
        return render(request, 'process_payment.html', {'form': form})
    else:
        return redirect('login')
    


@csrf_exempt
def payment_done(request):
    get_plan_value =request.session['planvalue']
    user=request.session['Free']
    if get_plan_value == '27':
        get_user = Register.objects.get(email=user)
        get_user.type_standard = True
        get_user.type_pro = False
        get_user.type_advanced = False
        get_user.save()
    elif get_plan_value == '47':
        get_user = Register.objects.get(email=user)
        get_user.type_standard = False
        get_user.type_pro = True
        get_user.type_advanced = False
        get_user.save()
    elif get_plan_value == '67':
        get_user = Register.objects.get(email=user)
        get_user.type_standard = False
        get_user.type_pro = False
        get_user.type_advanced = True
        get_user.save()
    else:
        pass

    return redirect('login')

def remove(request,id):
    if 'Free' in request.session:
        data=Original.objects.get(id=id)
        data.delete()
        return redirect('album')


@csrf_exempt
def payment_canceled(request):
    return render(request, 'payment_cancelled.html')

def mailrespond(request):

    if request.method=='POST':
        iid=request.GET.get('iid')
        file1=Original.objects.get(id=iid)
        file_path=file1.out_image.url
        file_name = file_path.split('/')[-1]
        email=request.POST['email']
        html_con = render_to_string('email1.html',{'abc':file_name})
        html_text = strip_tags(html_con)
        email = EmailMultiAlternatives(
            'test link',
            html_text,
            'photolizshare@gmail.com',
            [email],
            
        )
        email.attach_alternative(html_con,'text/html')
        email.send()
        messages.success(request,f'Email sent successfully')
        return redirect('dashboard')
    return render(request,'email.html')

def aboutus(request):
    return render(request,'aboutus.html')

def contactus(request):
    if request.method=='POST':
        name=request.POST['name']
        email=request.POST['email']
        location=request.POST['location']
        msg=request.POST['msg']
        Contact(name=name,email=email,location=location,message=msg).save()
        return render(request,'contact-us.html',{"error":"Send Successfully"})
    else:
        return render(request,'contact-us.html')

def termsofuse(request):
    return render(request,'terms-of-use.html')

def privacy(request):
    return render(request,'privacy-policy.html')