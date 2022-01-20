import os
from skimage import io, transform
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from model import U2NETP # small version u2net 4.7 MB
import cv2
import os
import numpy as np
from numpy import asarray
import cv2
from PIL import Image as Img
import argparse

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def save_output(image_name,pred,d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    pb_np = np.array(imo)
    out_img = asarray(pb_np)
    out_img = out_img/255
    THRESHOLD = 0.7
    # refine the output
    out_img[out_img > THRESHOLD] = 1
    out_img[out_img <= THRESHOLD] = 0
    shape = out_img.shape
    a_layer_init = np.ones(shape = (shape[0],shape[1],1))
    mul_layer = np.expand_dims(out_img[:,:,0],axis=2)
    a_layer = mul_layer*a_layer_init
    rgba_out = np.append(out_img,a_layer,axis=2)
    input_img = cv2.imread(image_name)
    inp_img = asarray(input_img)
    inp_img = inp_img/255
    # since the output image is rgba, convert this also to rgba, but with no transparency
    a_layer = np.ones(shape = (shape[0],shape[1],1))
    rgba_inp = np.append(inp_img,a_layer,axis=2)
    # simply multiply the 2 rgba images to remove the backgound
    rem_back = (rgba_inp*rgba_out)
    rem_back_scaled = asarray(rem_back)
    rem_back_scaled = rem_back_scaled * 255
    save_img_name = os.path.splitext(img_name)[0]
    print("save_img_name:",save_img_name)
    final_image_path = save_img_name + '.png'
    print("final_image_path:" ,final_image_path)
    cv2.imwrite(final_image_path,rem_back_scaled)
    return final_image_path


class ImageDataset():
  def __init__(self,img_path,transform):
    self.transform=transform
    self.img_path=img_path

  def __getitem__(self,img_path):
    label = []
    label_name_list = []
    path = self.img_path
     
    image=cv2.imread(path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    label = []
    label_name_list = []
    if(0==len(label_name_list)):
            label_3 = np.zeros(image.shape)
    else:
        label_3 = io.imread(label_name_list[0])

    label = np.zeros(label_3.shape[0:2])
    if(3==len(label_3.shape)):
        label = label_3[:,:,0]
    elif(2==len(label_3.shape)):
        label = label_3

    if(3==len(image.shape) and 2==len(label.shape)):
        label = label[:,:,np.newaxis]
    elif(2==len(image.shape) and 2==len(label.shape)):
        image = image[:,:,np.newaxis]
        label = label[:,:,np.newaxis]
    sample = {'image':image, 'label':label}
    if self.transform:
        sample = self.transform(sample)
    return sample
    
#==========================dataset load==========================
class RescaleT(object):

    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size


    def __call__(self,sample):
        image, label = sample['image'],sample['label']

        h, w = image.shape[:2]
        # print(image.shape)
        if isinstance(self.output_size,int):
            if h > w:
                new_h, new_w = self.output_size*h/w,self.output_size
            else:
                new_h, new_w = self.output_size,self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
        lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

        return {'image':img,'label':lbl}



class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,flag=0):
        self.flag = flag

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        tmpLbl = np.zeros(label.shape)

        if(np.max(label)<1e-6):
            label = label
        else:
            label = label/np.max(label)

        # change the color space
        if self.flag == 2: # with rgb and Lab colors
            # print("\n FLAG2 \n")
            tmpImg = np.zeros((image.shape[0],image.shape[1],6))
            tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
            if image.shape[2]==1:
                tmpImgt[:,:,0] = image[:,:,0]
                tmpImgt[:,:,1] = image[:,:,0]
                tmpImgt[:,:,2] = image[:,:,0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
            tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
            tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
            tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
            tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
            tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
            tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
            tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
            tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

        elif self.flag == 1: #with Lab color
            # print("\n FLAG1\n")
            tmpImg = np.zeros((image.shape[0],image.shape[1],3))

            if image.shape[2]==1:
                tmpImg[:,:,0] = image[:,:,0]
                tmpImg[:,:,1] = image[:,:,0]
                tmpImg[:,:,2] = image[:,:,0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

        else: # with rgb color
            # print("\n ELSE BLOCK")
            tmpImg = np.zeros((image.shape[0],image.shape[1],3))
            image = image/np.max(image)
            if image.shape[2]==1:
                tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
            else:
                tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
                tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

        tmpLbl[:,:,0] = label[:,:,0]

        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='',
                        required=True, help='Inference image filename')
    args = parser.parse_args()
    # --------- 1. get image path and name ---------
    model_name='u2netp'# fixed as u2netp
    prediction_dir = 'media' # changed to 'results' directory which is populated after the predictions
    model_dir = 'app/src2/u2netp.pth' # path to u2netp pretrained weights
    
    # --------- 3. model define ---------
    net = U2NETP(3,1)            
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    net.eval()

    # --------- 4. inference for each image ---------   
    
    path = args.image

    transformed_dataset = ImageDataset(path,transform= transforms.Compose([
                                               RescaleT(320),ToTensorLab(flag=0)
                                           ]))

    for i in range(1):
        sample = transformed_dataset[i]
        input_test = sample['image']
        input_test = input_test.unsqueeze(0)
        input_test = input_test.float()
        d1,d2,d3,d4,d5,d6,d7= net(input_test)

    # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
        
        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(path,pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7
        

if __name__ == "__main__":
    main()
