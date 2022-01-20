# import PIL module
from PIL import Image
import argparse
  
def watemark(path):
    # Front Image
    logo = 'D:/Siddhi/bgremove/app/static/bglogo.png'

    # Back Image
    input_image = path
    
    # Open Front Image
    frontImage = Image.open(logo)
    width,height = frontImage.size


    # Open Background Image
    background = Image.open(input_image)
    width1, height1 = background.size
    width = round(width1/5)
    height = round(height1/5)
    frontImage = frontImage.resize((width,height))

    # Convert image to RGBA
    frontImage = frontImage.convert("RGBA")
    
    # # Convert image to RGBA
    background = background.convert("RGBA")


    # # Calculate width to be at the center
    width = (background.width - frontImage.width) //120
    # # Calculate height to be at the center
    height = (background.height - frontImage.height) // 120


    # Paste the frontImage at (width, height)
    background.paste(frontImage, (width, height), frontImage)
    # background.paste(frontImage, (0,0), mask = frontImage)
    # Save this image
    file1=input_image.split("/")[-1]
    print(file1)
    background.save("D:/Siddhi/bgremove/media/"+file1, format="png")
    abc=background
    print(abc.stdout)
    
    return background

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='',
                        required=True, help='Inference image filename')
    args = parser.parse_args()
    path = args.image
    print(path)
    watemark(path)
    

if __name__ == "__main__":
    main()
