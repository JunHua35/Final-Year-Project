from sklearn.cluster import KMeans
import cv2 as cv
from collections import Counter
from PIL import Image
from numpy import asarray
import glob
import os

#ims_path='/media/monash/SSD/MCS1/Mesonet/deepfake_database/train_test/df/' #path
#ims_list=os.walk(ims_path)
# image = [cv.imread(file) for file in glob.glob('/media/monash/SSD/MCS1/Mesonet/deepfake_database/train_test/df*.jpg')]

# img = cv.imread("../MesoNet/deepfake_database/train_test/df/df05030.jpg")   #path of original image when using module 1
# img = cv.imread("../MesoNet/deepfake_database/train_test/df/df05030.jpg")  #path of original image when using module 1.a

# initialized counter for all 3 skin tones
fair_counter = 0
mild_counter = 0
dark_counter = 0

'''
Directory the images will be classified into
'''
classified_dir = "../MesoNet/faceforensics_Dataset_classified/real" 


'''
Source Directory
'''
dir = "../MesoNet/faceforensics_Dataset/train_dataset/train_images/real"
for filename in os.listdir(dir):
    # print(filename[-3::] == ".jpg")
    if(filename[-3::] == "jpg") or (filename[-3::]=="png"):
        # print(filename[-4::])
        f = os.path.join(dir, filename)
        # print(f)
        image = cv.imread(f)
        img = cv.imread(f)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        resized_image = cv.resize(image, (1200, 600))

        def RGB2HEX(color):
            """
            Description: Function to convert RGB to HEX color code
            Input: RGB values
            Output: HEX color code
            """
            return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


        def get_image(image_path):
            """
            Description: Function that reads an image then convert it to another color space, 
            in this case BGR to RGB then output the new image
            Input: Path of an image
            Output: Image converted to RGB color space
            """
            image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            return image

        # convert image to numpy array
        data = asarray(image)


        # create Pillow image
        image2 = Image.fromarray(data)

        # resize and reshape the images accordingly
        modified_image = cv.resize(image, (600, 400), interpolation = cv.INTER_AREA)
        modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

        # compute KMeans clustering with k=2
        clf = KMeans(n_clusters = 2)
        # Compute cluster centers and predict cluster index for each sample
        labels = clf.fit_predict(modified_image)

        counts = Counter(labels)
        
        # Obtain the coordinates of cluster centers
        center_colors = clf.cluster_centers_
        # Obtain ordered colors by iterating through the keys
        ordered_colors = [center_colors[i] for i in counts.keys()]
        hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
        rgb_colors = [ordered_colors[i] for i in counts.keys()]   


        c1=hex_colors[0]
        c2=hex_colors[1]

        d1=c1.lstrip('#')
        d2=c2.lstrip('#')

        res1 = int(d1, 16)
        res2 = int(d2, 16)

        if 8421504>res1>0: #to ignore the darker shade of the background
            fc=res2
            print("The detected skin tone is:",res2," with hex color code as",c2)

        else:
            fc=res1
            print("The detected skin tone is:",res1,"(bg color) with hex color code as",c1)  


        print("The detected skintone is:")
        font = cv.FONT_HERSHEY_TRIPLEX

        if 16777215>fc>12619362: #fair range
            print("Fair")

            # cv.putText(img,'Skin Tone: FAIR',(10,50), font, 1,(0,255,0),2)
            # cv.imshow("Result",img)
            # cv.waitKey(0)
            # cv.imwrite('/media/monash/SSD/MCS1/Skin_Tone_Classifier-main/Fair/Fair.jpg',image)
            # i = 0
            # for imgPath in image:
            cv.imwrite(os.path.join(classified_dir, f'./Fair/Fair_{filename}'),img)
            fair_counter += 1
            # i += 1

        elif 12619362>fc>10300000:    #mild range
            print("Mild")
            
            # cv.putText(img,'Skin Tone: MILD',(10,50), font, 0.5,(0,255,0),2)
            # cv.imshow("Result",img)
            # cv.waitKey(0)
            #cv.imwrite('/media/monash/SSD/MCS1/Skin_Tone_Classifier-main/Mild/Mild.jpg',image)
            # i = 0
            # for imgPath in image:
            cv.imwrite(os.path.join(classified_dir, f'./Mild/Mild_{filename}'),img)
            # i += 1
            mild_counter += 1

        else:
            print("Dark")    
            # cv.putText(img,'Skin Tone: DARK',(10,50), font, 0.5,(0,255,0),2)
            # cv.imshow("Result",img)
            # cv.waitKey(0)
            #cv.imwrite('/media/monash/SSD/MCS1/Skin_Tone_Classifier-main/Dark/Dark.jpg',image)
            # i = 0
            # for imgPath in image:
            cv.imwrite(os.path.join(classified_dir, f'./Dark/Dark_{filename}'),img)
            # i += 1
            dark_counter += 1
    else:
        print("outside")
            

 
