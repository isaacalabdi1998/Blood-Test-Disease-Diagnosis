# Artifical Intellgince Course Final Project

###### Project Description: Blood Test Disease Diagnosis, This Project Will Determine Whether A Blood Test Sample Has Malaria Or Not.

# 1- Importing Libraries
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
```

# 2- Loading The Image Dataset
```python
path="..\\AIProject\\Dataset\\cell_images"
data=[]
label=[]

for folder in os.listdir(path):
    joined_path=os.path.join(path,folder)
    count=0 # read only up to 6000 images in each folder to reduce data size
    for image in os.listdir(joined_path):
        if count<6000:
            try:
                path_to_img=joined_path+"\\"+image
                img=cv2.resize(cv2.imread(path_to_img),(75,75)) #resize all images into 75 by 75 pixels
                data.append(img)
        
                if folder == "Parasitized":
                    label.append(0) # label is 0 if its infected, 1 otherwise
                else:
                    label.append(1)
                count=count+1
            except Exception as e:
                pass # handle any exception and do nothing
     
        else:
            break

label=np.array(label)
data=np.array(data)
print(data.shape) 
print(label.shape) #should be equal to number of samples read, ie images.
```
######(12000, 75, 75, 3)
######(12000,)



# 3- Data Preprocessing

###### Before Applying Any Algorithm On The Data.Some Steps Must Be Performed First, Which Are :
######    1- the format of the data should be (n_samples,n_features), where the features are the pixels in this case. to do this we need to multiply the height,width and channel together.
######   2- Normalize the feature values to be between 0 and 1
######   3- the data must be split into a training an test set to avoid overfitting and to achieve higher model accuracy

# first, convert data into (n_samples,n_features)
n_samples=len(data)
print("Data Format Before Conversion",data.shape)
data=data.reshape(n_samples,-1)
print("Data Format Before Conversion",data.shape)

Data Format Before Conversion (12000, 75, 75, 3)
Data Format Before Conversion (12000, 16875)

# Normalize Dataset
from sklearn.preprocessing import MinMaxScaler

print(f"Max and Min Value Before Normalization {np.max(data)}, {np.min(data)}")
scaler=MinMaxScaler() # Should be 255 and 0 Before Normalization Process

scaled_data=scaler.fit_transform(data)

print(f"Max and Min Value After Normalization {np.max(scaled_data)}, {np.min(scaled_data)}")
# Should be 1 and 0 After Normalization Process

Max and Min Value Before Normalization 255, 0
Max and Min Value After Normalization 1.0, 0.0

from sklearn.model_selection import train_test_split
# Train Test Split Step, Dataset Will Be Split Into 75% Used For Training, The Other 25% For Testing
x_train,x_test,y_train,y_test=train_test_split(scaled_data,label,test_size=0.25,random_state=100)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
(9000, 16875) (3000, 16875) (9000,) (3000,)


