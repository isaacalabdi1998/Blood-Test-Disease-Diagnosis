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
###### -(12000, 75, 75, 3)
###### -(12000,)



# 3- Data Preprocessing

###### Before Applying Any Algorithm On The Data.Some Steps Must Be Performed First, Which Are :
######    1- the format of the data should be (n_samples,n_features), where the features are the pixels in this case. to do this we need to multiply the height,width and channel together.
######   2- Normalize the feature values to be between 0 and 1
######   3- the data must be split into a training an test set to avoid overfitting and to achieve higher model accuracy
```python
# first, convert data into (n_samples,n_features)
n_samples=len(data)
print("Data Format Before Conversion",data.shape)
data=data.reshape(n_samples,-1)
print("Data Format Before Conversion",data.shape)
```

###### Data Format Before Conversion (12000, 75, 75, 3)
###### Data Format Before Conversion (12000, 16875)

```python
# Normalize Dataset
from sklearn.preprocessing import MinMaxScaler

print(f"Max and Min Value Before Normalization {np.max(data)}, {np.min(data)}")
scaler=MinMaxScaler() # Should be 255 and 0 Before Normalization Process

scaled_data=scaler.fit_transform(data)

print(f"Max and Min Value After Normalization {np.max(scaled_data)}, {np.min(scaled_data)}")
# Should be 1 and 0 After Normalization Process
```
###### Max and Min Value Before Normalization 255, 0
###### Max and Min Value After Normalization 1.0, 0.0

```python
from sklearn.model_selection import train_test_split
# Train Test Split Step, Dataset Will Be Split Into 75% Used For Training, The Other 25% For Testing
x_train,x_test,y_train,y_test=train_test_split(scaled_data,label,test_size=0.25,random_state=100)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
```
###### (9000, 16875) (3000, 16875) (9000,) (3000,)


# 4- Model Selection

###### Multiple Algorithms Will Be Implemented, Including :

######   1- Random Forest Classifier
######   2- K-Nearest Neighbor
######   3- Logistic Regression
######   4- Support Vector Machine

### Random Forest Model

```python
from sklearn.ensemble import RandomForestClassifier
rf_clf=RandomForestClassifier(n_estimators=200,random_state=100)
rf_clf.fit(x_train,y_train)
rf_y_pred=rf_clf.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
rf_score=accuracy_score(y_test,rf_y_pred)
score
```
###### 0.8263333333333334

```python
# What is a Confusion Matrix? https://www.w3schools.com/python/python_ml_confusion_matrix.asp
rf_cm=confusion_matrix(y_test,rf_y_pred)
show_matrix=ConfusionMatrixDisplay(confusion_matrix=rf_cm,display_labels=[False,True])
show_matrix.plot()
plt.show()
```

![Confusion Matrix](https://github.com/isaacalabdi1998/Blood-Test-Disease-Diagnosis/raw/main/Dataset/Confusion%20Matrix.png)


### K-Nearest Neighbors Model
```python
from sklearn.neighbors import KNeighborsClassifier
knn_clf=KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(x_train,y_train)
knn_y_pred=knn_clf.predict(x_test)

knn_score=accuracy_score(y_test,knn_y_pred)
knn_score
```

###### 0.6196666666666667

```python
knn_cm=confusion_matrix(y_test,knn_y_pred)
show_matrix=ConfusionMatrixDisplay(confusion_matrix=knn_cm,display_labels=[False,True])

show_matrix.plot()
plt.show()
```

![Confusion Matrix](https://github.com/isaacalabdi1998/Blood-Test-Disease-Diagnosis/raw/main/Dataset/1_Confusion%20Matrix.png)


### Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression
lr_clf=LogisticRegression(solver='saga') # The 'Saga' Solver is Most Appropiate For Large Datasets
lr_clf.fit(x_train,y_train)
lr_y_pred=lr_clf.predict(x_test)

lr_score=accuracy_score(y_test,lr_y_pred)
lr_score



lr_cm=confusion_matrix(y_test,lr_y_pred)
show_matrix=ConfusionMatrixDisplay(confusion_matrix=lr_cm,display_labels=[False,True])

show_matrix.plot()
plt.show()
```
![Confusion Matrix](https://github.com/isaacalabdi1998/Blood-Test-Disease-Diagnosis/raw/main/Dataset/2_Confusion%20Matrix.png)



### Support Vector Machine Model

```python
from sklearn.svm import SVC
svm_clf=SVC(kernel='linear')
svm_clf.fit(x_train,y_train)
svm_y_pred=svm_clf.predict(x_test)

svm_score=accuracy_score(y_test,svm_y_pred)
svm_score
```
```python
svm_cm=confusion_matrix(y_test,svm_y_pred)
show_matrix=ConfusionMatrixDisplay(confusion_matrix=svm_cm,display_labels=[False,True])

show_matrix.plot()
plt.show()
```
![Confusion Matrix](https://github.com/isaacalabdi1998/Blood-Test-Disease-Diagnosis/raw/main/Dataset/3_Confusion%20Matrix.png)



# 4- Model Selection

### Since The Random Forest Model Is The Best Performing Model, This Model Will Be Chosen.

```python
# A Final Test

choice=np.random.randint(len(x_test)) # choose a test image at random
sample_img=x_test[choice].reshape(75,75,3)

cv2.imshow("img",sample_img)
cv2.waitKey()

result=rf_clf.predict(x_test[choice].reshape(1,-1))
print("Infected With Malaria" if not result[0] else "Not Infected")
```


# 5- Final Thoughts And Some Considerations

###### - The Random Forest Model Can Be Used To Obtain Accurate Results, But To Increase The Accuracy Of The Model Even Further, There Are A Few Thing That Can Be Done, Such As :
###### - Applying Principal Component Analysis To Reduce Feature Size And To Obtain Only The Most Important Pixel Information
###### - Hyperparameter Tuning
###### - Increase Sample Size

# END OF PROJECT

