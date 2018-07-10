# pca-face-recognition
A simple face recognition demo using PCA algorithm.

# Use KNN classifier

Using the ORL face database.

I set ```k = 90```, and the accuracy reach 92.5%.

```shell
$ python pca_face_recognition.py
s1/10.pgm is the most similar to s1/5.pgm
s2/10.pgm is the most similar to s2/8.pgm
s3/10.pgm is the most similar to s3/9.pgm
s4/10.pgm is the most similar to s4/5.pgm
s5/10.pgm is the most similar to s40/5.pgm
s6/10.pgm is the most similar to s6/6.pgm
s7/10.pgm is the most similar to s7/5.pgm
s8/10.pgm is the most similar to s8/3.pgm
s9/10.pgm is the most similar to s9/5.pgm
s10/10.pgm is the most similar to s8/6.pgm
s11/10.pgm is the most similar to s11/1.pgm
s12/10.pgm is the most similar to s12/9.pgm
s13/10.pgm is the most similar to s13/5.pgm
s14/10.pgm is the most similar to s14/1.pgm
s15/10.pgm is the most similar to s15/2.pgm
s16/10.pgm is the most similar to s16/3.pgm
s17/10.pgm is the most similar to s17/7.pgm
s18/10.pgm is the most similar to s18/5.pgm
s19/10.pgm is the most similar to s19/3.pgm
s20/10.pgm is the most similar to s20/2.pgm
s21/10.pgm is the most similar to s21/8.pgm
s22/10.pgm is the most similar to s22/9.pgm
s23/10.pgm is the most similar to s23/1.pgm
s24/10.pgm is the most similar to s24/9.pgm
s25/10.pgm is the most similar to s25/3.pgm
s26/10.pgm is the most similar to s26/8.pgm
s27/10.pgm is the most similar to s27/1.pgm
s28/10.pgm is the most similar to s28/1.pgm
s29/10.pgm is the most similar to s29/3.pgm
s30/10.pgm is the most similar to s30/2.pgm
s31/10.pgm is the most similar to s31/5.pgm
s32/10.pgm is the most similar to s32/9.pgm
s33/10.pgm is the most similar to s33/2.pgm
s34/10.pgm is the most similar to s34/6.pgm
s35/10.pgm is the most similar to s35/5.pgm
s36/10.pgm is the most similar to s36/6.pgm
s37/10.pgm is the most similar to s37/9.pgm
s38/10.pgm is the most similar to s38/5.pgm
s39/10.pgm is the most similar to s39/6.pgm
s40/10.pgm is the most similar to s5/1.pgm
accuracy: 0.925000
```

# Use SVM classifier

```pca_svm_face_recogition.m``` is a Matlab code which implements a face recognition program using PCA to reduce the dimension of the features and one-vs-one multiclass SVM to classify the image.

I used PCA to reduce the data to 50 dimensions and then use SVM linear kernel function to classify, finally, I got an accuracy of 0.9437.

```
accuracy =

    0.9437
```

Here is the eigen faces:

![eigen_faces.jpg](https://github.com/techping/pca-face-recognition/raw/master/eigen_faces.jpg)