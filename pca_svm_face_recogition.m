%% File:   pca_svm_face_recognition.m
%% Author: Ziping Chen
%% Date:   2018-07-10

clear; clc; close all;
photo_each_person = 6;
people = 40;

img_data = zeros(112 * 92, people * photo_each_person);
for i=1:people
    for j=1:photo_each_person
        addr = strcat('orl_faces/s', num2str(i), '/', num2str(j), '.pgm');
        img = imread(addr);
        img = img(1:112 * 92);
        img_data(:, photo_each_person * (i - 1) + j) = img';
    end
end

img_avg = mean(img_data, 2);
avg_face = reshape(img_avg, 112, 92);
figure;
subplot(1, 1, 1);
imshow(avg_face, []);
title(strcat('avg face'));

temp = repmat(img_avg, 1, people * photo_each_person);
img_min = img_data - temp;

figure;
subplot(1, 1, 1);
imshow(reshape(img_min(:, 1), 112, 92), []);
title(strcat('min face'));

k = 50;
w = (img_min' * img_min);
%w = cov(img_data);
[V, ~] = eigs(w, k);
V = img_min * V;
VT = fliplr(V);

figure;
for i=1:50
    v = VT(:, i);
    eig_face = reshape(v, 112, 92);
    subplot(5, 10, i);
    imshow(eig_face, []);
    title(strcat('eigen face ', num2str(i)));
end

feature_train = img_min' * VT;

%%%%%%%%%%%%%%%%%%%%%%%%%% svm train %%%%%%%%%%%%%%%%%%%%%%%

for i=1:people - 1
    for j=i + 1:people
        train_data = [feature_train((i - 1) * photo_each_person + 1:i * photo_each_person, :);feature_train((j - 1) * photo_each_person + 1:j * photo_each_person, :)];
        group = [ones(photo_each_person, 1); zeros(photo_each_person, 1)];
        multi_svm_struct{i}{j} = svmtrain(train_data, group, 'kernel_function', 'linear');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%% test %%%%%%%%%%%%%%%%%%%%%%%

test_ = zeros(112 * 92, people * (10 - photo_each_person));
for i=1:people
    for j=(photo_each_person + 1):10
        addr = strcat('orl_faces/s', num2str(i), '/', num2str(j), '.pgm');
        img = imread(addr);
        img = img(1:112 * 92);
        test_(:, (10 - photo_each_person) * (i - 1) + (j - photo_each_person)) = img';
    end
end

realclass = repelem([1:1:people]', 10 - photo_each_person);

test_min = zeros(112 * 92, people * (10 - photo_each_person));
for i=1:people * (10 - photo_each_person)
    test_min(:, i) = test_(:, i) - img_avg;
end

feature_test = test_min' * VT;

%%%%%%%%%%%%%%%%%%%%%%%%%% svm test %%%%%%%%%%%%%%%%%%%%%%%

vote = zeros(size(feature_test, 1), people);
for i=1:people - 1
    for j=i + 1:people
        group = svmclassify(multi_svm_struct{i}{j}, feature_test);
        vote(:, i) = vote(:, i) + (group == 1);
        vote(:, j) = vote(:, j) + (group == 0);
    end
end

[~, class] = max(vote, [], 2)

accuracy = sum(class == realclass) / length(class)