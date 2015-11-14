%Test
clc
clearvars
close all
tic
%% Load the data, split it into train and test data
display('Loading data...');

XTrain = [];
YTrain = [];
batchParts = 5;
for n = 1 : batchParts
    path = strcat('small_data_batch_', num2str(n));
    load(path);
    if n ~= batchParts        
        XTrain = [XTrain; data];
        YTrain = [YTrain; labels];
    else
        XTest = data;
        YTest = labels;
    end
end

dXTrain = double(XTrain);
dXTest = double(XTest);

%% Train the data using the intensity values as features
display('Trainging (with intensity features)...')
NBModel = fitNaiveBayes(dXTrain, YTrain);
Y = NBModel.predict(dXTest);
cMat = confusionmat(YTest, Y);
intensityAcc = sum(diag(cMat)) / sum(sum(cMat))

%% Setup VLFeat
run('vlfeat-0.9.20/toolbox/vl_setup');

%% Uncomment the following lines to visualize the HoG features
%{
cellSize = 8;
I = im2single(imread('link.jpg'));
hog = vl_hog(I, cellSize, 'verbose');
imhog = vl_hog('render', hog, 'verbose') ;
clf ; imagesc(imhog) ; colormap gray ;
%}
%% Uncomment the following lines to visualize gradients
% [Gmag, Gdir] = imgradient(rgb2gray(imread('link.jpg')), 'prewitt');
% figure; imshowpair(Gmag, Gdir, 'montage');

%% Extract HoG features
cellSize = 4;
XTrainFeat = [];
w = waitbar(0,'Extracting HoG (Training Set)...');
size_Xtrain = size(XTrain, 1);
for i = 1 : size_Xtrain
    im = im2single(reshape(XTrain(i,:), 32, 32, 3));
    hog = vl_hog(im, cellSize);
    XTrainFeat = [XTrainFeat;hog(:)'];
    waitbar(i / size_Xtrain);
end
close(w)

XTestFeat = [];
w = waitbar(0,'Extracting HoG (Testing Set)...');
size_XTest = size(XTest, 1);
for i = 1 : size_XTest
    im = im2single(reshape(XTest(i,:), 32, 32, 3));
    hog = vl_hog(im, cellSize);
    XTestFeat = [XTestFeat; hog(:)'];
    waitbar(i / size_XTest);
end
close(w)

%% Train on HoG features
display('Trainging (with HOG features)...');

NBModel = fitNaiveBayes(XTrainFeat, YTrain);
Y = NBModel.predict(XTestFeat);
cMat = confusionmat(YTest, Y);
hogAcc = sum(diag(cMat)) / sum(sum(cMat))

%%
toc