% function ROISegmentTrain(pROIImg, pROISeg, fPrefix, pOut, imgSizeCheck, numLabels)
% ROISegmentTrain: Using trainnet, ROISegmentTrain_Lambda creates a 
% a deep learning segmantic segmentation model that is capable of
% indentifying and segmenting single or multi-label segmentations.
%
% Syntax: ROISegmentTrain
%
% Input:
%   pROIImg = (string) Path to folder of nii image training data
%   pROISeg = (string) Path to folder of nii segmentation training data
%   fPrefix = (string) Prefix of saved model .mat file 
%             (e.g.'SPGR-SegModel' -->  SPGR-SegModel_[currentDate].mat)
%   pOut = (string) path of .mat file output location
%   imgSizeCheck = 1 = 512^2, or 2=3k^2
%   numLabels = (integer) Number of labels in addition to "background"
%                           --> Single ROI --> numLabels = 1
% Output:
%   None
% Functions:
%   - imageDatastore (Mathworks, built from Datastore)
%   - pixelLabelDatastore (Mathworks)
%   - ImgRead512 (MKoff, saved in: \\10.101.100.209\Lab_Data\ConsoliniJ\IRB2022-1920_LAT_EPI\segmentation_training)
%   - MaskRead512 (MKoff, saved in: \\10.101.100.209\Lab_Data\ConsoliniJ\IRB2022-1920_LAT_EPI\segmentation_training)
%   - ImgRead3k (MKoff, saved in: \\10.101.100.209\Lab_Data\ConsoliniJ\IRB2022-1920_LAT_EPI\segmentation_training)
%   - MaskRead3k (MKoff, saved in: \\10.101.100.209\Lab_Data\ConsoliniJ\IRB2022-1920_LAT_EPI\segmentation_training)
%   - partitionImdsPxds (MKoff, saved in: \\10.101.100.209\Lab_Data\ConsoliniJ\IRB2022-1920_LAT_EPI\segmentation_training)
%   - augmentImageAndLabel (MKoff, saved in: \\10.101.100.209\Lab_Data\ConsoliniJ\IRB2022-1920_LAT_EPI\segmentation_training)
%   - modelLoss (MKoff, saved in: \\10.101.100.209\Lab_Data\ConsoliniJ\IRB2022-1920_LAT_EPI\segmentation_training)
%   - deeplabv3plusLayers (Mathworks)
%   - unetLayers (Mathworks)
%   - trainingOptions (Mathworks)
%   - trainNetwork (Mathworks)
%   - semanticseg (Mathworks)
%   - evaluateSemanticSegmentation (Mathworks)
%
% Notes:
%   - https://www.mathworks.com/help/vision/ref/deeplabv3plus.html
%   - https://www.mathworks.com/help/deeplearning/ug/create-simple-semantic-segmentation-network-in-deep-network-designer.html
%   - https://www.mathworks.com/help/vision/ug/semantic-segmentation-using-deep-learning.html
%   - (Modified from Matthew Koff's MFKTrainADeepLabV3NetworkExample.m)
%   - You may need to set the path to include the sub functions. To do
%     this, go to Home > Set Path > Add with Subfolders > Select MATLAB >
%     Close.
%
% Written by Jack Consolini 
% Last Update: 2025-06-24
%
%% User Inputs (outside of function) TEST WITHIN LAMBDA SERVER!
% *** MR THA Training and Test of Script ***
pROIImg='/home/mldl/Documents/MATLAB/LateralEpi/ImgWithSeg';
pROISeg='/home/mldl/Documents/MATLAB/LateralEpi/Seg';
pOut='/home/mldl/Documents/MATLAB/LateralEpi'; % here is where your trained model will output.
fPrefix='MR_LatEpi_Seg';
imgSizeCheck = 1;
numLabels = 5;
functionOff=1;
%Random Counter
rng("default");
% *** XR THA Training and Test of Script ***
% pROIImg='/data/XR_THA/ROI';
% pROISeg='/data/XR_THA/ROISeg';
% pOut='/home/mldl/Documents/MATLAB/SteinPrograms/THA';
% fPrefix='XR_THA_Seg';
% imgSizeCheck = 2;
% numLabels = 2;

%% User Inputs (inside of function)
if functionOff==0
    if nargin == 0
        % Identify training image directories and .mat file prefix
        txt='User inputs';
        fprintf('%s\n%s\n',txt,numdash(txt));
        pROIImg=input('Enter ROI image files path: ','s');
        pROISeg=input('ROI segmentation files path: ','s');
        pOut=input('Enter data output directory: ','s');
        fPrefix=input('Enter prefix for saved .mat file: ','s');
        % Set image size, to indicate which readfile to use
        imgSizeCheck=input('Are images 512x512 (1) or 3Kx3K (2): ');
        numLabels=input(sprintf('Number of labels (e.g. 2, 3, 4...)): '));
    elseif nargin < 6
        error('%s - Check number of required inputs',upper(mfilename));
    end
end

%% A) Set number of labels and class names
% Background == 0 (always), label number increases linearly by 1
classes = "C1";
for i=1:numLabels
    if numLabels == 1
        classes = [classes;"C2"];
        break
    else
        labelName = strcat("C",string(i+1));
        classes = [classes; labelName];
    end
end
classNames = classes';
pixelLabelID = 0:numLabels;
numClasses = length(pixelLabelID);

%% B) Set up image (training images) and label (segmentation images) datastores
imdsLoc=pROIImg;
pxdsLoc=pROISeg;
if imgSizeCheck == 2
    imds = imageDatastore(imdsLoc, ...
    'FileExtensions','.nii','ReadFcn',@(x) ImgRead3k(x));
    pxds = pixelLabelDatastore(pxdsLoc,classes, pixelLabelID, ...
    'FileExtensions','.nii','ReadFcn',@(x) MaskRead3k(x));
    imageSize=[3e3 3e3 3];
elseif imgSizeCheck == 1
    imds = imageDatastore(imdsLoc, ...
    'FileExtensions','.nii','ReadFcn',@(x) ImgRead512(x));
    pxds = pixelLabelDatastore(pxdsLoc,classes, pixelLabelID, ...
    'FileExtensions','.nii','ReadFcn',@(x) MaskRead512(x));
    imageSize=[512 512 3];
end
%Calculate pixels per label to determine class weight (Alexa Tan, 8/1/25)
tbl=countEachLabel(pxds);
imageFreq=tbl.PixelCount ./ tbl.ImagePixelCount;
%imageFreq=sqrt(imageFreq);
%imageFreq=(imageFreq.^2);
classWeights=median(imageFreq) ./ imageFreq;
%classWeights(3:6)=0;

%% C) Split into train and test
prop=[0.8 0.1 0.1];
[imdsTrain,imdsValid,imdsTest,pxdsTrain,pxdsValid,pxdsTest] = partitionImdsPxds(imds,pxds,prop);
% Check out training images, Default ==1 , used for "preview" below
imdsTrain.ReadSize=1; 
% Make datastore for modeling
dsTrain=combine(imdsTrain,pxdsTrain);
dsValid=combine(imdsValid,pxdsValid);
dsTest=combine(imdsTest,pxdsTest);

%% D) Incorporate image augmentation into training
% What this does is attaches the augmentation function to the training
% dataset. This makes it so rather than adding augmented images into the
% dataset, it is making it so as the training occurs, images within the
% dataset may be augmented. --> need to make sure the augmentation is
% appropraite
% Generates different setting for each patient, 
%
% Simple translating, rotation, scaling, and shearing augmentation
xTrans = [-20 20]; % check if this is 10% of pixels or field of view, determine units, scale is likely based on 1.
yTrans = [-20 20];
Rot=[-15 15];
Scale=[0.7 1.3];
xShear=[-5 5];
yShear=[-5 5];
fillVal=classNames(1);
% Augment data - running this as a class, augmentation has to do with
% number of epochs you run, so each time you run you are only doing 30
% segmentations, if each epoch are you getting different augmentation 
dsTrain = transform(dsTrain, @(data)augmentImageLabel(data,xTrans,yTrans,Rot,Scale,xShear,yShear,fillVal));
% - what is the actual number of augmented data files that are produced,
% and how is this added to the dataset.
%% D.1) Preview how the augmented data appears
% Save the dsTrain data without any prior augmentation!
dsHold=dsTrain;
% Use read function on the dsHold so that we do not affect any
% location/index markers from any of the other combined datasets
dsHold=dsTrain;
dataPreview = read(dsHold);
B=labeloverlay(dataPreview{1},dataPreview{2},'Transparency',0.7);
imshow(permute(B(:,end:-1:1,:),[2 1 3]));
title("Augmented Images for Image Classification")

%% E) Set-up network with deep learning architecture
network = deeplabv3plus(imageSize,numClasses,'resnet50');

%% F) Set training options
% User defined parameters
initialLearningRate = 0.01;
l2reg = 0.0001;
% Epoch refers to one complete pass through the entire training dataset, at
% the end of an epoch, the model is updated, in attempts to reduce the loss
maxEpochs = 50; %Reduced from 200 to 50 - Alexa Tan 8/12/25
% Mini batch size is the number of images to train on at each iteration,
% this may be restricted by computational power. 
miniBatchSize = 1;
LearnRateDropPeriod=15;
LearRateDropFactor=0.5;
% If there are a small number of training files, may need to manually
% increase valFrequency to be non-zero or if it is too large, reduce!
valFrequency = floor(numel(imdsTrain.Files)/miniBatchSize);
if valFrequency == 0
    valFrequency = 1;
end
fprintf('Validation Frequency: %i\n',valFrequency);
valPatience = 15;
VerboseFrequency=100;
% Initialize training options
options = trainingOptions('sgdm',...
    'InitialLearnRate',initialLearningRate, ...
    'Momentum',0.9,...
    'L2Regularization',l2reg,...
    'MaxEpochs',maxEpochs,...
    'MiniBatchSize',miniBatchSize,...
    'LearnRateSchedule','piecewise',... 
    'LearnRateDropPeriod',LearnRateDropPeriod, ...
    'LearnRateDropFactor',LearRateDropFactor, ...
    'Shuffle','every-epoch',...
    'GradientThresholdMethod','l2norm',...
    'GradientThreshold',0.05, ...
    'Plots','training-progress', ...
    'VerboseFrequency',VerboseFrequency,...     % 'ExecutionEnvironment','multi-gpu',...
    'ValidationData',dsValid, ...
    'ValidationFrequency',valFrequency,...
    'Metrics','accuracy',...
    'Verbose',true,...
    'ValidationPatience',valPatience); 
% Specify loss function (Alexa Tan, 8/1/25)
Lossfcn =  @(Y,T) crossentropy(Y,T,classWeights,WeightsFormat="CB"); %Added WeightsFormat="CB" due to error (Alexa Tan, 8/5/25)
% Lossfcn =  @(Y,T) crossentropy(Y,T);
%Lossfcn = "crossentropy";

%% H) Train segmentation model
fprintf('Training the network ... ');
SegNet = trainnet(dsTrain,network,Lossfcn,options);
fprintf('Done!\n');

%% I) Save the trained model as a matlab session for future intialization 
% Get the current date and time
currentDateTime = datetime('now','Format','yyyyMMdd_hhmmss');
% Define your file extension
fileExtension = '.mat';
% Construct the output filename
fOut = strcat(fPrefix, '_', string(currentDateTime), fileExtension);
% Save file
fOut=fullfile(pOut,fOut);
save(fOut);
fprintf('Model saved as %s\n',fOut);
%% Test segmentation on validation
%validn=rand; testn=rand;
validn=0.1; testn=0.1;
rndtest=ceil(validn*length(imdsValid.Files));
I = readimage(imdsValid,rndtest);
C = semanticseg(I,SegNet,Classes=classes);
B = labeloverlay(permute(I(:,end:-1:1,:),[2,1,3]),permute(C(:,end:-1:1),[2,1]),Transparency=0.4);
figure; subplot(221); imshow(B);title(['Validation Image ',num2str(rndtest)])
P = readimage(pxdsValid,rndtest);
B = labeloverlay(permute(I(:,end:-1:1,:),[2,1,3]),permute(P(:,end:-1:1),[2,1]),Transparency=0.4);
subplot(223); imshow(B);title(['Ground Truth Image '])
rndtest=ceil(testn*length(imdsTest.Files));
I = readimage(imdsTest,rndtest);
C = semanticseg(I,SegNet,Classes=classes);
B = labeloverlay(permute(I(:,end:-1:1,:),[2,1,3]),permute(C(:,end:-1:1),[2,1]),Transparency=0.4);
subplot(222); imshow(B);title(['Test Image ',num2str(rndtest)]) 
P = readimage(pxdsTest,rndtest);
B = labeloverlay(permute(I(:,end:-1:1,:),[2,1,3]),permute(P(:,end:-1:1),[2,1]),Transparency=0.4);
subplot(224); imshow(B);title(['Ground Truth Image '])    
%% J) Access acurracy of model by applying to Test images
try
    pxdsResults = semanticseg(imdsTest,SegNet,'minibatchsize',12,...
    'ExecutionEnvironment','gpu','OutputType','uint8','WriteLocation',tempdir,...
    'Verbose',true);
catch
    pxdsResults = semanticseg(imds,SegNet,'minibatchsize',1,...
    'OutputType','uint8','WriteLocation',tempdir,'Verbose',true);
end
%% Extract out segmentation statistics
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest);
metrics.ClassMetrics;
metrics.ConfusionMatrix;
% Determine normalized confusion matrix
normConfMatData = metrics.NormalizedConfusionMatrix.Variables;
figure, h = heatmap(classes,classes,100*normConfMatData);
h.XLabel = 'Predicted Class';
h.YLabel = 'True Class';
h.Title = 'Normalized Confusion Matrix (%)';
% Determine Intersection over union (IoU) metric to evaluate performance
% IoU: for segmentation, IoU quantifies how well the predicted segmentation
% aligns with the actual segmentation.
imageIoU = metrics.ImageMetrics.WeightedIoU;
figure
histogram(imageIoU,60)
title('Image Mean IoU')

%% K.a) See real assessment of success of segmentation model (user select)
% Sorted IoU
[sortedIoU,idx]=sort(imageIoU);
num2plot=input(sprintf('Enter image to display (# of %i): ',length(idx)));
% Enter '1' for the best agreement and whatever the final number is the worst
% agreement. The order does NOT correspond to slice number!!
idx=flipud(idx);
num2plot=idx(num2plot);
% Displaying the evaluation of the segmentation
displayImg_Lambda(num2plot,imdsTest,pxdsTest,pxdsResults,imageIoU);

%% K.b) Read a test image and display predicted vs. actual results (random selection)

maxVal=numel(imdsTest.Files);
% Random number of Test file
ImageNumber=randi([1 maxVal]);
fprintf('Reading Test Image: %i\n',ImageNumber)
img = readimage(imdsTest,ImageNumber);
img=permute(img,[2 1 3]);

% Read in true segmentation
labelTrue=readimage(pxdsTest,ImageNumber)';
% Read in predicted segmentation
labelPred=readimage(pxdsResults,ImageNumber)';
    
% Make image of true segmentation
Btrue=labeloverlay(img,labelTrue,'transparency',0.5);
% Make image of predicted segmentation
Bpred=labeloverlay(img,labelPred,'transparency',0.5);

try
    MontPicts=cat(4,img,Btrue,Bpred);
catch
    MontPicts=cat(4,repmat(img,1,1,3),Btrue,Bpred);
end
figure, montage(MontPicts,'Size',[1 3])
title(sprintf('Test Image vs. Truth vs. Prediction. IoU = %.4f', imageIoU(ImageNumber)));

% end
