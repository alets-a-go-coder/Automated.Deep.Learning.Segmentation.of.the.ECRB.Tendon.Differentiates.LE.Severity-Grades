function ROIAutoSegmenter_SegModOnly(pModels, fSeg,fImg)
% ROIAutoSegmenter: segments nii files based on .mat output from ROISegmentTrain_Lambda. 
% Segments a 512x512 image.
%
% Syntax: ROIAutoSegmenter
%
% Input:
%   pModels = (char) classification and segmentation file paths
%   fSeg = (char) segmentation file name
%   pImg = (char) path of the .nii or .nii.gz file
%   fImg = (char) name of the .nii or .nii.gz file
% Output:
%   None
%
% Notes
%   -This code is located in: \\10.101.100.209\Lab_Data\ConsoliniJ\IRB2022-1920_LAT_EPI
%   -https://www.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html
% Written by Jack Consolini
% Last Update: 2026-06-12
%
%% Get classification and segmentation CNN (.mat session)
txt='Semantic Image Segmentation';
fprintf('\n%s\n%s\n',txt,numdash(txt))
if nargin < 3
    % Select segmentation model
    fModels=fSeg;
    % [fSeg,pSeg]=uigetfile('*.mat','Select ROI Segmentation Model File'); 
    % Select Nifti file to be segmented
    [fImg,pImg]=uigetfile('*.nii.gz','Nifti file to be segmented'); 
    [~,name,ext]=fileparts(fImg);
else
    pSeg = pModels;
    fModels=fSeg;
    % Folder name pattern for determing working directory
    [pImg,name,ext]=fileparts(fImg)
end
cd(pImg)

%% Load segmentation model
load(fullfile(pModels,fModels),'SegNet','pixelLabelID'); 

%% B) Load .nii or .nii.gz file
% Read in Image and if compressed nifti, convert to .nii and uncompress if .gz
fprintf('File to segment: %s\n',name);
switch ext
    case '.nii'
        writeCompressed=false; % Save final segmentation as nii
    case '.gz'
        writeCompressed=true; % Save final segmentation as nii.gz
end
% Read in image information
switch ext
    case {'.nii','.gz'}
        infoImg=niftiinfo(fImg);
        VImg=niftiread(infoImg);
    case '.mhd'
        [StrData, Success, Message] = elxMetaIOFileToStrDatax(fImg);
        if ~Success, error('%s - Problem reading file',upper(mfilename)); end
        VImg=StrData.Data;
end

%% C) Set-up output segmentation array to be filled with binary information
VSeg=zeros(size(VImg),'uint16');

%% D) Determine through-plane axis and slices to segment
% Smallest dimension is through-plane (e.g. 42 for full volumes, 1 for LPS exports)
[~, sliceAxis] = min(size(VImg));
sliceRange = 1:size(VImg, sliceAxis);
fprintf('Volume size: [%s], through-plane axis: %i (%i slices)\n', ...
    num2str(size(VImg)), sliceAxis, numel(sliceRange));

% seg_slices = input('Please enter an array containing slices for segmentation (e.g., [15]): ');
if numel(sliceRange) == 1
    seg_slices = 1;
else
    % Full 3D volume: set to the slice index containing label 1 (e.g. 15 for P1)
    seg_slices = 1;
end

%% E) Perform auto-segmentation slice by slice
for ii = 1:numel(sliceRange)
    j = sliceRange(ii);
    VSlice2D = extractVolumeSlice(VImg, j, sliceAxis);
    sliceSize = size(VSlice2D);

    fprintf('\tWorking on Slice %i (axis %i): ', j, sliceAxis);
    sliceNum = strcat("slice_", string(j));
    fOutImg = strcat(name, '_', sliceNum, '.nii');
    niftiwrite(VSlice2D, fOutImg, 'Compressed', false);

    VSegSlice = zeros(sliceSize, 'uint16');
    if ~ismember(j, seg_slices)
        fprintf('--> Nothing to segment\n');
        VSegSliceUS = zeros(sliceSize, 'uint16');
    else
        fprintf('----> Performing Segmentation\n');
        % SegNet was trained on 512x512 inputs (Img512x3Read downsamples as needed)
        readFcn = @(x) Img512x3Read(x);
        imdsSeg = imageDatastore(fOutImg, 'FileExtensions', {'.nii', '.gz'}, 'ReadFcn', readFcn);

        pxdsResults = semanticseg(imdsSeg, SegNet, 'OutputType', 'uint8');

        sliceSeg = readimage(pxdsResults, 1);
        if ~isequal(size(sliceSeg), sliceSize)
            sliceSeg = imresize(sliceSeg, sliceSize);
        end
        for k = 1:numel(pxdsResults.ClassNames)
            idx = find(sliceSeg == pxdsResults.ClassNames{k});
            VSegSlice(idx) = pixelLabelID(k);
        end
        VSegSliceUS = imresize(VSegSlice, sliceSize, 'nearest');
    end

    VSeg = writeVolumeSlice(VSeg, VSegSliceUS, j, sliceAxis);
    delete(fOutImg)
end

%% F) Generate output file (while making sure seg and img file header match)
% Generate output filename
switch ext
    case {'.nii'}
        % Generate output filename
        fSegOut=strcat(name,'_AutoSeg.nii');
    case {'.gz'}
        % Generate output filename
        [~,name]=fileparts(fImg);
        vals=split(name,'.');
        fSegOut=strcat(vals{1},'_AutoSeg.nii.gz');
    case '.mhd'
        % Generate output filename
        fSegOut=strcat(name,'_AutoSeg',ext);
end
% Save segmentation
switch ext
    case {'.nii','.gz'}
        % Save segmentation
        niftiwrite(VSeg,fSegOut,'Compressed',writeCompressed);
    case '.mhd'
        % Update data and data type
        StrData.ElementType='MET_USHORT';
        StrData.Data=VSeg;

        % Save segmentation
        [Success, Message]= elxStrDataxToMetaIOFile(StrData,fSegOut);
        if ~Success, fprintf('Problem saving mhd: %s\n',Message); end
end
fprintf('AutoSeg saved as: %s\n',fSegOut)

%% G) Delete slice folders after segmentation file has been created
files = string(dirPlus(pImg,'Struct',false,'FileFilter', '.png'));
for j=1:numel(files)
    if contains(files(j), 'slice') == 1
        delete(files(j));
        folder = extractBefore(files(j), "\pixel");
        rmdir(folder)
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% SUBROUTINES FOR READING IMAGE DATASTORE %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function VSlice2D = extractVolumeSlice(VImg, sliceIndex, sliceAxis)
switch sliceAxis
    case 1
        VSlice2D = squeeze(VImg(sliceIndex, :, :));
    case 2
        VSlice2D = squeeze(VImg(:, sliceIndex, :));
    case 3
        VSlice2D = VImg(:, :, sliceIndex);
end
end

function VVol = writeVolumeSlice(VVol, VSlice2D, sliceIndex, sliceAxis)
switch sliceAxis
    case 1
        VVol(sliceIndex, :, :) = VSlice2D;
    case 2
        VVol(:, sliceIndex, :) = VSlice2D;
    case 3
        VVol(:, :, sliceIndex) = VSlice2D;
end
end

function im = squeezeNiftiSlice(im)
if ndims(im) == 3
    im = squeeze(im);
end
end

function data = Img1024x3Read(filename)
% Read in the image
im = squeezeNiftiSlice(niftiread(filename));
% Resize as needed
if size(im,1)~=1024 && size(im,2)~=1024 % Make everything 1024x1024
    data=imresize(im,[1024 1024]);
else
    data=im;
end
% Stretch contrast to make it within an 8-bit range
data=strchcntrst(data,8);
% Make it an 16-bit RGB image
data=repmat(uint8(data),1,1,3);
end

function data = Img512x3Read(filename)
% Read in the image
im = squeezeNiftiSlice(niftiread(filename));
% Resize as needed
if size(im,1)~=512 && size(im,2)~=512 % Make everything 512x512
    data=imresize(im,[512 512]);
else
    data=im;
end
% Stretch contrast to make it within an 8-bit range
data=strchcntrst(data,8);
% Make it an 8-bit RGB image
data=repmat(uint8(data),1,1,3);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function data = Img3Kx3Read(filename)
% Read in the image
im = squeezeNiftiSlice(niftiread(filename));
% Resize as needed
if size(im,1)~=2.999e3 && size(im,2)~=2.999e3 % Make everything 512x512
    data=imresize(im,[2.999e3 2.999e3]);
else
    data=im;
end
% Stretch contrast to make it within an 8-bit range
data=strchcntrst(data,8);
% Make it an 8-bit RGB image
data=repmat(uint8(data),1,1,3);
end