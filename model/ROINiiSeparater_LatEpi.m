function ROINiiSeparater_LatEpi(pROIImg, pROISeg, pNoROIImg, fileName, subNum)
% Nii_Separater - Will take list of nii image files and
% corresponding nii seg file and separate into folders of the ROI images, 
% images with no ROI and ROI segmentation to facilitate training of 
% classification model and semantic image segmentation model.
%
% Syntax: Nii_separater
%
% Input:
%   pROIImg = output directory for images with ROI in them
%   pROISeg = output directory for segmentations of ROI 
%   pNoROIImg = output directory for images with only Bkgrnd in them
%   fileName = file you are to seperate
%   subNum = number of subject
%
% Output:
%   None
%
% Notes:
%   - Put a single hip's nii image and nii segmentation files into a the 
%     same directory
%   - The input xlsx file should have (provide path for every image):
%       col1Name=Path - path of nii img and seg files: c:\temp\hip001
%       col2Name=ImgName - name of image file: hip001.nii
%       col3Name=SegName - name of segmentation file: hip001_SEG.nii
% 
% Written by: Jack Consolini (Modified from Matt's nii_img_seg_separate)
% Last Update: 2024-06-11
%% A) Identify images to separate and folders to place seperate nii files into
if nargin < 1
    [fIn,pIn]=uigetfile('*.xlsx','Please Select File of List of Files')
    % Identify output directories, and create as needed
    pROIImg='Z:\ConsoliniJ\Incubator Projects\Auto_Mes_in_ZTE\for_lambda\ImgWithSeg\';
    pROISeg='Z:\ConsoliniJ\Incubator Projects\Auto_Mes_in_ZTE\for_lambda\Seg\';
    pNoROIImg='Z:\ConsoliniJ\Incubator Projects\Auto_Mes_in_ZTE\for_lambda\ImgNoSeg\';
    % pROIImg='c:\temp\ImgWithSeg\';
    % pROISeg='c:\temp\Seg\';
    % pNoROIImg='c:\temp\ImgNoSeg\';
    if ~exist(pROIImg,'dir'), mkdir(pROIImg); end
    if ~exist(pNoROIImg,'dir'), mkdir(pNoROIImg); end
    if ~exist(pROISeg,'dir'), mkdir(pROISeg); end
else
    [pIn,fIn,~]=fileparts(fileName);
end
T=readtable(fullfile(pIn,fIn));

%% B) Segment the individual images
% Go through list of image/seg file to process
i=size(T,1);
% for i=1:size(T,1)
    
    % Current image name
    fImg=T.ImgName{i};
    subNum=extractBefore(fImg, ".nii")
    fprintf('Working on: %s\n',fImg);
    [~,nameImg]=fileparts(fImg);

    % Current segmentation name
    fSeg=T.SegName{i};
    [~,nameSeg]=fileparts(fSeg);

    % Get header info for Img and Seg
    infoImg=niftiinfo(fullfile(T.Path{i},fImg));
    infoSeg=niftiinfo(fullfile(T.Path{i},fSeg));

    % Read in Img and Seg
    Img=niftiread(infoImg);
    Seg=niftiread(infoSeg);
    
    % Make sure matrix sizes match up (among things to check)
    isXR=0;
    if size(infoSeg.ImageSize,2) == 2
        isXR = 1;
    % If you are working with X-ray, the segmentation might only have 2
    % dimensions, so it could be fine, but just make note of that
        if (infoImg.ImageSize(1) == infoSeg.ImageSize(1)) && (infoImg.ImageSize(2) == infoSeg.ImageSize(2))
            warning(['The in-plane dimensions of the image and seg match (imgSize %i x %i, segSize %i x %i), ' ...
                'but through plane dimensions do not (or through plane is missing), just be aware of this. \n'], ...
                infoImg.ImageSize(1),infoImg.ImageSize(2),infoSeg.ImageSize(1),infoSeg.ImageSize(2));
        else
            error('Problem with matrix size:\n\tfImg=%s\n\tfSeg=%s\n',fImg,fSeg);
        end
    elseif size(infoSeg.ImageSize,2) == 3 && ~all(infoImg.ImageSize==infoSeg.ImageSize)
        error('Problem with matrix size:\n\tfImg=%s\n\tfSeg=%s\n',fImg,fSeg);
    end

    % Determine which scan plane to segment through for non-isotropic scans
    imgSize = size(Img);
    % For X-ray, size(infoSeg.ImageSize,3) should be 1, because 3 is
    % technically out of bounds, so it will overwrite to make it so you
    % only segment on the singular image slice.
    if isXR == 1
        sliceNum = 1;
        index = 3;
    % For MR, this will go through all slices in the through plane
    % direction
    elseif size(infoSeg.ImageSize,2) == 3 
        [sliceNum, index] = min(imgSize);
    end

    % Go through the slices of Img and Seg
    for j=1:sliceNum
        fprintf('\tSlice: %i - ',j);

        % Isolate individual slice of Img and Seg data
        if index == 1
            indImg=Img(j,:,:);
            indSeg=Seg(j,:,:);
        elseif index == 2
            indImg=Img(:,j,:);
            indImg = permute(indImg, [1, 3, 2]);
            indSeg=Seg(:,j,:);
            indSeg = permute(indSeg, [1, 3, 2]);
        elseif index == 3
            indImg=Img(:,:,j);
            indSeg=Seg(:,:,j);
        end
        % Find locations of where segmentation exists, if they do
        idx=find(indSeg, 1);

        % Create output filenames for individual slice of Img and Seg
        % fImgOut=sprintf('s000%i_Img_Slice%s.nii',subNum,leadzero(j,10));
        % fSegOut=sprintf('s000%i_Seg_Slice%s.nii',subNum,leadzero(j,10));
        fImgOut=sprintf('%s_Img_Slice%s.nii',subNum,leadzero(j,10));
        fSegOut=sprintf('%s_Seg_Slice%s.nii',subNum,leadzero(j,10));

        if isempty(idx)
            fprintf('No ROI Found\n')
            % No segmentation, i.e. fat, on this slice
            % Save the individual image slice in NoROIImg dir
            niftiwrite(indImg,fullfile(pNoROIImg,fImgOut));
            % No need to save segmentation
        else
            fprintf('ROI Found\n')
            % Segmentation of fat found on slice
            % Save the individual image slice in ROIImg dir
            niftiwrite(indImg,fullfile(pROIImg,fImgOut));
            % Save the individual segmentation slice in ROISeg dir
            niftiwrite(indSeg,fullfile(pROISeg,fSegOut));
        end
    end
% end

