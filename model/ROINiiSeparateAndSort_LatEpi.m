% function ROINiiSeparateAndSort_LatEpi(ext, seg_fileIndexer)
% ROINiiSeparateAndSort: 
%   This script uses "ROINiiSeparater.m" to output trabecular bone 
%   images, trabecular segmentation, and non-trabecular into separate 
%   folders for training the deep learning segementation algorithm.
%
% Syntax: ROINiiSeparateAndSort
%
% Input:
%   ext = folder your training data is in and where your separated training
%         data will end up.
%   fileIndexer = suffix at the end of segmentation file for identification
%                 in file search. This must be uniformly named for all
%                 training files!
%   niiNameBeforePattern = pattern used to help the program identify the nii
%                          file name from the segmentation file.
% Output:
%   None
%
% Written by: Jack Consolini
% Last Update: 2024-06-11
%
% Test:
% For MR Retrieval THA SegmentationZ:\Ek\Data\LateralEpi\AI
ext='\\10.101.100.209\Lab_Data\Ek\Data\LateralEpi\2024\'; %update to be path with your subject folders, AI folder 
seg_fileIndexer = 'Seg.nii.gz'; % Update this to be the common name of your segmentation files (sgmt...)
niiFile = 'cor_pd.nii.gz';
%
%% A) Starting directory & nii folders
% Output directories
pROIImg=strcat(ext,"ImgWithSeg\");
pROISeg=strcat(ext,"Seg\");
pNoROIImg=strcat(ext,"ImgNoSeg\");
if ~exist(pROIImg,'dir'), mkdir(pROIImg); end
if ~exist(pNoROIImg,'dir'), mkdir(pNoROIImg); end
if ~exist(pROISeg,'dir'), mkdir(pROISeg); end
%% B) Check for separation file and perform .nii separation
segList = dirPlus(ext,'Struct',true,'FileFilter', seg_fileIndexer);
for i = 1:length(segList)
    checkSeparation = dirPlus(segList(i).folder,'Struct',true,'FileFilter', '_Separation.xlsx');
    if ~isempty(checkSeparation) == 0
        % Create spreadsheet for "ROINiiSeparater.m"
        tableName = strcat(extractBefore(segList(i).name,'.nii'),'_Separation.xlsx');
        tableHeader = {'Path', 'ImgName', 'SegName'};
        tableData = {string(segList(i).folder) string(niiFile) string(segList(i).name)};
        T = cell2table(tableData, 'VariableNames', tableHeader);
        fullTablePath = fullfile(segList(i).folder,tableName);
        writetable(T,fullTablePath)
        fprintf('Separation table created for %s\n', fullTablePath);
        % Separate nii files
        ROINiiSeparater_LatEpi(pROIImg, pROISeg, pNoROIImg, fullTablePath,i)
        fprintf('Separated .nii file for %s\n', segList(i).name);
    elseif ~isempty(checkSeparation) == 1
        % Separate nii files
        fullTablePath = strcat(checkSeparation.folder,'\',checkSeparation.name);
        fprintf('.nii file ALREADY separated for exists for %s\n', checkSeparation.name);
        ROINiiSeparater_LatEpi(pROIImg, pROISeg, pNoROIImg, fullTablePath,i)
        fprintf('Separated .nii file for %s\n', segList(i).name);
    end
end
