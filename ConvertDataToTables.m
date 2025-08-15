% upload data
% patient_folder = fullfile("C:\Users\asus\OneDrive\מסמכים\לימודים\פרויקט גמר\Code\Recordings\patient1");
clear all
close all
patientFolder = 'C:\Users\asus\OneDrive\מסמכים\לימודים\פרויקט גמר\Code\Recordings\patient1';
cd (patientFolder)
% load('CSC1_LFP.mat') - not usable, just real timing of all LFP
load('CSC1_LFP_per_event_type.mat')
load('events_tt.mat')

% filter data
VowelLabels = {'A','E','I','O','U'};
ImageryLabels = {'IMAGERY_A','IMAGERY_E','IMAGERY_I','IMAGERY_O','IMAGERY_U'};
relevant_indices_vowels = find(ismember(all_event_types,VowelLabels));
relevant_indices_Imagery = find(ismember(all_event_types,ImageryLabels));

VowelsCellArray = LFP_per_event_type(relevant_indices_vowels); % extract the 10 trials cells
ImageryCellArray = LFP_per_event_type(relevant_indices_Imagery);

% convert data from timetable to table
for i=1:length(relevant_indices_vowels)
    VowelsCellArray{i,1} = timetable2table(VowelsCellArray{i,1});
    VowelsCellArray{i,2} = all_event_types(relevant_indices_vowels(i)); % Label the data
    VowelsCellArray{i,1}{:,end+1} = char(VowelsCellArray{i,1}{:,1}); % the time is in type "duration", turn it to char vec
    VowelsCellArray{i,1}.Time = []; % erase the duration var    
    VowelsCellArray{i,1} = table2cell(VowelsCellArray{i,1}); % turn the table to cell array
end
for i=1:length(relevant_indices_Imagery)
    ImageryCellArray{i,1} = timetable2table(ImageryCellArray{i,1});
    ImageryCellArray{i,2} = all_event_types(relevant_indices_Imagery(i)); % Label the data
    ImageryCellArray{i,1}{:,end+1} = char(ImageryCellArray{i,1}{:,1}); % the time is in type "duration", turn it to char vec
    ImageryCellArray{i,1}.Time = []; % erase the duration var    
    ImageryCellArray{i,1} = table2cell(ImageryCellArray{i,1}); % turn the table to cell array
end

% add second channel
clear all_event_types
clear LFP_per_event_type
load('CSC2_LFP_per_event_type.mat')

C2_relevant_indices_vowels = find(ismember(all_event_types,VowelLabels));
C2_relevant_indices_Imagery = find(ismember(all_event_types,ImageryLabels));

C2_VowelsCellArray = LFP_per_event_type(C2_relevant_indices_vowels); % extract the 10 trials cells
C2_ImageryCellArray = LFP_per_event_type(C2_relevant_indices_Imagery);

% convert data from timetable to table
for i=1:length(C2_relevant_indices_vowels)
    C2_VowelsCellArray{i,1} = timetable2table(C2_VowelsCellArray{i,1});
    C2_VowelsCellArray{i,2} = all_event_types(C2_relevant_indices_vowels(i)); % Label the data
    C2_VowelsCellArray{i,1}{:,end+1} = char(C2_VowelsCellArray{i,1}{:,1}); % the time is in type "duration", turn it to char vec
    C2_VowelsCellArray{i,1}.Time = []; % erase the duration var    
    C2_VowelsCellArray{i,1} = table2cell(C2_VowelsCellArray{i,1}); % turn the table to cell array
end
for i=1:length(C2_relevant_indices_Imagery)
    C2_ImageryCellArray{i,1} = timetable2table(C2_ImageryCellArray{i,1});
    C2_ImageryCellArray{i,2} = all_event_types(C2_relevant_indices_Imagery(i)); % Label the data
    C2_ImageryCellArray{i,1}{:,end+1} = char(C2_ImageryCellArray{i,1}{:,1}); % the time is in type "duration", turn it to char vec
    C2_ImageryCellArray{i,1}.Time = []; % erase the duration var    
    C2_ImageryCellArray{i,1} = table2cell(C2_ImageryCellArray{i,1}); % turn the table to cell array
end

% Combine the 2 channels
for i=1:length(C2_relevant_indices_vowels)
    if C2_VowelsCellArray{i,2}{:} ~= VowelsCellArray{i,2}{:}
        disp('mismatching labeling on line i')
        disp(i)
        break
    end
    VowelsCellArray{i,3} = VowelsCellArray{i,2};
    VowelsCellArray{i,2} = C2_VowelsCellArray{i,1};
end

for i=1:length(C2_relevant_indices_Imagery)
    if C2_ImageryCellArray{i,2}{:} ~= ImageryCellArray{i,2}{:}
        disp('mismatching imagery labeling on line i')
        disp(i)
        break
    end
    ImageryCellArray{i,3} = ImageryCellArray{i,2};
    ImageryCellArray{i,2} = C2_ImageryCellArray{i,1};
end

% Save the converted data
[~,patientNum,~] = fileparts(patientFolder);
filename = strcat('C:\Users\asus\OneDrive\מסמכים\לימודים\פרויקט גמר\Code\Recordings\',patientNum,'_ConvertedData_noTables');
save(filename,"ImageryCellArray","VowelsCellArray");


