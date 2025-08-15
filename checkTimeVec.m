% check if all times are the same

VowelsCellArrayCC = LFP_per_event_type(C2_relevant_indices_vowels); % extract the 10 trials cells

% Example cell array of timetables (replace these with your actual timetables)
timetables = VowelsCellArrayCC; %{timetable1, timetable2, timetable3, timetable4, timetable5};

% Get the time values of the first timetable
timeReference = timetables{1}.Time;

% Initialize a flag to true
areTimesIdentical = true;

% Loop through the other timetables and compare the time values
for i = 2:length(timetables)
    if ~isequal(timeReference, timetables{i}.Time)
        areTimesIdentical = false;
        break;
    end
end

% Display the result
if areTimesIdentical
    disp('The time values are identical for all timetables.');
else
    disp('The time values are not identical for all timetables.');
end
