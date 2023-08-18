clc
clear 
close all

train_data=readtable('training.csv');
predict_data=readtable('prediction.csv');


% Generate example RNA sequence data and labels
numSequences = 1000;
sequenceLength = 16;

% Generate random RNA sequences (A, C, G, U)
rnaBases = ['A', 'C', 'G', 'U'];
rnaSequences = cell2mat(train_data.Var1);

% Generate random labels (coding: 1, non-coding: 0)
labels = train_data.Var2;

% Convert RNA sequences to numerical data using one-hot encoding
rnaEncoding = containers.Map({'A', 'C', 'G', 'T'}, [1, 2, 3, 4]);
encodedSequences = NaN(numSequences, sequenceLength);
for i = 1:numSequences
    encodedSequences(i, :) = arrayfun(@(base) rnaEncoding(base), rnaSequences(i, :));
end

% Split the data into training and test sets
splitRatio = 0.8;
splitIndex = round(numSequences * splitRatio);
xTrain = encodedSequences(1:splitIndex, :);
yTrain = labels(1:splitIndex);
xTest = encodedSequences(splitIndex+1:end, :);
yTest = labels(splitIndex+1:end);

% Build the neural network model
hiddenLayerSize = 50;
net = patternnet(hiddenLayerSize);

% Train the model
net = train(net, xTrain', yTrain');

% Test the model
predictions = net(xTest');
predictions_old=predictions;

predictions(predictions>=0.5)=1;
predictions(predictions<0.5)=0;


accuracy = sum(predictions == yTest') / numel(yTest);

disp(['Test Accuracy: ', num2str(accuracy)]);


%%%%%%%%%%%%%%PREDICT%%%%%%%%%

rnaSequences_predict = cell2mat(predict_data.Var1);
for i = 1:size(rnaSequences_predict,1)
    encodedSequences_predict(i, :) = arrayfun(@(base) rnaEncoding(base), rnaSequences_predict(i, :));
end
labels_predict= predict_data.Var2;

predictions_fff = net(encodedSequences_predict');
predictions_oldfff=predictions_fff;

predictions_fff(predictions_fff>=0.5)=1;
predictions_fff(predictions_fff<0.5)=0;

accuracy = sum(predictions_fff == labels_predict') / numel(labels_predict);

disp(['Test Accuracy: ', num2str(accuracy)]);

new_predictions=[predict_data.Var1 array2table(predictions_fff')];

writetable(new_predictions,'new_predictions_f.csv')
