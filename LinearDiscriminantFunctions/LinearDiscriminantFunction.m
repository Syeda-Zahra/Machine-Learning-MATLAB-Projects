%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BME777: LAB 2: Linear Discriminant Functions.
% Acknowledgement: We thankfully acknowledge UCI Machine Learning Repository for the 
% dataset used in this lab exercise.
% Indian Liver Patient Dataset.
% Link: https://archive.ics.uci.edu/ml/datasets/ILPD+%28Indian+Liver+Patient+Dataset%29#
% Class1: Liver patient. Class2: non Liver patient.
% DataLab2_1: Features: TP Total Proteins and ALB Albumin with modification for problem simplification. 
% Features 8-9. 
% DataLab2_2: Features: TP Total Proteins and A/G Ratio	Albumin and
% Globulin Ratio. Features 8-10.
% 50 samples were extracted for each class.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% 1. Data: 100x3 dataset. The first column contains the feature x1, the second
% column contains the feature x2. The class labels are given in the third
% column.
% 2. ClassSplit: Threshold where classes are divided. See the third
% column of the Data to choose the correct threshold.
% 3. DataSplitRate: Threhold to split the data in each class into training and testing data.
% For e.g., DataSplitRate = 0.4 ==> 40% data of class 1,2 is for training.
% 60% of the data is for testing.
% 4. InitialParameterSet: Initial values of the set of parameters. For
% e.g., InitialParameterSet = [0 0 1].
% 5. LearningRate: Learning rate when updating the algorithm.
% 5. Theta: The expected cost that the optimized parameter set may give.
% 6. MaxNoOfIteration: the maximum number of iterations the algorithm may run.
%
% Output:
% 1: TrainedParameterSet: The set of optimized parameters.
% 2: NoOfIteration: The number of iteration when the algorithm converges.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example of use:
% load DataLab2_1.mat
% Data = DataLab2_1;
%ClassSplit = 50;
%DataSplitRate = 0.4;
% InitialParameterSet = [0 0 1];
% LearningRate = 0.01;
% Theta = 0;
% MaxNoOfIteration = 300;
% [OptimizedParameterSet,NoOfIteration] = ...
% lab2(Data,ClassSplit,DataSplitRate, ... 
% InitialParameterSet,LearningRate,Theta,MaxNoOfIteration);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [TrainedParameterSet,NoOfIteration] =lab2(Data,ClassSplit,DataSplitRate,InitialParameterSet,LearningRate,Theta,MaxNoOfIteration)

close all;

[Len,~] = size(Data);

% Split the data into two classes based on ClassSplit. 
index1=find(Data(:,3)==1);
index2=find(Data(:,3)==2);

Class1 =[];
Class1(1,:) = Data(index1,3);
Class1(2,:) = Data(index1,1);
Class1(3,:) = Data(index1,2);



Class2 =[];
Class2(1,:)=Data(index2,3);
Class2(2,:)=Data(index2,1);
Class2(3,:)=Data(index2,2);

% Calculate the number of training samples.
Train_Num1 = ClassSplit*DataSplitRate;
Train_Num2 =ClassSplit*DataSplitRate;

% Split the data in class 1 into training and testing sets. 
Train_Class1 =[];
Train_Class1=Class1(:,1:Train_Num1);
Test_Class1 =Class1(:,Train_Num1+1:end);


% Split the data in class 2 into training and testing sets.
% Do not forget to normalize the training data of class 2;
Train_Class2 =[];
Train_Class2=Class2(:,1:Train_Num2);%spliting into training data
%%%normalize training data%%
Train_Class2(1,:)=-1;  %entire first row should be -1
Train_Class2(2:3,:)=Train_Class2(2:3,:)*-1; %remaining two rows must be opposite of what they were eg. if positive turns into negative

Test_Class2 =Class2(:,Train_Num2+1:end);

figure
scatter(Train_Class1(2,:),Train_Class1(3,:)); 
hold on; 
scatter(Train_Class2(2,:)*-1,Train_Class2(3,:)*-1);
title('Feature Space');
xlabel('x1'); 
ylabel('x2'); 
legend('Class 1','Class 2');

% Prepare the training data including all training samples of classs 1 and
% 2.
Train_Data = zeros(3, Train_Num1 + Train_Num2);
Train_Data(:,1:Train_Num1)=Train_Class1; 
Train_Data(:,Train_Num1+1:Train_Num1+Train_Num2)=Train_Class2;

% Prepare the test data including all test samples of classs 1 and
% 2.
Test_Data = zeros(3,(ClassSplit-Train_Num1)+(ClassSplit-Train_Num2));
Test_Data(:,1:(ClassSplit-Train_Num1)) = Test_Class1;
Test_Class2(1,:)=-1;
Test_Class2(2:3,:) = Test_Class2(2:3,:)*-1;
Test_Data(:,ClassSplit-Train_Num1+1:end) = Test_Class2;

% Implement basic gradient algorithm.
OptParams = InitialParameterSet;%

PerceptronFunction = zeros(MaxNoOfIteration,1);%
Criterion = 1;%
NoOfIteration = 1;%
GradientOfCost = zeros(1,3);

%Magnitude Function
m= @(v) (sqrt(v(1)*v(1)+v(2)*v(2)+v(3)*v(3)));
ClassificationAccuracyT=[];
NumOfMisclassified=[];


while ((Criterion>Theta))

       
    g = OptParams*Train_Data; %find misclassified samples
    
    misclass=find(g<0);
    
     for i=1: length(misclass)
         j=misclass(i)
         
         %gets S1-Sj
         for k=1:3
          GradientOfCost(1,k)= (Train_Data(k,j));
         end
         
        % Calculate the classification accuracy of the predictions on the test data.
       %ClassificationAccuracy = (length(Test_Data)-length(misclass))/length(Test_Data)*100;
       %T=table(ClassificationAccuracy,misclass);
        
      OptParams= OptParams + LearningRate*GradientOfCost;
       gcheck=OptParams*Test_Data;
       misclassified = find(gcheck<0);
        PerceptronFunction(NoOfIteration)=sum(OptParams.*GradientOfCost);
        
     end
       
     NumOfMisclassified(NoOfIteration,:)=i;
     ClassificationAccuracyT(NoOfIteration,:)=(length(Test_Data)-length(misclassified))/length(Test_Data)*100;
      %PerceptronFunction(NoOfIteration)=sum(OptParams.*GradientOfCost);
      
     %criterion 
      Criterion = m(GradientOfCost);
     
      
       
     if (NoOfIteration==MaxNoOfIteration)%stops/breaks the algorithm  once the max iteration number is reached
    break;
     end
    
    NoOfIteration = NoOfIteration + 1;
    
end

% Plot the values of the perceptron function.
figure;
plot(PerceptronFunction);title('Perceptron Function');

% Plot data of class 1, class 2 and the estimated boundary.
figure;
y = @(x,OptParams) (((OptParams(3)*x)+OptParams(1))/-OptParams(2));
x=[-10:10];
estimatedboundary=y(x,OptParams);
scatter(Train_Class1(2,:),Train_Class1(3,:)); 
hold on; 
scatter(Train_Class2(2,:)*-1,Train_Class2(3,:)*-1);
hold on;
plot(estimatedboundary,x);
axis([3 10 0 6]); 
xlabel('x1'); 
ylabel('x2');
legend('Class 1','Class 2');
title('Feature Space with Decision Boundary for Training Set');


%outputs the number of iteration
NoOfIteration


%Getting optimal params
TrainedParameterSet = OptParams;

%Displaying accuruacy and misclassifications in a table
gcheck=OptParams*Test_Data;
misclassification = find(gcheck<0);
ClassificationAccuracy = (length(Test_Data)-length(misclassification))/length(Test_Data)*100;
TABLEA=[ClassificationAccuracyT,NumOfMisclassified];

T = array2table(TABLEA,...
    'VariableNames',{'Accuracy','Misclassifications'})
end