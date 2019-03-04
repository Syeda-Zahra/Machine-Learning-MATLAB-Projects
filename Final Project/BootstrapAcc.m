% InitialParameterSet = [0 0 1];
% LearningRate = 0.01;
% Theta = 0;
% MaxNoOfIteration = 300;
% NumberOfBootStraps=500;
function [BootstrapAccuracy,FinalClassificationAccuracy]=Bootstrap(InitialParameterSet,LearningRate,Theta,MaxNoOfIteration,NumberOfBootStraps)
close all;
load('Class_1.mat');
load('Class_2.mat');

Class_1=Class_1;
Class_2=Class_2;

Data_1 = Class_1(:,2:end);
Data_2 = Class_2(:,2:end);
%disp(Data_1)


Feature1_Class1 = Data_1(:,5);
Feature1_Class2 = Data_2(:,5);
Feature2_Class1 = Data_1(:,7);
Feature2_Class2 = Data_2(:,7);

Data= [];
Data(:,1)=[Feature1_Class1(:); Feature1_Class2(:)];
Data(:,2)=[Feature2_Class1(:); Feature2_Class2(:)];
Data(1:length(Data_1),3)= ones;
Data(length(Data_1)+1:length(Data),3)= ones+1;

[Len,~] = size(Data);

% Split the data into two classes based on ClassSplit. 
Class1 =[];
Class2 =[];

i=1; j=1;
while i<=Len
    if Data(i,3)==1
        Class1(i,1)=Data(i,1);
        Class1(i,2)=Data(i,2);
    
    else
        Class2(j,1)=Data(i,1);
        Class2(j,2)=Data(i,2);
        j=j+1;
    end
    i=i+1;
end

% Calculate the number of training samples.
[ClassSize1,~]=size(Class1);
[ClassSize2,~]=size(Class2);

Train_Num1=ClassSize1;
Train_Num2=ClassSize2;

Train_Class1=[];

i=1;
while i<=ClassSize1
    Train_Class1(i,:)=Class1(i,:);
    i=i+1;
end
% 
% Split the data in class 2 into training and testing sets.
Train_Class2=[];
i=1;
while i<=ClassSize2
        Train_Class2(i,:)=Class2(i,:);
    i=i+1;
end

% Append all training and test sets
Train_Class1=[ones(Train_Num1,1) Train_Class1];
Train_Class2=-1.*[ones(Train_Num2,1) Train_Class2]; %%Normalized Training Class 2

% Prepare the training data including all training samples of classs 1 and 2.
Train_Data=zeros(Train_Num1 + Train_Num2,3);
Train_Data(1:Train_Num1,1)=Train_Class1(:,1); 
Train_Data(Train_Num1+1:Train_Num1+Train_Num2,1)=Train_Class2(:,1);  %21-50

Train_Data(1:Train_Num1,2:3)=Train_Class1(:,2:3);
Train_Data(Train_Num1+1:Train_Num1 + Train_Num2,2:3)=Train_Class2(:,2:3);

for y=1:NumberOfBootStraps

j=randi(9475,1,9475); %randomizing data
Training=j(1:ClassSize1);
Testing=j(ClassSize1+1:ClassSize1+ClassSize2);

for i=1:2675
    Train_Data1(i,:)=Train_Data(Training(i),:);
    Test_Data1(i,:)=Train_Data(Testing(i),:);
end

for i=1:2675
    Train_Data2(i,:)=Train_Data(Testing(i),:);
    Test_Data2(i,:)=Train_Data(Training(i),:);
end

% Implement basic gradient algorithm.
OptParams = InitialParameterSet;
PerceptronFunction = zeros(MaxNoOfIteration,1);
Criterion = 1;
NoOfIteration = 1;
m= @(v) (sqrt(v(1)*v(1)+v(2)*v(2)+v(3)*v(3)));

while ((Criterion>Theta))
    
    % Update the PerceptionFunction and The GradientOfCost.
        g=OptParams*Train_Data1';
        misclass=find(g<0,(Train_Num1+Train_Num2));       
        if (length(misclass)==1)
            GradientOfCost=Train_Data1(misclass,:);
        else        
            GradientOfCost=sum(Train_Data1(misclass,:));
        end
        
    % Update the optimized parameters.
        OptParams = OptParams + LearningRate*GradientOfCost;
        PerceptronFunction(NoOfIteration)=sum(OptParams.*GradientOfCost);
        
    % Update the value of the criterion to stop the algorithm.
       Criterion = m(GradientOfCost);
    %Break the algorithm when the NoOfIteration = MaxNoOfIteration.
        if(NoOfIteration == MaxNoOfIteration)
            break;
        end
        NoOfIteration = NoOfIteration + 1;   
end


gT=OptParams*Test_Data1';
NoOfAccuracy=find(gT<0,(length(Test_Data1)));

ClassificationAccuracy1 =((length(Test_Data1)-length(NoOfAccuracy))/length(Test_Data1)*100);

% Implement basic gradient algorithm for train data 2
while ((Criterion>Theta))
    
    % Update the PerceptionFunction and The GradientOfCost.
        g=OptParams*Train_Data2';
        misclass=find(g<0,(Train_Num1+Train_Num2));
        GradientOfCost=sum(Train_Data2(misclass,:));
        
    % Update the optimized parameters.
        OptParams = OptParams + LearningRate*GradientOfCost;
        PerceptronFunction(NoOfIteration)=sum(OptParams.*GradientOfCost);
    

    %Break the algorithm when the NoOfIteration = MaxNoOfIteration.
        if(NoOfIteration == MaxNoOfIteration)
            break;
        end
        NoOfIteration = NoOfIteration + 1;   
end

gT=OptParams*Test_Data2';
NoOfAccuracy=find(gT<0,(length(Test_Data2)));

ClassificationAccuracy2 =((length(Test_Data2)-length(NoOfAccuracy))/length(Test_Data2)*100);

FinalClassificationAccuracy(y)=(ClassificationAccuracy1+ClassificationAccuracy2)/2;
end

BootstrapAccuracy=num2str(mean(FinalClassificationAccuracy));
Acc=['The Bootstrap Classification Accuracy is ', BootstrapAccuracy,'%'];
disp(Acc);

end