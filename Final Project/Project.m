

%disp(Data)
function [JackKnife]=LOO(ClassSplit,InitialParameterSet,LearningRate,Theta,MaxNoOfIteration);
%convert the files
%Class_1=csvread('BME777_BigData_Extract_class1.csv', 1, 0);
%save('Class_1.mat','Class_1');

%Class_2=csvread('BME777_BigData_Extract_class2.csv', 1, 0);
%save('Class_2.mat','Class_2');

load('Class_1.mat');
load('Class_2.mat');

Class_1=Class_1;
Class_2=Class_2;

Data_1 = Class_1(:,2:end);
Data_2 = Class_2(:,2:end);
%disp(Data_1)


Feature1_Class1 = Data_1(:,9);
Feature1_Class2 = Data_2(:,9);
Feature2_Class1 = Data_1(:,10);
Feature2_Class2 = Data_2(:,10);

Data= [];
Data(:,1)=[Feature1_Class1(:); Feature1_Class2(:)];
Data(:,2)=[Feature2_Class1(:); Feature2_Class2(:)];
Data(1:length(Data_1),3)= ones;
Data(length(Data_1)+1:length(Data),3)= ones+1;

[Len,~] = size(Data);

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
[Class_Size1,~]=size(Class1);
[Class_Size2,~]=size(Class2);
test=1000;

% Calculate the number of training samples.
Train_Num1=Class_Size1;
Train_Num2=Class_Size2;



while (test<=length(Data))
    
% Split the data in class 1 into training sets. 
Train_Class1=[];

i=1;
while i<=Class_Size1
    Train_Class1(i,:)=Class1(i,:);
    i=i+1;
end

% Split the data in class 2 into training sets.
Train_Class2=[];

i=1;
while i<=Class_Size2
        Train_Class2(i,:)=Class2(i,:);
    i=i+1;
end

% Append all training and test sets
Train_Class1=[ones(Train_Num1,1) Train_Class1]; %finding all 1's
Train_Class2=[ones(Train_Num2,1) Train_Class2].*-1; %finding all 1's and normalizing training data for class 2

% Prepare the training data including all training samples of classs 1 and 2.
Train_Data=zeros(Train_Num1 + Train_Num2,3);
Train_Data(1:Train_Num1,1)=Train_Class1(:,1); 
Train_Data(Train_Num1+1:Train_Num1+Train_Num2,1)=Train_Class2(:,1);  %21-50
Train_Data(1:Train_Num1,2:3)=Train_Class1(:,2:3);
Train_Data(Train_Num1+1:Train_Num1 + Train_Num2,2:3)=Train_Class2(:,2:3);

Test_Data=zeros(1,3);
Test_Data(1,:)=Train_Data(test,:);

%Takes out the Test 1x3
Train_Data(test,:)=[];

% Implement basic gradient algorithm.
OptParams = InitialParameterSet;
PerceptronFunction = zeros(MaxNoOfIteration,1);
Criterion = 1;
NoOfIteration = 1;

m= @(v) (sqrt(v(1)*v(1)+v(2)*v(2)+v(3)*v(3)));

while ((Criterion>Theta))
    % Update the PerceptionFunction and The GradientOfCost.
        g=OptParams*Train_Data';
        misclass=find(g<0);
     
        misclass=find(g<0,(Train_Num1+Train_Num2));
        if (length(misclass)==1)
            GradientOfCost=Train_Data(misclass,:);
        else        
            GradientOfCost=sum(Train_Data(misclass,:));
        end      
        
    % Update the optimized parameters.
        OptParams = OptParams + LearningRate*GradientOfCost;
        PerceptronFunction(NoOfIteration)=sum(OptParams.*GradientOfCost);
        
    % Update the value of the criterion to stop the algorithm.
         Criterion = m(GradientOfCost);
         
    %Break the algorithm 
        if(NoOfIteration == MaxNoOfIteration)
            break;
        end
        NoOfIteration = NoOfIteration + 1;   
end

% Calculate the classification accuracy of the predictions on the test data.
gTest=OptParams*Test_Data';
NoOfAccuracy=find(gTest<0,(length(Test_Data)));

ClassificationAccuracy =((1-length(NoOfAccuracy))/1*100);
accuracy(test)=ClassificationAccuracy;

test=test+1;
end

JackKnife=mean(accuracy);

end
