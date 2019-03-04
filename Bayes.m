%%Name: Syeda Zahra
% load Diabetes.mat;
% Data = Diabetes;
% FeatureX = 160;
% LabelColumn = 3;
% FeatureForClassification = 2;
function [PosteriorProbabilities,DiscriminantFunctionValue]=lab1(FeatureX,Data,FeatureForClassification, LabelColumn)
 close all;
% Get number of samples.
[ro,~] = size(Data);

% Select feature for classification (1 or 2).  
SelectedFeature=Data(:,FeatureForClassification);

% Get class labels.
Label=Data(:,LabelColumn);

%%%%%%%%Plot the histogram and box plot of features related to two classes%%%%%%%%%%
index1=find(Label==1);
index2=find(Label==2);

Class1=SelectedFeature(index1);
Class2=SelectedFeature(index2);

% Plot hist.
histogram(Class1,'FaceColor','r');
hold on
histogram(Class2, 'FaceColor', 'b');
title('Class 1 vs Class 2 Histogram');
legend('Class 1','Class 2');



% Plot boxplot.
figure;
boxplot([Class1, Class2]);
title('Class1 vs Class 2 Boxplot');


% pause
% close all
    
%%%%%%%%%%%%%%%%%%%%%%%Compute Prior Probabilities%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate prior probability of class 1 and class 2.

Pr1=(size(index1))/size(Label);
Pr2=(size(index2))/size(Label);

% % Calculate the mean and the standard deviation of the class conditional density p(x/w1).

m11=mean(Data(1:268,FeatureForClassification))
std11=std(Data(1:268,FeatureForClassification)) 
% % 
% % % Calculate the mean and the standard deviation of the class conditional density p(x/w2).
m12=mean(Data(269:536,FeatureForClassification));
std12=std(Data(269:536,FeatureForClassification));
% 
% % Calculate the class-conditional probability of class 1 and class 2.
cp11=(1/(sqrt(2*pi)*std11))*exp(-0.5*((FeatureX-m11)/std11)^2);
cp12=(1/(sqrt(2*pi)*std12))*exp(-0.5*((FeatureX-m12)/std12)^2);

%%%%Compute Posterior Probability%%%%%%

disp('Posterior probabilities for the test feature');

px=(Pr1*cp11)+(Pr2*cp12) %evidence p(x)

Pos1=(Pr1*cp11)/px
Pos2=(Pr2*cp12)/px

      

disp('Discriminant function value for the test feature');
L11=0;
L12=10;
L21=2;
L22=0;

R1=(L11*Pos1)+(L12*Pos2);
R2=(L21*Pos1)+(L22*Pos2);

%%%%%%%%%%%%%%%%finding threshold 1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
syms Thx

classp11=(1/(sqrt(2*pi)*std11))*exp(-0.5*((Thx-m11)/std11)^2);
classp12=(1/(sqrt(2*pi)*std12))*exp(-0.5*((Thx-m12)/std12)^2);

pThx=(Pr1*classp11)+(Pr2*classp12); 

Po11=(Pr1*classp11)/pThx;
Po12=(Pr2*classp12)/pThx;

Rr1=(L11*Po11)+(L12*Po12);
Rr2=(L21*Po11)+(L22*Po12);

Thr1=solve(Po11==Po12);
Thr2=solve(Rr1==Rr2);

disp('Threshold');
disp(int16(Thr1));  
disp(int16(Thr2));

% Compute the g(x) for min err rate class.
DiscriminantFunctionValue = R2-R1

  
if(DiscriminantFunctionValue<=0)
    disp('Choose class 2 for the test feature');

else
    disp('Choose class 1 for the test feature');



end
