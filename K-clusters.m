%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BME777: Lab 4:  Unsupervised Learning and Algorithm Independent Machine Learning
% Breast Tissue Dataset: https://archive.ics.uci.edu/ml/datasets/Breast+Tissue
% Feature 1: I0	Impedivity (ohm) at zero frequency.
% Feature 2: HFS high-frequency slope of phase angle.
% Feature 3: AREA area under spectrum.
% Only ther first 3 classes of the orginal dataset are used.
% 14 samples of each class were extracted for clustering. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% 1. Data: Data for clustering.
% 2. InitialMean1: Initial mean vector of class 1.
% 3. InitialMean2: Initial mean vector of class 2.
% 4. InitialMean3: Initial mean vector of class 3.
% 5. MaxNoOfIteration: Maximum number of iteration.
% Outputs:
% 1. FinalMean1: Final mean vector of class 1 given by k-means.
% 2. FinalMean2: Final mean vector of class 2 given by k-means.
% 3. FinalMean3: Final mean vector of class 3 given by k-means.
% 4. Label: Label for each sample in the original data.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example of use:
% load DataLab4.mat;
% Data = Breast_Tissue;
% InitialMean1 = [400 0.7 5800];
% InitialMean2 = [250 0.3 400];
% InitialMean3 = [300 1.1 1080];
% MaxNoOfIteration = 1;
% [FinMean1, FinMean2, FinMean3,Label] = lab4(Data,InitMean1,InitMean2,InitMean3,MaxNoOfIteration);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [FinalMean1,FinalMean2,FinalMean3,Label]=lab4(Data,InitialMean1,InitialMean2,InitialMean3,MaxNoOfIteration)
close all; 
c=3; %number of clusters

if InitialMean3~=0
   PrevMean = [InitialMean1; InitialMean2]; %since no x3 at this point
else
   PrevMean = [InitialMean1; InitialMean2; InitialMean3];
end

x1=Data(:,1);
x2=Data(:,2);
x3=Data(:,3);



Label = zeros(length(Data),1);
Itr = 0;

while(1)
    
      Itr = Itr + 1;
      %use norm of data - means for report 
    %%%%%%%%%%%%Compute Euclidean distance from each sample to the given means%%%%%%%%%%%%
    for i=1:length(Data)
       D1 = (Data(i,1)-InitialMean1(1))^2+(Data(i,2)-InitialMean1(2))^2+(Data(i,3)-InitialMean1(3))^2; 
       D2 = (Data(i,1)-InitialMean2(1))^2+(Data(i,2)-InitialMean2(2))^2+(Data(i,3)-InitialMean2(3))^2;
       D3 = (Data(i,1)-InitialMean3(1))^2+(Data(i,2)-InitialMean3(2))^2+(Data(i,3)-InitialMean3(3))^2;

%          if InitialMean3~=0
%            D3 = Data(:,3);
%           end
    %%%%%%%%%%%%%Identify the minimum distance from the sample to the means%%%%%%%%%%%%%%% 
   if InitialMean3~=0
            [~,Index] = min([D1 D2 D3]);
         else
           [~,Index] = min([D1,D2]);
        end
       
    %%%%%%%%%%%%Label the data samples based on the minimum euclidean distance%%%%%%%%%%%%  
       Label(i) = Index;
       T=table(i, Index);
       disp(T);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%Compute the new means%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    FinalMean1 = mean(Data(Label == 1,:))
    FinalMean2 = mean(Data(Label == 2,:))
    if InitialMean3~=0
        FinalMean3 = mean(Data(Label == 3,:))
    else
        FinalMean3 = 0;%%trying to make it only x1 x2
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%Check for criterion function%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%If criteria not met repeate the above%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if InitialMean3~=0
        CurrMean =[FinalMean1; FinalMean2; FinalMean3]
    else
        CurrMean =[FinalMean1; FinalMean2];
    end

    if (Itr==MaxNoOfIteration) % Check conditions to stop the algorithm.
        break;
    end
    PrevMean = CurrMean;  
    
end

figure;
     for j=1:c
        n = Label == j;
        cluster = Data(n,:);
        FinalMean(j,:) = mean(cluster); 
        hold on;
        if j==1
            scatter3(cluster(:,1), cluster(:,2),cluster(:,3),'b');%plot cluster 1
        end
        if j==2
            scatter3(cluster(:,1), cluster(:,2),cluster(:,3),'g');%plot cluster 2
        end
        if j==3
            scatter3(cluster(:,1), cluster(:,2),cluster(:,3),'r');%plot cluster 3
        end   
     end
    
    scatter3 (FinalMean(:,1), FinalMean(:,2), FinalMean(:,3), '*')%it will plot scatter plot of all three means
    view(50,50);
    title("Scatter plot of clusters and final means");
    legend("cluster 1", "cluster 2", "cluster 3","Final Means");
    grid on;
    hold off;

end

%********************************************

%Leave-One-Out and Bootstrap Methods
