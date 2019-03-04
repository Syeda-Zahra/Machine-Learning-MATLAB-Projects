%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BME777: LAB 3: Multilayer Neural Networks.
% Statlog (Heart) Dataset: https://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29
% The first two features are contained in the first two columns.
% 1st feature: Resting blood pressure.
% 2nd feature: Oldpeak = ST depression induced by exercise relative to rest.
% The third column contains the label information.
% Class +1: Absence of heart disease.
% Class -1: Presence of heart disease.
% 50 samples were extracted for each class.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% 1. Eta: Learning rate.
% 2. Theta: Threhold for the cost function to escape the algorithm.
% 3. MaxNoOfIteration: Maximum number of iteration.
% 4. Problem: 1: XOR, otherwise: Classification problem with given dataset.
% 5. Data: the dataset used for training NN when problem ~=1.
% Outputs:
% 1. J: an array of cost.
% 2. w: trained weight matrix.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example of use:
% [J,w] = lab3(Eta,Theta,MaxNoOfIteration,Problem,0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load DataLab3.mat;
%Data=DataLab3;
%Eta=0.1;
% Theta=0.001;
% MaxNoOfIteration=300;
%Problem=1;
%use mesh grid command for x1x2 

function [J,w] = lab3(Eta ,Theta, MaxNoOfIteration,Problem, Data)
close all;

%initializations
if Problem == 1
x1 = [-1 -1 1 1];%inputs
x2 = [-1 1 -1 1]; 
t = [-1 1 1 -1]; %expected/teaching values

wih1 = [0.69 0.39 0.41];%weight vectors
wih2 = [0.65 0.83 0.37];
who1 = [0.42 0.59 0.56];

else
   wih1 = [0.69 0.39 0.41];%weight vectors
   wih2 = [0.65 0.83 0.37];
   who1 = [0.42 0.59 0.56];
 
%    wih1 = [2.69 2.39 2.41];%increased weight vectors to see what would happen to the accuracy.
%    wih2 = [2.65 2.83 2.37];
%    who1 = [2.42 2.59 2.56];
 


    % Add data to feature 1,2 and label vectors.
    x1 = Data(:,1)';
    x2 = Data(:,2)';
    t = Data(:,3)';
    
end


x = zeros(length(x1),3);
x(:,1)= 1;
x(:,2)=x1;
x(:,3)=x2;
n = length(x1); %number of samples

%bias initialization
Y = zeros(1,3);
Y(1,1) = 1; %bias 

J = zeros(1, MaxNoOfIteration);
r=0;

%Nececessary functions
A = @(x)  (exp(x)-exp(-x))/(exp(x)+exp(-x)); %activation function = tanh(x)
DA = @(x) (1-((A(x))^2));%deactivation 
mag= @(v) (sqrt(v(1)*v(1)+v(2)*v(2)+v(3)*v(3)));%magnitude function

while (1)
  
   m=0; %sample position
   Zk = zeros(1,length(x));
   r=r+1;
    
    % Initialize gradients of the three weight vectors.
    DeltaWih1 = [0 0 0]; % Inputs of bias, x1,x2 to hidden neuron 1.
    DeltaWih2 = [0 0 0]; % Inputs of bias, x1,x2 to hidden neuron 2.
    DeltaWho1 = [0 0 0]; % Inputs of bias, y1,y2 to output neuron.
    
    while(m<n)
         m = m + 1;
   
 
    netj1=(wih1)*x(m,:).';
    netj2=(wih2)*x(m,:).';
    
    Y(1,2)=A(netj1);
    Y(1,3)=A(netj2);
    netk=(who1)*Y.';
    Zk(m)=A(netk);
    
    sensitivity_k=(t(m)-Zk(m))*DA(netk);
    sensitivity_j1=DA(netj1)*sensitivity_k*who1(2);
    sensitivity_j2=DA(netj2)*sensitivity_k*who1(3);
    
    %updates
    DeltaWih1=DeltaWih1+Eta*sensitivity_j1*x(m,:); %should get (1x3)
    DeltaWih2=DeltaWih2+Eta*sensitivity_j2*x(m,:);
    DeltaWho1=DeltaWho1+Eta*sensitivity_k*Y;
   end
   
   %update weight vectors after running through all samples
   wih1=wih1+DeltaWih1;
   wih2=wih2+DeltaWih2;
   who1=who1+DeltaWho1;
 
%    r=r+1; %incrememnt of iterations
   
   %summation of error
   J(r)=0.5*(sum((Zk-t).^2));
%    criterion=mag(sensitivity_k*Y); %updating criterion
   
   %check condition to stop algorithm
   if (r==MaxNoOfIteration)
       break
   end
  
end

%this plots the vector spaces in 1D
%plot weight vectors in x and y space
% decision = @(x,a) (((-a(2)*x)-a(1))/a(3));
% x = [-1:1];
% 
% %decision boundary in x-space
% w1 = decision(x,wih1);
% w2 = decision(x,wih2);
% 
% figure
% plot(x,w1,'b'); hold on; plot(x,w2,'b');
% title('Decisition surface in X-space');
% 
% 
% %decision boundary in y-space
% w3=decision(x,who1);
% figure
% plot(w3); 
% title('Decision surface in Y-Space');

%error/iteration
figure
plot(J); 
title('Error per iteration');
ylabel('J(r)'); 
xlabel('Number of Epoch');

%%Calculating Classification Accuracy.
i=0;
for i= 1: length(Zk)
    if Zk(i)< 0       
     Zknew(i)= -1;   
    else       
     Zknew(i)= 1;
    end
    i = i +1;
end

Accuracy = 0;
for i= 1: length(Zk)
   if Zknew(i)==t(i)
       Accuracy=Accuracy+1;
   end  
end

ClassificationAccuracy =num2str((Accuracy/length(Zk))*100);
W=['Classification accuracy: ', ClassificationAccuracy,'%'];
disp(W);

%Decision Boundary
decision1 = -1:0.01:1;%goes from -1 to 1
decision2= -1:0.01:1;

%using meshgrid to produce decision boundaries.
[X1, X2] = meshgrid(decision1,decision2); %use this to get coordinates
LengthOfData = length(decision1)*length(decision2); 

xspace = zeros(LengthOfData,3);%for x-space
xspace(:,1) = 1; %This is the bias

totalLength = length(xspace);
z = zeros(totalLength,1);
w1 = zeros(totalLength,1);
w2 = zeros(totalLength,1);

k=1; 
for i = 1:length(decision1) %this for-loop helps plot the boundaries by plotting 
    for j = 1:length(decision2) 
        xspace(k,2) = X1(i,j);
        xspace(k,3) = X2(i,j);
        k=k+1;
    end
end


%for y-space
y(1,1)=1; %bias 

   for i = 1:totalLength
        netj1=(wih1)*xspace(i,:).';
        netj2=(wih2)*xspace(i,:).';
        netk=(who1)*y.';
        y(1,2)=A(netj1);
        y(1,3)=A(netj2);
        zk=A(netk); 
        
        w1(i)=y(1,2);
        w2(i)=y(1,3); 
        
        if zk> 0 %values between 1 or -1
            z(i) = 1;
        else 
            z(i) = -1;
        end
            
   end
   
   Xi =xspace(:,2);%x1
   Xii =xspace(:,3); %x2
   
   Index1 = z>0;
   Index2 = z<0; 
   z(Index1) = 1; 
   z(Index2) = -1;
   
figure; 
scatter3(Xi(Index1),Xii(Index1), z(Index1)); 
hold on; 
scatter3(Xi(Index2),Xii(Index2), z(Index2)); 
(view(0,90));%two see in 2D
title('X-Space'); 
xlabel('X2')
ylabel('X1')


figure;
scatter3(w1(Index1), w2(Index1), z(Index1)); 
hold on; 
scatter3(w1(Index2), w2(Index2), z(Index2)); 
(view(0,90)); 
title('Y-Space');
xlabel('Hidden Layer 1')
ylabel('Hidden Layer 2')

end



