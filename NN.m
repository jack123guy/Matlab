Weight1 = rand(10,12) - rand(10,12);  
%Input-hidden1 column��neuron�Ӽ� row��input�ƶq
Weight2 = rand(1,10) - rand(1,10);
%hidden1-hidden2


Threshold = rand(10,2);
%�����ȭn�A�]�w
Op = rand(1,2000);
%2000��input = 2000��output

%Neuron�������B�@
Input = rand(12,2000);% ex:12��input��,2000�����
A = mat2cell(Input,12,[1500,250,250])
Check = 0; %Validation Check
epoch = 0;
for p = 1:2000 %�C����J�@��input�|���@���ץ�

    for i = 1:10
        %Hidden1(i,1) = (Input' * Weight1(:,i)) - Threshold(:,1);
        Hidden1(i,p) = (Weight1 * Input(:,p) - Threshold(:,1)); % 10x1
    end
    

F1 = 2 * (1 + exp(1).^(-Hidden1)).^(-1) - 1; % tansig 10x1

    for i = 1:10
        Hidden2(i:p) = Weight2 * F1 - Threshold(:,2); % 1x1
    end

RealOp = Hidden2; % 1x1 1�دe�f �a�߯g
    
Error(p,1) = (Op.^2 - RealOp.^2) /2; % 1x1

    %validation check
    if p > 1 
        if Error(p,1) < Error(p-1,1)
            Check = Check + 1;
        end
        if Check > 10  %Validation Check = 10
            sprintf('Validation check limited');
            break;
        end    
    end
    
    
    
c = 0.01; %�ǲ߱`��

%�|��output���B��
%   for i = 1:4
%       R0(i:1) = (RealOp(i:1) - Op(i,1)) * Op(i,1) * (1 - Op(i,1));% �ǲ߫H�����p�� 4x1
%   end
%W2 = Weight2';
%    for i = 1:10
%       R1(i,1) = F1(i,1) * (1-F(i,1)) * (sum(W2)*R0)); % 10x1
%    end

%�@��output�p��
        R0 = (Op - RealOp) * RealOp * (1 - Op);
    for i = 1:10
        R1(i,1) = F1(i,1) * (1-F(i,1))* R0 * Weight2; %10x1
    end
    
    for i = 1:12
        delw1(:,i) = c * R1 * Input(i,p); % 10x12
    end
        delw2 = c * R0 * F1; % 1x1
        
        Weight1 = Weight1 -delw1;
        Weight2 = Weight2 -delw2;
end
epoch = 1:p
line(epoch,Error(epoch,1)); %�e�XError����