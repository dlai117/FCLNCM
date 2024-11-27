function [P, R, F, RI, FM, J] = Evaluate(label, result)

P = 0;
R = 0;
RI = 0;
num = numel(label);
I = max(label);
J = max(result);

%%  异常处理
% if size(label) ~= size(result)
%    error('First two arguments must be two vectors with the same length.'); 
% end
% if J > I
%    return
% end

%%
confusion_matrix = zeros(I, J);
for i = 1:I
    tmp = label==i;
    for j = 1:J
        confusion_matrix(i,j) = sum(tmp & result==j);
    end
end

TP_FP_num = sum(confusion_matrix, 2);
TP_FN_num = sum(confusion_matrix, 1);

TP_FP = 0;
for i = 1 : size(TP_FP_num, 1)
    if TP_FP_num(i)>=2
        TP_FP  = TP_FP + nchoosek(TP_FP_num(i), 2);
    end
end

TP_FN = 0;
for i = 1 : size(TP_FN_num, 2)
%     if TP_FN_num(i)~=0
    if TP_FN_num(i)>=2
        TP_FN  = TP_FN + nchoosek(TP_FN_num(i), 2);
    end
end

choice = nchoosek(1:num, 2);
TP = 0;
for i = 1:size(choice, 1)
   if label(choice(i, 1)) == label(choice(i, 2)) && ...
           result(choice(i, 1)) == result(choice(i, 2)) 
      TP = TP + 1; 
   end
end

all = nchoosek(num, 2);
% all1 = nchoosek(150, 2);
FP = TP_FP - TP;
FN = TP_FN - TP;
TN = all - TP - FP - FN;
P = TP / (TP + FP);
R = TP / (TP + FN);
F = 2*(P*R)/(P+R);
FM = sqrt((TP/(TP+FP))*(TP/(TP+FN)));
J = TP / (TP +FN+ FP);
RI = (TP + TN) / all;