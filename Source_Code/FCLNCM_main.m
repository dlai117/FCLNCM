% FCLNCM: Feature-Weight and Cluster-Weight Learning in Neutrosophic 𝑐-Means Clustering for Gene Expression Data
% Last code change 27/11/2024
%%
clear; clc;
%% 
m = 2; % m = 2
delta = 0.15;
max_iter = 100;
min_impro = 1e-5;
% 
Re = [];elapsedTime = [];
alpha_list = [-10,-8,-6,-4,-2,2,4,6,8,10];
for i = 1:10
for ALPHA = 1:1:1                % ALPHA -> {1,2,3...,8,9,10}
    for beta = 0:0.1:0.9 % beta -> {0.1,0.2,...,0.8,0.9}
        for w1 = 0:0.1:1          % w1 -> {0.01,0.02,...,0.49,0.5}
            for w2 = 0:0.1:1  % w2 -> {0.4,0.41,...,0.97,0.98}
                w3 = 1-w1-w2;
                alpha = alpha_list(ALPHA);
              
                if w3 > 0
%                     try
                        X = dlmread('West-2001.txt');
                        Y = X(:,end);
                              
                        X(:,end) = [];
                        C = max(Y);
                        N = size(X, 1);
                        S = size(X, 2);
                        X=(X-ones(N,1)*min(X))./(ones(N,1)*(max(X)-min(X)));    
%                         Y(Y==1) = 2;Y(Y==-1) = 1;
                     
                        [V, U] = fcm(X, C);
                        
%                         mf = (w1*T).^m;
%                         V = mf*X./(sum(mf, 2)*ones(1, M));
                        
                        [~, O] = max(U);
                        [~, A1] = label_map(O, Y);
                        V(A1, :) = V;
                        U(A1, :) = U;
                        T = U;
                        I = rand(1, N);
                        F = rand(1, N);
                        col_sum = sum(T)+I+F;
                        T = T./col_sum(ones(C, 1), :);
                        I = I./col_sum;
                        F = F./col_sum;
                        T = T'; I = I'; F = F';
%                         z = repmat(1/C,1,C);
%                         w = repmat(1/size(X,2),1,size(X,2));
%                         zimax = repmat(1/size(X,1),1,size(X,1));
                        %-------------------- Iteration ------------------%
                        [T,I,F,re] =  FCLNCM_func(m,delta,max_iter,min_impro,alpha,beta,w1,w2,w3,X,N,S,C,V,T);
                        
                        elapsedTime = [elapsedTime toc];
                        matrix = [T.';I.';F.'];
                        [max_matrix, I1] = max(matrix);
                        index4 = find(matrix(C+1,:) == max_matrix);
                        index5 = find(matrix(C+2,:) == max_matrix);
% 
                        I1 = I1';
%                         I1 = [I1 linspace(1,2100,2100)'];
                        index = [index4 index5];
                        label_1 = Y;
                        label_2 = Y;
                        I1(index, :) = [];
                        label_1(index, :) = [];
                        [new_label, A2] = label_map(I1, label_1);

%                         V(A2, :) = V;
%                         U1 = U(A2, :);

                        err = 0; ERR=[];
                        for j = 1:length(label_1)
                            if new_label(j) ~= label_1(j)
                                err = err + 1;
%                                  ERR = [ERR I1(j,2)];
                                 ERR = [ERR j];
                            end
                        end
                        Aerror = err/size(X, 1);
                        Aimpre = length(index4)/size(X, 1);
                        Aoutlier = length(index5)/size(X, 1);
% 
                        TT = T;
                        for tmp14 = 1:length(index4)
                            TT(index4(tmp14),re(index4(tmp14), :)) = TT(index4(tmp14),re(index4(tmp14), :))+I(index4(tmp14))/2;
                        end
                        Ti = TT./(sum(TT, 2)*ones(1, C));
                        [~,I2] = max(Ti.');
                        [New_label,~] = label_map(I2.',label_2);
                        Acc = length(find(New_label == label_2))/length(label_2);
                        nmi = NMI(label_2.',New_label.');
                        [P,R,F1,RI,FM,J] = Evaluate(label_2,New_label);
                        Re = [Re;[alpha,beta,w1,w2,w3,Acc,Aimpre,Aoutlier,nmi,P,R,F1,RI,FM,J]];
                         fprintf("ACC:%f,IMP:%f,OUT:%f,NMI:%f,P:%f,R:%f,F:%f,RI:%f,FM:%f,J:%f\n",Acc,Aimpre,Aoutlier,nmi,P,R,F1,RI,FM,J);
%                          fprintf("ERR:%f\n",Aerror);
%                     catch
%                         disp("Error");
%                     end
                end
            end
        end
    end
end
end
time = sum(elapsedTime)/10;