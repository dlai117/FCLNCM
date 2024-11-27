% FCLNCM: Feature-Weight and Cluster-Weight Learning in Neutrosophic 𝑐-Means Clustering for Gene Expression Data
% Last code change 27/11/2024
%%
clear; clc;
%% 
m = 2;        % The default value of m is 2
delta = 0.01; % delta -> {0.01,0.1,1,10,100}
max_iter = 100;
min_impro = 1e-5;
 
Re = [];elapsedTime = [];
alpha_list = [-10,-8,-6,-4,-2,2,4,6,8,10];
for i = 1:10
for ALPHA = 1:1:10                   % ALPHA -> {1,2,3...,8,9,10}
    for beta = 0.1:0.1:0.9           % beta -> {0.1,0.2,...,0.8,0.9}
        for w1 = 0.01:0.01:0.98      % w1 -> {0.01,0.02,...,0.97,0.98}
            for w2 = 0.01:0.01:0.98  % w2 -> {0.01,0.02,...,0.97,0.98}
                w3 = 1-w1-w2;
                alpha = alpha_list(ALPHA);
              
                if w3 > 0
                    try
                        X = dlmread('West-2001.txt');
                        Y = X(:,end);
                              
                        X(:,end) = [];
                        C = max(Y);
                        N = size(X, 1);
                        S = size(X, 2);
                        X=(X-ones(N,1)*min(X))./(ones(N,1)*(max(X)-min(X)));    
                     
                        [V, U] = fcm(X, C);
                        
                        
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
                        %-------------------- Iteration ------------------%
                        [T,I,F,re] =  FCLNCM_func(m,delta,max_iter,min_impro,alpha,beta,w1,w2,w3,X,N,S,C,V,T);
                        
                        elapsedTime = [elapsedTime toc];
                        matrix = [T.';I.';F.'];
                        [max_matrix, I1] = max(matrix);
                        index4 = find(matrix(C+1,:) == max_matrix);
                        index5 = find(matrix(C+2,:) == max_matrix);
 
                        I1 = I1';
                        index = [index4 index5];
                        label_1 = Y;
                        label_2 = Y;
                        I1(index, :) = [];
                        label_1(index, :) = [];
                        [new_label, A2] = label_map(I1, label_1);

                        err = 0; ERR=[];
                        for j = 1:length(label_1)
                            if new_label(j) ~= label_1(j)
                                err = err + 1;
                                 ERR = [ERR j];
                            end
                        end
                        Aerror = err/size(X, 1);
                        Aimpre = length(index4)/size(X, 1);
                        Aoutlier = length(index5)/size(X, 1);
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
                    catch
                        disp("Error");
                    end
                end
            end
        end
    end
end
end
time = sum(elapsedTime)/10;