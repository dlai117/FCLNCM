clear; clc;
%% 
m = 2; % m = 2
% delta = 1e10;
delta = 0.15;
max_iter = 100;
min_impro = 1e-5;
% DWNCM
Re = [];elapsedTime = [];
alpha_list = [-10,-8,-6,-4,-2,2,4,6,8,10];
for i = 1:1
for ALPHA = 5:1:5                % ALPHA -> {1,2,3...,8,9,10}
    for beta = 0.1:0.1:0.1 % beta -> {0.1,0.2,...,0.8,0.9}
        for w1 = 0.07:0.01:0.07          % w1 -> {0.01,0.02,...,0.49,0.5}
            for w2 = 0.25:0.01:0.25  % w2 -> {0.4,0.41,...,0.97,0.98}
                w3 = 1-w1-w2;
                alpha = alpha_list(ALPHA);
              
                if w3 > 0
%                     try
%                         load("colon.mat");
%                         Y(Y==-1) = 2;
                        alpha =6;
                        X = dlmread('Synthetic_Datasets5.txt');
                        Y = X(:,end);
                        
%                         X(any(X == 1, 2), :) = []; % 删除Y中等于3的行
%                         X(X==2) = 1;X(X==3) = 2;
%                         
%                         Y(any(Y == 1, 2), :) = []; % 删除Y中等于3的行
%                         Y(Y==2) = 1;Y(Y==3) = 2;
                        
                        X(:,end) = [];
                        C = max(Y);
                        N = size(X, 1);
                        S = size(X, 2);
                        X=(X-ones(N,1)*min(X))./(ones(N,1)*(max(X)-min(X)));    % 对X做最大-最小归一化处理
%                         Y(Y==1) = 2;Y(Y==-1) = 1;
                     
                        [V, U] = fcm(X, C);
                        
%                         mf = (w1*T).^m;
%                         V = mf*X./(sum(mf, 2)*ones(1, M));
                        
                        [~, O] = max(U);
                        [~, A1] = label_map(O, Y);
                        V(A1, :) = V;
%                         V = [0,0,0,0;1,1,1,1;2,2,2,2];
                        V = [-5,5;5,-5];
                        U(A1, :) = U;
                        T = U;
                        I = rand(1, N);
                        F = rand(1, N);
                        col_sum = sum(T)+I+F;
                        T = T./col_sum(ones(C, 1), :);
                        I = I./col_sum;
                        F = F./col_sum;
                        T = T'; I = I'; F = F';
                        z = repmat(1/C,1,C);
                        w = repmat(1/size(X,2),1,size(X,2));
                        zimax = repmat(1/size(X,1),1,size(X,1));
                        %-------------------- Iteration ------------------%
                        ob = []; V_matrix{1} = V;
                        tic;
                        for item = 1:max_iter
                            % Computing imprecise cluster Vimax
                            Vimax = zeros(N, S);
                            re = zeros(N, 2);
                            for tmp1 = 1:N
                                if T(tmp1, 1) >= T(tmp1, 2)
                                    p = 1; q = 2;
                                else
                                    p = 2; q = 1;
                                end
                                if C >= 3
                                    for k = 3:C
                                        if T(tmp1, k) >= T(tmp1, p)
                                            q = p; p = k;
                                        elseif T(tmp1, k) > T(tmp1, q)
                                            q = k;
                                        end
                                    end
                                end
                                Vimax(tmp1, :) = (V(p, :)+V(q, :))/2;
                                re(tmp1, :) = [p q];
                            end
                    
                            % Computing neutrosophic partitions T, I, F
                            dist_v1 = ones(N, C);
%                             for tmp2 = 1:C
%                                 dist_v1(: ,tmp2) = sum(w.^alpha .* (z(tmp2).^beta) .* (X-V(tmp2, :)).^2, 2);
%                             end
                            
                            for tmp2 = 1:N
                                dist_v1(tmp2, :) = sum(w.^alpha .* ((z.^beta).'*ones(1, S)) .* (ones(C, 1)*X(tmp2, :)-V).^2, 2).';
                            end
                            dist_v1(dist_v1==0) = 1e-10;
                            
                            dist_vimax1 = sum(w.^alpha .* ((zimax.^beta).' .* (X-Vimax).^2), 2);
                            dist_vimax1(dist_vimax1==0) = 1e-10;
                            K1 = 1/w1*sum(dist_v1.^(-1/(m-1)), 2) + 1/w2*dist_vimax1.^(-1/(m-1)) + 1/w3*delta^(-2/(m-1));
                            T_old = T;
                            T = 1/w1*dist_v1.^(-1/(m-1))./(K1*ones(1, C));
                            I = 1/w2*dist_vimax1.^(-1/(m-1))./K1; 
                            F = 1/w3*delta^(-2/(m-1))./K1;   

                            % Computing feature weight w
                            dist_v2 = ones(C, S);
                            for tmp3 = 1:C
                                dist_v2(tmp3, :) = (z(tmp3).^beta) .* (w1*T(:, tmp3).').^m * (X-V(tmp3, :)).^2;
                            end
%                             dist_v2 = ones(N, S);
%                             for tmp3 = 1:N
%                                 dist_v2(tmp3, :) = (z.^beta) .* (w1*T(tmp3, :)).^m * (X(tmp3, :)-V).^2;
%                             end
                            Dist_v2 = sum(dist_v2, 1) + sum((zimax.^beta).' .* (w2*I).^m .* (X-Vimax).^2, 1);    
                            Dist_v2(Dist_v2==0) = 1e-10;
                            w = Dist_v2.^(-1/(alpha-1))./sum(Dist_v2.^(-1/(alpha-1)));

                            % Computing cluster weight z
                            dist_v3 = ones(C, S);
                            for tmp4 = 1:C
                                dist_v3(tmp4, :) = (w.^alpha) .* ((w1*T(:, tmp4).').^m * (X-V(tmp4, :)).^2);
                            end
                            Dist_v3 = sum(dist_v3, 2);
                            Dist_v3(Dist_v3==0) = 1e-10;
                            z = Dist_v3.^(-1/(beta-1))./sum(Dist_v3.^(-1/(beta-1)));
                            z = z';
                            
                            % Computing impreicse cluster weight zimax
                            dist_v4 = (w2*I).^m * (w.^alpha) .* (X-Vimax).^2;
                            Dist_v4 = sum(dist_v4, 2).';
                            Dist_v4(Dist_v4==0) = 1e-10;
                            zimax = Dist_v4.^(-1/(beta-1))./sum(Dist_v4.^(-1/(beta-1)));
                            
                            % Computing cluster cneter V
                            V = (w1*T).^m.'*(X)./(sum(((w1*T).^m).', 2)*ones(1, S));
                            V_matrix{item+1} = V;

                            % Computing objective function
                            dist_v5 = ones(N, C);
                            for tmp5 = 1:C
                                dist_v5(: ,tmp5) = sum(w.^alpha .* (z(tmp5).^beta) .* ((w1*T(:, tmp5)).^m .* (X-V(tmp5, :)).^2), 2);
                            end
                            ob_fun = sum(sum(dist_v5)) + sum(sum( w.^alpha .* ((zimax.^beta).' .* (w2*I).^m .* (X-Vimax).^2) )) + sum(delta^2*(w3*F).^m);
                            ob = [ob ob_fun];
                            
                            % check termination condition
                            if item > 1 && norm(T-T_old) < min_impro
                                break;
                            end
                        end
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
%                         label_2 = Y;
                        I1(index, :) = [];
                        label_1(index, :) = [];
                        [new_label, A2] = label_map(I1, label_1);

                        V(A2, :) = V;
                        U1 = U(A2, :);

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
%                         TT = T;
%                         for tmp14 = 1:length(index4)
%                             TT(index4(tmp14),re(index4(tmp14), :)) = TT(index4(tmp14),re(index4(tmp14), :))+I(index4(tmp14))/2;
%                         end
%                         Ti = TT./(sum(TT, 2)*ones(1, C));
%                         [~,I2] = max(Ti.');
%                         [New_label,~] = label_map(I2.',label_2);
%                         Acc = length(find(New_label == label_2))/length(label_2);
%                         nmi = NMI(label_2.',New_label.');
%                         [P,R,F1,RI,FM,J] = Evaluate(label_2,New_label);
%                         Re = [Re;[alpha,beta,w1,w2,w3,Acc,Aimpre,Aoutlier,nmi,P,R,F1,RI,FM,J]];
%                          fprintf("ACC:%f,IMP:%f,OUT:%f,NMI:%f,P:%f,R:%f,F:%f,RI:%f,FM:%f,J:%f\n",Acc,Aimpre,Aoutlier,nmi,P,R,F1,RI,FM,J);
                         fprintf("ERR:%f\n",Aerror);
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
%% visualization
% X = dlmread('Synthetic_Datasets5.txt');
% Y = X(:,end);
% X(any(X == 1, 2), :) = []; % 删除Y中等于3的行
% X(X==2) = 1;X(X==3) = 2;
% 
% Y(any(Y == 1, 2), :) = []; % 删除Y中等于3的行
% Y(Y==2) = 1;Y(Y==3) = 2;
% 
% X(:,end) = [];

% matrix = [T.';I.';F.'];
% [max_matrix, I1] = max(matrix);

% Find the data points with highest grade of membership in cluster 1
% index1 = find(matrix(1,:) == max_matrix);
% Find the data points with highest grade of membership in cluster 2
% index2 = find(matrix(2,:) == max_matrix);
% Find the data points with highest grade of membership in cluster 3
% index3 = find(matrix(3,:) == max_matrix);
% Find the data points with highest grade of membership in cluster 4
% index4 = find(matrix(4,:) == max_matrix);
% % Find the data points with highest grade of membership in cluster 5
% index5 = find(matrix(5,:) == max_matrix);
% % Find the data points with highest grade of membership in cluster 6
% index6 = find(matrix(6,:) == max_matrix);
% plot(X(index2,1),X(index2,2),'o','color','r');
% hold on;

% plot(X(index1,1),X(index1,2),'s','color','g');
% plot(X(index3,1),X(index3,2),'+','color','y');
% plot(X(index4,1),X(index4,2),'p','color','m');
% plot(X(index5,1),X(index5,2),'*','color','c');
% plot(X(index6,1),X(index6,2),'*','color','y');
% Plot the cluster centers

% 标记出错误的样本点
% plot(X(ERR,1),X(ERR,2),'s','color','k');
% plot(X(2001:end,1),X(2001:end,2),'d','color','y');
% 
% 
% V = V.*(max(X)-min(X))+min(X);
% plot(V(:,1),V(:,2),'+','color','k')
% plot(V_matrix{1}(:,1),V_matrix{1}(:,2),'+','color','k')
% for i = 2:size(V_matrix, 2)
%     V_matrix{i} = V_matrix{i}.*(max(X)-min(X))+min(X);
% end
% for i = 1:size(V_matrix, 2)-1
%     plot([V_matrix{i}(1,1),V_matrix{i+1}(1,1)],[V_matrix{i}(1,2),V_matrix{i+1}(1,2)],'--.','color','k')
% end
% for i = 1:size(V_matrix, 2)-1
%     plot([V_matrix{i}(2,1),V_matrix{i+1}(2,1)],[V_matrix{i}(2,2),V_matrix{i+1}(2,2)],'--.','color','k')
% end
% for i = 1:size(V_matrix, 2)-1
%     plot([(V_matrix{i}(1,1)+V_matrix{i}(2,1))/2,(V_matrix{i+1}(1,1)+V_matrix{i+1}(2,1))/2],[(V_matrix{i}(1,2)+V_matrix{i}(2,2))/2,(V_matrix{i+1}(1,2)+V_matrix{i+1}(2,2))/2],'--.','color','k')
% end
% axis equal;
% hold off; 