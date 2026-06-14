%% 遗传算法主程序 - 校园快递驿站选址优化（改进GA，含精英保留）
% 编码方式：实数编码
% 选择策略：锦标赛选择
% 交叉算子：SBX交叉
% 变异算子：高斯变异
% 终止条件：最大代数200 或 连续30代无改善
% 改进点：引入精英保留策略，每代保留最优的2个个体

clear; clc; close all;
rng(42);  % 设置随机种子，确保结果可重复

%% ==================== 1. 问题数据定义 ====================
points = [5,  12, 120;
          10, 15, 180;
          15, 8,  100;
          18, 20, 150;
          22, 5,  80;
          25, 18, 160;
          30, 10, 140;
          32, 28, 110;
          38, 15, 170;
          42, 25, 190];

x_coords = points(:, 1);
y_coords = points(:, 2);
weights = points(:, 3);

% 搜索边界
x_min = 0; x_max = 50;
y_min = 0; y_max = 30;  % Y边界为30

%% ==================== 2. 遗传算法参数设置 ====================
pop_size = 100;         % 种群规模（修改为100）
max_gen = 200;          % 最大进化代数
pc = 0.8;               % 交叉概率
pm = 0.05;              % 变异概率
tournament_size = 2;    % 锦标赛规模
elite_count = 2;        % 【改进点】精英保留数量
sigma_x = 5;            % x方向高斯变异步长
sigma_y = 3;            % y方向高斯变异步长
eta_c = 20;             % SBX分布指数
stall_limit = 30;       % 停滞上限代数
epsilon = 1e-6;         % 改善阈值

%% ==================== 3. 初始化种群 ====================
population = zeros(pop_size, 2);
for i = 1:pop_size
    population(i, 1) = x_min + rand() * (x_max - x_min);
    population(i, 2) = y_min + rand() * (y_max - y_min);
end

%% ==================== 4. 记录变量初始化 ====================
best_fitness_history = zeros(max_gen, 1);
avg_fitness_history = zeros(max_gen, 1);
best_individual_history = zeros(max_gen, 2);
stall_counter = 0;

%% ==================== 5. 主循环 ====================
fprintf('========== 改进遗传算法（含精英保留，种群规模=100）开始优化 ==========\n');

for gen = 1:max_gen
    %% 5.1 计算适应度
    fitness = zeros(pop_size, 1);
    for i = 1:pop_size
        x = population(i, 1);
        y = population(i, 2);
        total_dist = 0;
        for j = 1:length(x_coords)
            dist = sqrt((x - x_coords(j))^2 + (y - y_coords(j))^2);
            total_dist = total_dist + weights(j) * dist;
        end
        fitness(i) = 1 / (total_dist + 1);
    end
    
    %% 5.2 记录最优解
    [best_fitness, best_idx] = max(fitness);
    best_individual_history(gen, :) = population(best_idx, :);
    best_fitness_history(gen) = best_fitness;
    avg_fitness_history(gen) = mean(fitness);
    best_obj = 1 / (best_fitness + 1e-10) - 1;
    
    %% 5.3 停滞判断
    if gen > 1
        improvement = abs(best_fitness_history(gen) - best_fitness_history(gen-1)) ...
                      / (best_fitness_history(gen-1) + 1e-10);
        if improvement < epsilon
            stall_counter = stall_counter + 1;
        else
            stall_counter = 0;
        end
    end
    
    % 显示进度
    if mod(gen, 20) == 0 || gen == 1
        fprintf('第 %d 代，最优适应度: %.6f，最优目标值: %.2f\n', gen, best_fitness, best_obj);
    end
    
    % 终止条件判断
    if gen >= max_gen || stall_counter >= stall_limit
        fprintf('\n算法终止于第 %d 代，', gen);
        if gen >= max_gen
            fprintf('达到最大进化代数。\n');
        else
            fprintf('连续 %d 代无显著改善（改善率 < %.2e）。\n', stall_limit, epsilon);
        end
        break;
    end
    
    %% 5.4 【改进点】精英保留：保留最优的elite_count个个体
    [~, sorted_idx] = sort(fitness, 'descend');
    elites = population(sorted_idx(1:elite_count), :);
    
    %% 5.5 锦标赛选择（生成父代池，数量为pop_size - elite_count）
    parent_pool = zeros(pop_size - elite_count, 2);
    for i = 1:(pop_size - elite_count)
        candidates_idx = randi(pop_size, tournament_size, 1);
        candidates_fitness = fitness(candidates_idx);
        [~, winner_local_idx] = max(candidates_fitness);
        winner_idx = candidates_idx(winner_local_idx);
        parent_pool(i, :) = population(winner_idx, :);
    end
    
    %% 5.6 SBX交叉操作
    offspring = zeros(pop_size - elite_count, 2);
    for i = 1:2:(pop_size - elite_count)
        p1 = parent_pool(i, :);
        if i+1 <= pop_size - elite_count
            p2 = parent_pool(i+1, :);
        else
            p2 = p1;
        end
        
        if rand() < pc
            for j = 1:2
                u = rand();
                if u <= 0.5
                    beta = (2 * u)^(1/(eta_c+1));
                else
                    beta = (1/(2*(1-u)))^(1/(eta_c+1));
                end
                c1 = 0.5 * ((1+beta)*p1(j) + (1-beta)*p2(j));
                c2 = 0.5 * ((1-beta)*p1(j) + (1+beta)*p2(j));
                offspring(i, j) = c1;
                offspring(i+1, j) = c2;
            end
        else
            offspring(i, :) = p1;
            offspring(i+1, :) = p2;
        end
    end
    
    %% 5.7 高斯变异操作
    for i = 1:size(offspring, 1)
        if rand() < pm
            offspring(i, 1) = offspring(i, 1) + sigma_x * randn();
            offspring(i, 2) = offspring(i, 2) + sigma_y * randn();
        end
    end
    
    %% 5.8 边界处理
    % 子代边界
    offspring(offspring(:,1) < x_min, 1) = x_min;
    offspring(offspring(:,1) > x_max, 1) = x_max;
    offspring(offspring(:,2) < y_min, 2) = y_min;
    offspring(offspring(:,2) > y_max, 2) = y_max;
    
    % 精英边界
    elites(elites(:,1) < x_min, 1) = x_min;
    elites(elites(:,1) > x_max, 1) = x_max;
    elites(elites(:,2) < y_min, 2) = y_min;
    elites(elites(:,2) > y_max, 2) = y_max;
    
    %% 5.9 【改进点】合并精英和子代，形成新一代种群
    population = [elites; offspring];
end

%% ==================== 6. 输出最终结果 ====================
final_fitness = zeros(pop_size, 1);
for i = 1:pop_size
    x = population(i, 1);
    y = population(i, 2);
    total_dist = 0;
    for j = 1:length(x_coords)
        dist = sqrt((x - x_coords(j))^2 + (y - y_coords(j))^2);
        total_dist = total_dist + weights(j) * dist;
    end
    final_fitness(i) = 1 / (total_dist + 1);
end

[best_final_fitness, best_final_idx] = max(final_fitness);
best_location = population(best_final_idx, :);
optimal_value = 1 / (best_final_fitness + 1e-10) - 1;

fprintf('\n========== 优化结果 ==========\n');
fprintf('最优驿站坐标: (%.4f, %.4f)\n', best_location(1), best_location(2));
fprintf('最优目标函数值: %.4f\n', optimal_value);

%% ==================== 7. 绘制收敛曲线 ====================
figure(1);
best_obj_history = 1 ./ (best_fitness_history(1:gen) + 1e-10) - 1;
avg_obj_history = 1 ./ (avg_fitness_history(1:gen) + 1e-10) - 1;
plot(1:gen, best_obj_history, 'b-', 'LineWidth', 1.5); hold on;
plot(1:gen, avg_obj_history, 'r--', 'LineWidth', 1.5);
xlabel('进化代数');
ylabel('加权距离总和');
title('改进遗传算法收敛曲线（含精英保留）');
legend('最优值', '平均值');
grid on;

%% ==================== 8. 绘制需求点分布及最优驿站位置 ====================
figure(2);
scatter(x_coords, y_coords, weights/2, 'filled', 'MarkerEdgeColor', 'k');
hold on;
scatter(best_location(1), best_location(2), 150, 'rp', 'LineWidth', 2, ...
        'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'none');
for i = 1:length(x_coords)
    text(x_coords(i)+1, y_coords(i)+1, num2str(i), 'FontSize', 10, 'FontWeight', 'bold');
end
xlabel('X坐标');
ylabel('Y坐标');
title('需求点分布及最优驿站位置（改进GA）');
legend('需求点（面积=需求量）', '最优驿站位置', 'Location', 'best');
xlim([0 50]);
ylim([0 30]);
grid on;

%% ==================== 9. 输出各需求点距离 ====================
fprintf('\n========== 各服务区域到驿站的距离 ==========\n');
for i = 1:length(x_coords)
    dist = sqrt((best_location(1) - x_coords(i))^2 + (best_location(2) - y_coords(i))^2);
    fprintf('区域 %d: 坐标 (%2d, %2d), 需求量 %3d, 距离 %.2f, 加权距离 %.2f\n', ...
            i, x_coords(i), y_coords(i), weights(i), dist, weights(i)*dist);
end

fprintf('\n改进遗传算法优化完成！\n');