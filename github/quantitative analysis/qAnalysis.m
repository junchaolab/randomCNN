% calculate Score
clc; clear; close all
load('cnnLibrary.mat')
load('randomLibrary.mat')

randomBenchmark = [];
cnnBenchmark =[];
totalBenchmark = 100;

vMax = 0.02; % m/s
for i = 1: totalBenchmark
    i
    v1 = rand * vMax;
    v2 = rand * (vMax - v1);
    v3 = vMax - v1 - v2;
    c1 = 0.5 + rand * 0.5;
    c2 = rand * c1;
    c3 = rand * c2;
    vScore = 0;
    cScore = 0;
    totalScore = 0;
  
    
    % match cnn library
    matched =[];
    ave_score = -inf;
    
    for j = 1:length(cnn_library)
        %j
        tv1 = cnn_library(j,1);
        tv2 = cnn_library(j,2);
        tv3 = cnn_library(j,3);
        tc1 = cnn_library(j,4);
        tc2 = cnn_library(j,5);
        tc3 = cnn_library(j,6);
        new_v_distance = (abs(v1 - tv1) ^ 3 + abs(v2 - tv2) ^ 3 + abs(v3 - tv3) ^ 3)^(1/3); % max = 1.4422
        new_c_distance = (abs(c1 - tc1) ^ 3 + abs(c2 - tc2) ^ 3 + abs(c3 - tc3) ^ 3)^(1/3); % max = 1.4422
        %total_distance = (abs(v1 - tv1) ^ 6 + abs(v2 - tv2) ^ 6 + abs(v3 - tv3) ^ 6 + abs(c1 - tc1) ^ 6 + abs(c2 - tc2) ^ 6 + abs(c3 - tc3) ^ 6)^(1/6); % max = 1.348
        vScore = -5000 * new_v_distance + 100;  %%%
        cScore = -69.3385 * new_c_distance + 100;  %%%
        tScore = (vScore + cScore) / 2;
        %tScore = 0.1*vScore + 0.9*cScore;
        new_ave_score = tScore;
        if new_ave_score >= ave_score
            ave_score = new_ave_score;
            match_vScore = vScore;
            match_cScore = cScore;
            matched = cnn_library(j,:);
        end
        %break
    end
    cnnBenchmark =[cnnBenchmark; v1 v2 v3 c1 c2 c3 matched match_vScore match_cScore ave_score];
    
    % match random library
    matched =[];
    v_distance = inf;
    c_distance = inf;
    vc_distance = inf;
    score_diff = inf;
    ave_score = -inf;
    
    for j = 1:length(random_library)
        %j
        tv1 = random_library(j,1);
        tv2 = random_library(j,2);
        tv3 = random_library(j,3);
        tc1 = cnn_library(j,4);
        tc2 = cnn_library(j,5);
        tc3 = cnn_library(j,6);
        new_v_distance = (abs(v1 - tv1) ^ 3 + abs(v2 - tv2) ^ 3 + abs(v3 - tv3) ^ 3)^(1/3); % max = 1.4422
        new_c_distance = (abs(c1 - tc1) ^ 3 + abs(c2 - tc2) ^ 3 + abs(c3 - tc3) ^ 3)^(1/3); % max = 1.4422
        vScore = -5000  * new_v_distance + 100;  %%%
        cScore = -69.3385 * new_c_distance + 100;  %%%
        tScore = (vScore + cScore) / 2;
        %tScore = 0.1*vScore + 0.9*cScore;
        new_ave_score = tScore;
        if new_ave_score >= ave_score
            ave_score = new_ave_score;
            match_vScore = vScore;
            match_cScore = cScore;
            matched = random_library(j,:);
        end
        %break
    end
    randomBenchmark =[randomBenchmark; v1 v2 v3 c1 c2 c3 matched match_vScore match_cScore ave_score];
    
    
end


eachScoreDiff = cnnBenchmark(:, 13:15) - randomBenchmark(:, 13:15);
vIncreasedBenchmark = [];
cIncreasedBenchmark = [];
tIncreasedBenchmark = [];
for k = 1: totalBenchmark
    if eachScoreDiff(k,1) > 0 & eachScoreDiff(k,2) > 0
        tIncreasedBenchmark = [tIncreasedBenchmark; randomBenchmark(k, 13:15) cnnBenchmark(k, 13:15) eachScoreDiff(k,:)];
    elseif eachScoreDiff(k,1) > 0
        vIncreasedBenchmark = [vIncreasedBenchmark; randomBenchmark(k, 13:15) cnnBenchmark(k, 13:15) eachScoreDiff(k,:)];
    elseif eachScoreDiff(k,2) > 0
        cIncreasedBenchmark = [cIncreasedBenchmark; randomBenchmark(k, 13:15) cnnBenchmark(k, 13:15) eachScoreDiff(k,:)];
    end
end
vIncreaseRatio = length(vIncreasedBenchmark) / totalBenchmark
cIncreaseRatio = length(cIncreasedBenchmark) / totalBenchmark
tIcreasedRatio = length(tIncreasedBenchmark) / totalBenchmark
totalIncreasedRatio = vIncreaseRatio + cIncreaseRatio + tIcreasedRatio

vAveScoreDiff = mean(vIncreasedBenchmark)
cAveScoreDiff = mean(cIncreasedBenchmark)
tAveScoreDiff  = mean(tIncreasedBenchmark)

randomVscore = mean(randomBenchmark(:,13))
randomCscore = mean(randomBenchmark(:,14))
randomTscore = mean(randomBenchmark(:,15))

cnnVscore = mean(cnnBenchmark(:,13))
cnnCscore = mean(cnnBenchmark(:,14))
cnnTscore = mean(cnnBenchmark(:,15))

%csvwrite('cnnBenchmark.csv',cnnBenchmark)
%csvwrite('randomBenchmark.csv', randomBenchmark)
