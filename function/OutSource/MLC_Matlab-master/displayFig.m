clc
close all;
clear all;


% [1 1 0]	 Yellow
% [0 0 0]	 Black
% [0 0 1]	 Blue
% [0 1 0]	 Bright green
% [0 1 1]	 Cyan
% [1 0 0]	 Bright red
% [1 0 1]	 Pink
% [1 1 1]	 White
% [0.9412 0.4706 0]	Orange
% [0.251 0 0.502]	 Purple
% [0.502 0.251 0]	 Brown
% [0 0.251 0]	 Dark green
% [0.502 0.502 0.502]	Gray
% [0.502 0.502 1]	 Light purple
% [0 0.502 0.502]	 Turquoise
% [0.502 0 0]	 Burgundy 
% [1 0.502 0.502]	 Peach
dg = [0 0.4 0];
dr = [0.8 0 0];
db = [0 0.3 0.7];
dy = [0.6 0 0.6];	
dp = [0.502 0 0];


% global
x = 2:15;
pacc =   [0.5916, 0.5916, 0.5916, 0.5916, 0.5916, 0.5916, 0.5916, 0.5916, 0.5916, 0.5916, 0.5916, 0.5916, 0.5916, 0.5916];
pacci =  [0.5766, 0.5634, 0.5193, 0.5052, 0.4823, 0.4994, 0.4802, 0.4807, 0.4998, 0.4906, 0.5023, 0.4911, 0.5123, 0.5114];
paccmi = [0.6373, 0.6448, 0.6477, 0.6502, 0.6543, 0.6419, 0.6332, 0.6290, 0.6299, 0.6269, 0.6328, 0.6369, 0.6303, 0.6253];

% x1 = [0.05:0.01:0.09, 0.1:0.05:0.3];

figure('Position', [50 50 1000 800]);

plot(x, pacc, '-.x', 'MarkerEdgeColor', db, 'Color', db, 'MarkerSize', 13, 'MarkerFaceColor',db, 'LineWidth', 3);
hold on;
plot(x, pacci, '-o','MarkerEdgeColor', dg, 'Color', dg, 'MarkerSize', 10, 'MarkerFaceColor',dg, 'LineWidth', 3);
hold on;
plot(x, paccmi, '--s', 'MarkerEdgeColor', dr, 'Color', dr, 'MarkerSize', 10, 'MarkerFaceColor',dr, 'LineWidth', 3);
hold on;

axis([2 15 0.4 0.7]);

xlabel('$r$',  'Interpreter', 'Latex','FontSize', 40, 'FontWeight','bold');
ylabel('Exact-Match', 'FontSize', 20, 'FontWeight','bold');
lgd1 = legend('CC', 'CBMLC', 'CLMLC');
set(lgd1,  'fontsize', 20);
% errorbar(x, pacci, paccmi/10);
% hold on;
% errorbar(x, pacc, paccmi/5);

% [hleg1, hobj1] = legend('PACC', 'PACC-IG', 'PACC-MLIG','PACC-CFS', 'PACC-MLCFS');
% textobj = findobj(hobj1, 'type', 'text');
% set(textobj, 'Interpreter', 'latex', 'fontsize', 15);



