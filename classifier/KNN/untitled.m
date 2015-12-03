close all
plot(K, hoga);hold on
plot(K, hogAccKNNs)
xlabel('k','FontSize',14)
ylabel('accuracy','FontSize',14)
h_legend = legend('KNN', 'distance-weighted KNN');
set(h_legend,'FontSize',14);
title('grid search of KNN', 'FontSize', 14)