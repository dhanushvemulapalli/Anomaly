Number of TP samples: 5122
Number of AU samples: 7437

================================================================================

🔹 Training RandomForest (baseline)...

📌 RandomForest (baseline) @ threshold 0.5
Accuracy:  0.8847
Precision: 0.8505
Recall:    0.8704
F1 Score:  0.8603
ROC-AUC:   0.9411

Confusion Matrix (0.5):
[[1663  196]
 [ 166 1115]]

Classification Report (0.5):
              precision    recall  f1-score   support

           0       0.91      0.89      0.90      1859
           1       0.85      0.87      0.86      1281

    accuracy                           0.88      3140
   macro avg       0.88      0.88      0.88      3140
weighted avg       0.89      0.88      0.88      3140


🔧 Threshold Tuning for RandomForest (baseline)
Threshold | Precision | Recall | F1
0.10      | 0.647     | 0.974 | 0.777
0.15      | 0.697     | 0.964 | 0.809
0.20      | 0.728     | 0.954 | 0.826
0.25      | 0.753     | 0.944 | 0.838
0.30      | 0.777     | 0.928 | 0.846
0.35      | 0.799     | 0.920 | 0.855
0.40      | 0.819     | 0.907 | 0.861
0.45      | 0.838     | 0.891 | 0.863
0.50      | 0.852     | 0.870 | 0.861
0.55      | 0.863     | 0.846 | 0.855
0.60      | 0.879     | 0.820 | 0.849
0.65      | 0.896     | 0.781 | 0.834
0.70      | 0.903     | 0.730 | 0.807
0.75      | 0.910     | 0.658 | 0.764
0.80      | 0.920     | 0.581 | 0.712
0.85      | 0.933     | 0.478 | 0.632

🏆 Best Threshold for RandomForest (baseline): 0.45, Best F1: 0.8634

📌 RandomForest (baseline) @ BEST threshold 0.45
Accuracy:  0.8850
Precision: 0.8377
Recall:    0.8907
F1 Score:  0.8634
Confusion Matrix (best):
[[1638  221]
 [ 140 1141]]

================================================================================