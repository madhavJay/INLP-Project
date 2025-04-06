# Results So Far
- Spearman has gone up as compared to the baseline, indicating that the sentiment of global news in a day, provides us with a better idea of how the stock market moves
- MAPE has gone up, indicating that the integration of the dual streams may not be correct and may need to be modified, could also be cause of scaling factors, need to remove and check

# Changes to be made
- Add dropout as there are few samples so overfitting may occur
- Modify the financial branch to improve MAPE
- Try and add more datapoints to make the model more rigorous 

# Other tasks
- Ablation studies to monitor the effect of various bert layers, as well as how domain-specific bert affects the task
- Hyperparameter tuning