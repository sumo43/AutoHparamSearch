### Auto Hyperparameter Search

This python script uses GPT3.5 to automatically find the best hyperparameters for your machine learning script.
The only requirement is that the script prints out the accuracy or loss at the end of training.

### Example usage

```
pip3 install -r requirements.txt

OPENAI_API="YOUR API KEY HERE"

python3 ahs.py examples/sklearn_example.py
```
