# Manufacturing Process Recommender (Flask + scikit-learn)

Run from the project folder, in this order:

```
pip install -r requirements.txt
python generate_data.py
python train_models.py
flask --app app run
```

Extras: `python test_rules.py` checks the twelve example rows. Set `SECRET_KEY` in the environment for any non-local use.

Note: classes are balanced to 400 rows each, so class frequencies do not match real-world frequencies. The model learns the decision logic, not how often each process is used.
