# Fashion Review Recommendation Predictor

Streamlit app (NLP practical exam, Set C): TF-IDF + Logistic Regression trained on women's
clothing reviews to predict whether a customer would recommend a product.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py          # or: streamlit run streamlit/app.py from the repo root
```
Required NLTK data (stopwords, wordnet, punkt_tab, averaged_perceptron_tagger_eng) is downloaded automatically on first start.

## Notes
- `recommend_pipeline.pkl` was trained with scikit-learn 1.5.2; keep the pinned version in `requirements.txt`.
- The "Threshold" slider sets the probability of "not recommended" above which a review is flagged
  (and sent to the quality team).
