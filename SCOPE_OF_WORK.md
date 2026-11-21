# Political Bias Radar Scope of Work

Introduction to Artificial Intelligence

Jose Ramirez, Mel Bailey, and Charles Freeman

## Research Question

Does training a BERT–LSTM classifier with a delayed streak reward improve its ability to predict the political leaning of news articles from publishers held out of training? We will predict the dataset’s left, center, and right labels and compare the reinforcement-learning model with the same architecture trained on article labels and with a traditional text classifier.

**Hypothesis.** We expect direct supervision to match or outperform streak-based training on held-out articles. The longest streak gives one score for a sequence of shuffled articles, while direct supervision gives feedback on every prediction.

## Data and Scope

We will start with the article-level dataset released by Baly et al. (2020) and use its publisher-based train, validation, and test split. We will verify text availability, check for duplicates across splits, and record class and publisher counts before training. Models will use the headline and the same fixed excerpt of cleaned article text. Publisher and topic fields will support data checks and error analysis.

The study covers an offline prototype for English-language political news. Its predictions estimate how the dataset labels an article’s political leaning. Center is a political-leaning label. The prototype will accept article text and return one of the three labels. Live news collection and public deployment are outside the project scope.

## Models and Training

**Traditional baseline.** Train multinomial naive Bayes on TF-IDF features and fit the vectorizer on the training split.

**Supervised comparison.** Use a frozen BERT encoder to produce one vector per article. Feed the sequence of vectors to an LSTM and train it to predict each article’s label using cross-entropy.

**Reinforcement learning.** Use the same encoder and LSTM architecture. Each episode contains a fixed number of shuffled articles. Reset the LSTM state at the start and sample one label per article. A correct prediction extends the streak and an error resets it. At episode end, use the longest streak as the sole reward and update the model with policy gradient (REINFORCE). The policy receives article vectors as input. The environment uses the reference labels to compute the reward.

## Evaluation Plan

Use the same articles and splits for every model. For the two LSTM models, match the article order, episode length, and training budget within each run. Select settings and checkpoints on the validation set. Evaluate each selected model on the held-out test set using one prediction per article. Use the same fixed test order and episode boundaries for both LSTMs and select the highest-scoring label at each step.

**Primary measure.** Report macro-F1, the average of the three class F1 scores. Show per-class F1, accuracy, and a confusion matrix to explain which labels are often confused.

**Training behavior.** Track longest streak and validation macro-F1 to see whether better training reward corresponds to better classification. Repeat the neural comparison with three random seeds and report the spread. Review a small sample of test errors for patterns by label or topic.

The matched LSTM comparison tests the overall effect of changing the training objective. If time permits, add a REINFORCE run that uses the total number of correct predictions as its terminal reward. That comparison tests whether rewarding consecutive correct predictions changes the result.

## Milestones and Deliverables

**Data and baseline.** Verify the released splits, record the preprocessing choices, and run the naive Bayes baseline.

**Model implementation.** Cache the BERT vectors, test the reward on a known sequence of predictions, and train both LSTMs.

**Evaluation and report.** Deliver runnable code with setup and run instructions, the split IDs and experiment settings, a results table and confusion matrix, and a short report on findings and limitations. Completion depends on producing a reproducible comparison. The result may favor either training method.

## Team Responsibilities

**Jose Ramirez.** Data preparation and any training-only augmentation, neural model implementation, and results analysis.

**Charles Freeman.** BERT embeddings, neural model implementation, and the reward and training loop.

**Mel Bailey.** Model architecture design and neural model implementation.

## Dataset Reference

Baly, R., Da San Martino, G., Glass, J., & Nakov, P. (2020). [We Can Detect Your Bias: Predicting the Political Ideology of News Articles](https://aclanthology.org/2020.emnlp-main.404/) (EMNLP). [Dataset and evaluation splits](https://github.com/ramybaly/Article-Bias-Prediction).
