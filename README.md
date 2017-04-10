<b>Context</b>:

Natural Language Processing(NLP), Text Similarity(lexical and semantic)

<b>Content</b>:

In each row of the included datasets(train.csv and test.csv), products X(description_x) and Y(description_y) are considered to refer to the same security(same_security) if they have the same ticker(ticker_x,ticker_y), even if the descriptions don't exactly match. You can make use of these descriptions to predict whether each pair in the test set also refers to the same security.

<b>Dataset info</b>:

<b>Train</b> - description_x, description_y, ticker_x, ticker_y, same_security. <b>Test</b> - description_x, description_y, same_security(to be predicted)
