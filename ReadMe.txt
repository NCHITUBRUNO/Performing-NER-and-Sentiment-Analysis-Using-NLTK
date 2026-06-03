This project is base on : Performing NER and sentiment analysis on product reviews using NLTK; integrating extracted features into a recommender.

To Run Locally:
bash
   pip install nltk scikit-learn pandas numpy
    # Place the CSV in the same folder, then:
    python NER_with_NLTK.ipynb 
Firstly,
        Check all Required skills
        Preview the dataset by importing data using pandas and preview the top 5 sample data.
        Check the full shape of the data, keeping eyes on nulls 
NB: Once all this is done, you will have a clear picture of the dataset.

Secondly,
        Ensure that all depencies are fully installed.
        Run: pip install nltk scikit-learn pandas numpy --break-system-packages -q . if not possible, install them one after the other.
        With all this installed, import nltk and download all its necessary packages.

        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('maxent_ne_chunker_tab', quiet=True)
        print('All NLTK data downloaded')

Thirdly, Run the whole entire script, with NER taking 200 reviews to see.

Lastly, Lab Walk through
       
Dataset
22,641 women's clothing reviews across 1,179 unique clothing items, with columns: Review Text, Rating (1–5★), Recommended IND, Department/Class/Division, Age.

Step-by-Step Pipeline
Step    What it does                                                            NLTK tool used
1       Load & clean data, fill NaNs                                            pandas
2       Lowercase, strip punctuation, tokenise, remove stop-words               word_tokenize, stopwords
3       VADER Sentiment → compound score + Positive/Negative/Neutral label      SentimentIntensityAnalyzer
4       NER → tokenise → POS tag → NE chunk → extract entity names + types      pos_tag, ne_chunk, Tree
5       Aggregate all reviews per Clothing ID into one "super-document"         pandas groupby
6       TF-IDF vectorise each item (2000 features, bigrams)                     TfidfVectorizer
7       Build 1179×1179 cosine similarity matrix                                cosine_similarity
8       Recommender: 70% text cosine + 30% sentiment proximitycombined score
9       Bonus rankings by sentiment & recommendation rate—

Key Results
VADER Sentiment correctly tracks star ratings:

1★ → +0.21 (still positive on average — reviews are wordy but reviews trend soft)
5★ → +0.85 (strongly positive) ✓

NER extracts entities like GPE (cities/places), PERSON names from text — useful for filtering brand-related mentions.
Recommender example — querying item 1078 (Dresses/Dresses, 4.19★) returns items 1081, 1094, 1095 with similarity scores ~0.97, all in the same department with matching sentiment profiles.    