# NLP-Driven-Content-Based-Job-Recommendation-System
📌 Job Recommendation System
A content-based job recommendation system using text data to recommend jobs to users based on their experience, interests, and previously viewed jobs. This project uses TF-IDF, CountVectorizer, SpaCy embeddings, and KNN to compute similarities between job descriptions.

🚀 Features
Clean and preprocess job and user data by:

Imputing missing values

Removing stopwords, punctuation, and non-alphanumeric characters

Performing tokenization and lemmatization

Merging relevant text columns into a single corpus

Build datasets by:

Merging four CSVs to create unified job and user profiles

Extract features using:

TF-IDF

CountVectorizer

SpaCy Word Embeddings

Compute similarity via:

Cosine Similarity

K-Nearest Neighbors (KNN)

Evaluate and visualize job recommendations for selected users.

🗂️ Project Structure
perl
Copy
Edit
job-recommendation-system/
├── dataset/                           # Place your local CSVs here after download
│   ├── Combined_Jobs_Final.csv
│   ├── Job_Views.csv
│   ├── Experience.csv
│   └── Positions_Of_Interest.csv
├── job_recommendation_system.ipynb   # Full implementation in Jupyter Notebook
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
📥 Dataset
Due to size restrictions, the dataset is hosted externally.
🔗 Download the dataset from Kaggle

After downloading, extract the CSV files and place them in the dataset/ folder as shown in the project structure.

🧰 Requirements
Install required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Download required NLP models:

NLTK (Windows/macOS/Linux)
python
Copy
Edit
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
SpaCy (Windows)
In your Python environment or script, run:

python
Copy
Edit
import spacy
spacy.cli.download("en_core_web_lg")  # Downloads and links the model

nlp = spacy.load("en_core_web_lg")    # Loads the model
🧠 Recommender Techniques Used
TF-IDF + Cosine Similarity

CountVectorizer + Cosine Similarity

SpaCy Vector Embeddings + Similarity

KNN-Based Similarity Search

📈 Evaluation
Since there's no explicit user rating data, relevance is manually verified.
Example: Applicant.ID = 326 is selected to assess job matches based on job title and profile alignment.
