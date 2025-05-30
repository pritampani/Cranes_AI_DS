📰 **Fake News Detection using Hybrid AI Model**

This project aims to build a Fake News Detection System using a hybrid AI approach that analyzes both textual and visual content. The model integrates traditional machine learning for text classification and deep learning for image classification to accurately detect whether a news item is fake or real.

🚀 Features
	•	✅ Text-based Detection using:
	•	Logistic Regression
	•	Random Forest
	•	Naive Bayes
	•	🖼️ Image-based Detection using:
	•	Convolutional Neural Networks (CNN) with pre-trained VGG16
	•	📊 Hybrid Model combining both modalities for improved accuracy
	•	📚 Model evaluation with confusion matrix, accuracy, and visual plots
	•	🧪 Easy testing with custom user input (text/image)

🧠 Model Architecture

Text Classification
	•	Preprocessing: Tokenization, stop-word removal, TF-IDF vectorization
	•	Models: Logistic Regression, Random Forest, Naive Bayes
	•	Output: Fake or Real news label

Image Classification
	•	Input: News-related images
	•	Model: VGG16 (pretrained on ImageNet) with custom classification head
	•	Output: Fake or Real image classification

Combined Decision
	•	Weighted or parallel decision-making based on text and image results

🛠️ Tech Stack
	•	Language: Python
	•	Libraries:
	•	sklearn for classical ML models
	•	keras, tensorflow for deep learning
	•	matplotlib, seaborn for visualization
	•	nltk, re for text preprocessing
	•	Dataset: Text and image data for real and fake news

📁 Directory Structure

├── models/                  # Trained models
├── data/                    # Dataset files (text + images)
├── outputs/                 # Plots and confusion matrices
├── notebooks/               # Jupyter/Colab notebooks
└── README.md                # This file

⚙️ How to Run
	1.	Install dependencies:

pip install -r requirements.txt


	2.	Run the main notebook:
Open majaorproject.ipynb in Google Colab or Jupyter.
	3.	Test your own input:
	•	Input a news headline or body text.
	•	Upload an associated image (optional).
	•	Get a prediction: Fake or Real.

📊 Results
	•	Text Accuracy:
	•	Logistic Regression: ~96%
	•	Random Forest: ~93%
	•	Naive Bayes: ~95%
	•	Image Accuracy (VGG16): ~92%
	•	Combined Model Accuracy: ~94–96% depending on ensemble strategy

📌 Future Work
	•	Incorporate more complex multimodal fusion techniques
	•	Expand dataset size and diversity
	•	Deploy as a web app or browser extension
	•	Add support for live news article scraping and verification

👨‍💻 Author

Pritam Pani
B.Tech CSE (2026) — C.V. Raman Global University

📄 License

This project is licensed under the MIT License.
