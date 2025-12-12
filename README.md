🧠 Toxic Comment Classifier using DistilBERT

🚀 A Machine Learning Powered Toxicity Detection Web App
This project is a Toxic Comment Classifier built using a fine-tuned DistilBERT model, deployed on Hugging Face Spaces, and accessible through a simple, intuitive Streamlit interface.

It predicts whether a given comment is Toxic or Not Toxic with high accuracy.

🔗 Live Demo

👉 Hugging Face Space:
https://huggingface.co/spaces/Pau22/Toxic_Comment_Classifier_using_DistilBERT

👉 Model on Hugging Face Hub:
https://huggingface.co/Pau22/distilbert-toxic-model

✨ Features

🔍 Real-time Toxic Comment Detection

🧠 Powered by DistilBERT, fine-tuned on Jigsaw Toxic Comment dataset

⚡ Fast inference using Hugging Face pipeline()

🎨 Streamlit-based UI with clean, modern design

📌 Includes example toxic & non-toxic comments

📊 Displays model evaluation metrics

📥 Easy to clone and run locally

🌐 Fully deployable on Hugging Face Spaces or Render

🧩 Tech Stack
Component	Technology
NLP Model	DistilBERT + Transformers
Deployment	Hugging Face Spaces
Frontend	Streamlit
Backend	Hugging Face Inference pipeline
Dataset	Jigsaw Toxic Comment Classification
📁 Project Structure
📦 Toxic Comment Classifier
│
├── app.py                # Streamlit Application
├── requirements.txt      # Python Dependencies
├── README.md             # Project Documentation
└── (No model files needed — loaded directly from HF Hub)

⚙️ Installation (Run Locally)
1. Clone the Repository
git clone https://github.com/USERNAME/REPO_NAME.git
cd REPO_NAME

2. Install Dependencies
pip install -r requirements.txt

3. Run the Streamlit App
streamlit run app.py

🧠 Model Details

Your model is located on the Hugging Face Hub:
➡ https://huggingface.co/Pau22/distilbert-toxic-model

Training Summary

Metric	Score
Loss	0.1062
Accuracy	0.9685
Precision	0.8337
Recall	0.8292
F1 Score	0.8314

Trained for 2 epochs using DistilBERT on the Jigsaw Toxic Comment dataset.

💡 Example Inputs
Toxic:

“You are the worst person ever.”

“Shut up you idiot.”

“You f*cking clown.”

Non-Toxic:

“Thank you for your help!”

“Have a lovely day!”

“I appreciate your effort.”

🖼️ Screenshots
🔹 UI Preview

Add your screenshot here (optional)

📜 License

This project is licensed under the MIT License — feel free to use, modify, and distribute.

🙌 Author

Pau22
🔗 Hugging Face: https://huggingface.co/Pau22

🔗 GitHub: https://github.com/paurasm22

⭐ Support the Project

If you found this useful, please consider:

⭐ Starring the GitHub repository

🤝 Sharing it with others

💬 Giving feedback or suggestions
