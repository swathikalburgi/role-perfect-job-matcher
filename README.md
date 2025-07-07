**RolePerfect – AI-Powered Job Matchmaker**
[![HuggingFace Spaces](https://img.shields.io/badge/Live%20on-Hugging%20Face-blue?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/swathikalburgi/job-recommender-app)

RolePerfect is a smart, intuitive app that helps users discover the top 10 job roles that match their profile — either by uploading a resume or selecting a job title.

Powered by NLP and cosine similarity, RolePerfect analyzes job roles and resumes using TF-IDF to deliver instant, personalized job recommendations.

**Demo**
👉 Try it here: [https://huggingface.co/spaces/swathikalburgi/job-recommender-app](https://huggingface.co/spaces/swathikalburgi/job-recommender-app)

**Dataset Source**
This project uses a cleaned version of the [AI Salary Dataset by Ruchi Bhatia](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries) from Kaggle.

**How It Works**
1. Upload your PDF resume
2. Or select a job title from the dropdown
3. App computes TF-IDF vectors and uses cosine similarity
4. You get the top 10 job matches with:
   - Job title
   - Location
   - Experience level
   - Salary (USD)
 
RolePerfect uses natural language processing (NLP) with TF-IDF and cosine similarity to match your resume or selected job title to the top 10 roles that best fit you.
I used a cleaned version of the Data Science Salaries Dataset because it covers a wide range of job titles, experience levels, locations, and salary data. From there, I focused on just the most relevant fields to keep things simple, fast, and useful!

**Built With**
- [Gradio](https://gradio.app/) – UI
- [Python](https://www.python.org/)
- [scikit-learn](https://scikit-learn.org/) – TF-IDF + cosine similarity
- [Pandas](https://pandas.pydata.org/)
- [PyPDF2](https://pypi.org/project/PyPDF2/) – Resume parsing
- [Hugging Face Spaces](https://huggingface.co/spaces) – App hosting


