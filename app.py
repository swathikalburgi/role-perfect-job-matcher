import gradio as gr
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your job data 
job_data = pd.DataFrame({
    'job_title': [
        "data analyst", "ai scientist", "marketing data analyst",
        "bi data analyst", "nlp engineer", "data scientist", "product data scientist",
        "machine learning engineer", "cloud data engineer", "financial data analyst"
    ],
    'company_location': ["us", "us", "dk", "in", "es", "us", "us", "uk", "fr", "us"],
    'experience_level': ["en", "se", "se", "mi", "se", "en", "en", "en", "mi", "se"],
    'salary_in_usd': [120000, 140000, 85000, 50000, 100000, 130000, 115000, 125000, 105000, 95000]
})

# Extract unique titles for dropdown (plus a 'None' option)
unique_titles = ['None'] + sorted(job_data["job_title"].unique().tolist())

# Helper to extract resume text
def extract_text_from_pdf(file_path):
    if file_path is None:
        return ""
    reader = PdfReader(file_path.name if hasattr(file_path, "name") else file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Recommendation logic
def recommend_jobs(resume_file, selected_title):
    if selected_title == "None" and not resume_file:
        return pd.DataFrame([["Please upload a resume or select a job title.", "", "", ""]],
                            columns=["Option", "job_title", "company_location", "experience_level"])
    
    if selected_title != "None":
        input_text = selected_title
    else:
        input_text = extract_text_from_pdf(resume_file)
    
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform([input_text] + job_data["job_title"].tolist())
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    job_data["similarity"] = similarity_scores
    top_jobs = job_data.sort_values(by="similarity", ascending=False).head(10).copy()
    top_jobs.reset_index(drop=True, inplace=True)
    top_jobs.insert(0, "Option", [f"{i+1}" for i in range(len(top_jobs))])
    return top_jobs[["Option", "job_title", "company_location", "experience_level", "salary_in_usd"]]

# Gradio UI
with gr.Blocks(css="""
body { background-color: white; }
#title-box { background-color: #a8d0f0; padding: 1em; border-radius: 10px; text-align: center; margin-bottom: 1em; }
#title-text { font-size: 1.8em; font-weight: bold; color: #333333; }
.button-sm { font-size: 0.8em !important; padding: 0.2em 1em !important; }
""") as demo:

    with gr.Column():
        gr.Markdown('<div id="title-box"><span id="title-text">RolePerfect – AI-Powered Job Matchmaker</span></div>')

        with gr.Row():
            with gr.Column():
                resume = gr.File(label="Upload your resume (PDF only)", type="filepath", file_types=[".pdf"])
                clear_resume_btn = gr.Button("Clear Resume", elem_classes="button-sm")

                job_title = gr.Dropdown(choices=unique_titles, label="Or select a job title", value="None")
                clear_job_btn = gr.Button("Clear Job Title", elem_classes="button-sm")

                submit_btn = gr.Button("Submit", elem_id="submit-btn")
                clear_all_btn = gr.Button("Clear")

            with gr.Column():
                output_table = gr.Dataframe(headers=["Option", "job_title", "company_location", "experience_level", "salary_in_usd"],
                                            label="Top 10 Recommended Roles")

        # Functional logic
        submit_btn.click(fn=recommend_jobs, inputs=[resume, job_title], outputs=output_table)
        clear_all_btn.click(lambda: (None, "None", pd.DataFrame()), inputs=[], outputs=[resume, job_title, output_table])
        clear_resume_btn.click(lambda: None, inputs=[], outputs=resume)
        clear_job_btn.click(lambda: "None", inputs=[], outputs=job_title)

# Launch the app
demo.launch()
