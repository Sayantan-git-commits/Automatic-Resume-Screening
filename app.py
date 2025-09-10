import io
import re
import uuid
import smtplib
from difflib import SequenceMatcher
from email.mime.text import MIMEText

from flask import Flask, request, render_template_string, send_file, redirect, url_for, flash
import pdfplumber
import docx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Flask Setup ------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"

# ------------------ Configurations ------------------
ALLOWED_EXT = {"pdf", "docx", "txt"}
RESULTS_STORE = {}
SELECTED_THRESHOLD = 50.0  # Minimum match % to trigger email

# Replace with your Gmail + App Password
SMTP_USER = "sayantanisback28@gmail.com"
SMTP_PASS = "rpnpehkvpmwmaxet"  # <-- Use your Gmail App Password here

# ------------------ Recommended Skills by Job ------------------
JOB_SKILLS = {
    "data analyst": ["python", "sql", "excel", "tableau", "power bi", "statistics", "machine learning"],
    "machine learning engineer": ["python", "tensorflow", "pytorch", "scikit-learn", "ml algorithms", "nlp"],
    "web developer": ["html", "css", "javascript", "react", "nodejs", "mongodb"],
    "data scientist": ["python", "pandas", "numpy", "matplotlib", "scikit-learn", "tensorflow"],
}

# ------------------ Stopwords ------------------
STOPWORDS = set("a about above after again against all am an and any are aren't as at be because been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves".split())

# ------------------ Helper Functions ------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens)

def extract_text_from_pdf(data: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

def extract_text_from_docx(data: bytes) -> str:
    document = docx.Document(io.BytesIO(data))
    return " ".join([p.text for p in document.paragraphs])

def extract_text(filename: str, data: bytes) -> str:
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(data)
    elif ext == "docx":
        return extract_text_from_docx(data)
    else:
        return data.decode("utf-8", errors="ignore")

# ------------------ Email Sending ------------------
def send_selection_email(to_email, candidate_name):
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587

    subject = "Congratulations! You Have Been Shortlisted 🎉"
    body = f"""
    Dear {candidate_name},

    Congratulations! Based on our resume screening, you have been shortlisted
    for the next round.

    We will contact you shortly with further details.

    Best regards,
    HR Team
    """

    msg = MIMEText(body, "plain")
    msg['Subject'] = subject
    msg['From'] = SMTP_USER
    msg['To'] = to_email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, [to_email], msg.as_string())
        print(f"✅ Email sent successfully to {to_email}")
        return True
    except Exception as e:
        print(f"❌ Error sending email to {to_email}: {e}")
        return False

# ------------------ Matching Logic ------------------
def normalize_skill(skill: str) -> str:
    """Normalize skills by removing case, hyphens, underscores, dots, and spaces."""
    skill = skill.lower()
    skill = re.sub(r'[-_.]', '', skill)
    skill = re.sub(r'\s+', '', skill)
    return skill.strip()

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def calculate_match(jd_text, resume_text):
    jd_text = jd_text.strip().lower()
    resume_text = clean_text(resume_text).lower()

    # Use predefined job skills if job title provided
    if jd_text in JOB_SKILLS:
        required_skills = set([normalize_skill(s) for s in JOB_SKILLS[jd_text]])
    else:
        required_skills = set([normalize_skill(s) for s in re.split(r'[, ]+', jd_text) if s.strip()])

    resume_text_normalized = normalize_skill(resume_text)
    resume_words = set([normalize_skill(w) for w in resume_text.split()])

    matched_skills = set()
    for skill in required_skills:
        if skill in resume_text_normalized:
            matched_skills.add(skill)
        else:
            for word in resume_words:
                if similar(skill, word) > 0.85:
                    matched_skills.add(skill)
                    break

    match_percent = (len(matched_skills) / len(required_skills)) * 100 if len(required_skills) > 0 else 0

    if required_skills.issubset(matched_skills):
        match_percent = 100.0

    return round(match_percent, 2), matched_skills, required_skills

# ------------------ HTML Template ------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Automatic Resume Screening</title>
    <style>
        body { font-family: Arial; background-color: #f5f5f5; padding: 20px; }
        .container { background: #fff; padding: 20px; border-radius: 10px; max-width: 700px; margin: auto; }
        h1 { color: #333; }
        input, textarea { width: 100%; margin-bottom: 10px; padding: 10px; }
        button { background-color: #4CAF50; color: white; padding: 10px; border: none; cursor: pointer; border-radius: 5px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }
        .alert { background-color: #d4edda; color: #155724; padding: 10px; margin-bottom: 15px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Automatic Resume Screening</h1>
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                <div class="alert">
                    {% for message in messages %}
                        {{ message }}
                    {% endfor %}
                </div>
            {% endif %}
        {% endwith %}
        <form method="POST" enctype="multipart/form-data">
            <label>Job Description / Job Title:</label>
            <textarea name="jd" rows="5" placeholder="E.g. Data Analyst or Python, SQL, Excel" required></textarea>
            <label>Upload Resumes:</label>
            <input type="file" name="resumes" multiple required>
            <label>Candidate Emails (comma separated, order matches resumes):</label>
            <input type="text" name="emails" required>
            <button type="submit">Analyze</button>
        </form>
        {% if results is not none and results|length > 0 %}
            <h2>Ranked Candidates</h2>
            <table>
                <tr><th>Filename</th><th>Match %</th></tr>
                {% for row in results %}
                    <tr>
                        <td>{{ row[0] }}</td>
                        <td>{{ row[1] }}%</td>
                    </tr>
                {% endfor %}
            </table>
            <a href="{{ url_for('download_results', token=token) }}">
                <button>Download CSV</button>
            </a>
        {% endif %}
    </div>
</body>
</html>
"""

# ------------------ Flask Routes ------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        jd = request.form['jd']
        resumes = request.files.getlist('resumes')
        emails = [e.strip() for e in request.form['emails'].split(",")]

        data = []
        for idx, file in enumerate(resumes):
            filename = file.filename
            candidate_name = filename.rsplit('.', 1)[0]
            if '.' in filename and filename.rsplit('.', 1)[-1].lower() in ALLOWED_EXT:
                raw = extract_text(filename, file.read())
                cleaned = clean_text(raw)
                email = emails[idx] if idx < len(emails) else ""
                match_percent, _, _ = calculate_match(jd, cleaned)
                data.append((filename, candidate_name, cleaned, email, match_percent))

        if data:
            df = pd.DataFrame(data, columns=['filename', 'candidate_name', 'text', 'email', 'match_percent'])
            df = df.sort_values(by='match_percent', ascending=False)

            emails_sent = 0
            for _, row in df.iterrows():
                if row['match_percent'] >= SELECTED_THRESHOLD and row['email']:
                    if send_selection_email(row['email'], row['candidate_name']):
                        emails_sent += 1

            flash(f"✅ Analysis complete! Emails sent to {emails_sent} selected candidates.")

            token = str(uuid.uuid4())
            RESULTS_STORE[token] = df[['filename', 'match_percent']].to_csv(index=False).encode()

            return render_template_string(
                HTML_TEMPLATE,
                results=df[['filename', 'match_percent']].values,
                token=token
            )
        else:
            flash("⚠️ No valid resumes uploaded!")
            return render_template_string(HTML_TEMPLATE, results=None)

    return render_template_string(HTML_TEMPLATE, results=None)

@app.route('/download/<token>')
def download_results(token):
    data = RESULTS_STORE.get(token)
    if not data:
        return redirect(url_for('index'))
    return send_file(io.BytesIO(data), download_name='ranked_resumes.csv', as_attachment=True)

# ------------------ Run Flask ------------------
if __name__ == '__main__':
    app.run(debug=True, port=8000)


