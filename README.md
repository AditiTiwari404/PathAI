# PathAI v1

## Run on Mac using VS Code

1. Extract the ZIP and open the `PathAI_v1` folder in VS Code.
2. Open Terminal → New Terminal.
3. Create and activate a virtual environment (recommended):

   `python3 -m venv .venv`

   `source .venv/bin/activate`

4. Install packages:

   `python3 -m pip install -r requirements.txt`

5. Start the website:

   `python3 -m streamlit run app.py`

6. Open the local address shown in Terminal (usually `http://localhost:8501`).

## Features
- Career goal and skills input
- Professional matching using TF-IDF and cosine similarity
- Profile education, career journey, milestones and source links
- Skill-gap suggestions and a learning action plan
- SQLite storage for saved profiles and search history
- Clearly labelled demonstration updates feed

## Limitations
This is a student prototype with a small curated sample dataset. It does not scrape LinkedIn, access private data, monitor social media, or send live notifications. The updates page contains example categories, not verified current events. Similarity scores measure text similarity, not probability of career success. Verify profile facts and links before presenting them as current.


Version 2 additions: redesigned Discover People cards and an on-demand public research lookup using the OpenAlex API. The Updates page searches scholarly publications for saved professionals when you click the check button; it is not a LinkedIn integration or automatic notification service.


Readability patch: uses custom high-contrast peach callouts instead of the default Streamlit info banner.
