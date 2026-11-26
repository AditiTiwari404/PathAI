import sqlite3
from datetime import datetime
import re
import pandas as pd
import requests
import html
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="PathAI", page_icon="🧭", layout="wide")

PEOPLE = [
{"name":"Fei-Fei Li","field":"AI Research","speciality":"Computer vision · Human-centred AI","education":"B.A. in Physics, Princeton University; Ph.D. in Electrical Engineering, Caltech.","journey":"Built an academic research career in computer vision. Her work helped establish ImageNet as an important resource for computer vision research.","skills":"python machine learning artificial intelligence computer vision research statistics deep learning","milestones":["Studied physics at Princeton","Completed a Ph.D. at Caltech","Built a research career in computer vision","Led work associated with ImageNet"],"now":"AI researcher and professor; check the institutional profile for current appointments.","source":"Stanford profile","url":"https://profiles.stanford.edu/fei-fei-li"},
{"name":"Katherine Johnson","field":"Mathematics & Aerospace","speciality":"Applied mathematics · Spaceflight","education":"Graduated from West Virginia State College with degrees in mathematics and French.","journey":"Worked as a mathematician at NACA/NASA and contributed calculations used in major American spaceflight missions.","skills":"mathematics physics aerospace engineering analytical thinking programming problem solving research","milestones":["Completed college studies in mathematics","Joined the NACA mathematics team","Contributed calculations for space missions","Became a recognised NASA mathematician"],"now":"Historical figure; her work is part of NASA's history.","source":"NASA biography","url":"https://www.nasa.gov/people/katherine-johnson/"},
{"name":"Marie Curie","field":"Science","speciality":"Physics · Chemistry · Radioactivity","education":"Studied physics and mathematics at the University of Paris (Sorbonne).","journey":"Pursued scientific study in Paris and conducted pioneering research on radioactivity.","skills":"physics chemistry mathematics research experimentation scientific writing laboratory","milestones":["Pursued higher studies in Paris","Conducted research on radioactivity","Received the 1903 Nobel Prize in Physics","Received the 1911 Nobel Prize in Chemistry"],"now":"Historical figure; her scientific legacy continues to influence research.","source":"Nobel Prize biography","url":"https://www.nobelprize.org/prizes/physics/1903/marie-curie/biographical/"},
{"name":"A. P. J. Abdul Kalam","field":"Aerospace Engineering","speciality":"Aerospace · Defence research","education":"Studied physics at St. Joseph's College and aerospace engineering at Madras Institute of Technology.","journey":"Worked in India's aerospace and defence research programmes, later served as President of India and inspired students.","skills":"physics aerospace engineering research leadership communication mathematics design","milestones":["Studied physics","Specialised in aerospace engineering","Worked in India's space and defence programmes","Served as President of India"],"now":"Historical figure, remembered for science, education and public service.","source":"President of India","url":"https://www.presidentofindia.gov.in/"},
{"name":"Radia Perlman","field":"Computer Science","speciality":"Computer networks · Network protocols","education":"Studied mathematics and computer science at MIT.","journey":"Developed foundational networking technologies and is known for the Spanning Tree Protocol.","skills":"computer science networking algorithms programming software engineering mathematics problem solving","milestones":["Studied mathematics and computer science","Worked on network protocols","Developed the Spanning Tree Protocol","Contributed to network engineering"],"now":"Technology professional; consult her public biography for current details.","source":"Public professional website","url":"https://www.radialperlman.com/"},
{"name":"Guido van Rossum","field":"Software Engineering","speciality":"Programming languages · Python","education":"Studied mathematics and computer science at the University of Amsterdam.","journey":"Created the Python programming language and worked in software engineering and language development.","skills":"python programming software engineering computer science language design algorithms","milestones":["Studied mathematics and computer science","Created the Python language","Guided Python's early development","Worked in software engineering"],"now":"Check public professional sources for current professional updates.","source":"Python official website","url":"https://gvanrossum.github.io/"},
{"name":"Margaret Hamilton","field":"Software Engineering","speciality":"Space systems · Software reliability","education":"Studied mathematics and philosophy at Earlham College.","journey":"Led software engineering work for NASA's Apollo guidance software and helped advance software engineering practices.","skills":"software engineering programming systems thinking testing mathematics leadership aerospace","milestones":["Studied mathematics","Worked on flight software","Led software engineering for Apollo guidance systems","Helped establish software engineering practices"],"now":"Technology pioneer; consult authoritative biographies for historical details.","source":"NASA","url":"https://www.nasa.gov/"},
{"name":"Jane Goodall","field":"Biology & Conservation","speciality":"Primatology · Field research","education":"Became known for field research on chimpanzees; later earned a Ph.D. from the University of Cambridge.","journey":"Conducted long-term field research in Gombe, Tanzania, and became an internationally recognised conservation advocate.","skills":"biology observation field research conservation ecology communication research","milestones":["Began field research in Gombe","Studied chimpanzee behaviour over decades","Earned a Ph.D. from Cambridge","Became a conservation advocate"],"now":"Historical figure; consult the Jane Goodall Institute for organisational information.","source":"Jane Goodall Institute","url":"https://janegoodall.org/"}
]
UPDATES = [
{"person":"Fei-Fei Li","kind":"Research","text":"Example category: research publication or public lecture.","url":"https://profiles.stanford.edu/fei-fei-li"},
{"person":"Jane Goodall","kind":"Organisation","text":"Example category: conservation programme announcement.","url":"https://gvanrossum.github.io/"},
{"person":"Guido van Rossum","kind":"Technology","text":"Example category: Python community or language news.","url":"https://gvanrossum.github.io/"}
]

@st.cache_resource
def database():
    con = sqlite3.connect("pathai_data.db", check_same_thread=False)
    con.execute("CREATE TABLE IF NOT EXISTS searches(id INTEGER PRIMARY KEY, goal TEXT, skills TEXT, created TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS saved(id INTEGER PRIMARY KEY, name TEXT UNIQUE, created TEXT)")
    con.commit()
    return con
db = database()

@st.cache_resource
def matcher():
    frame = pd.DataFrame(PEOPLE)
    frame["text"] = frame["field"] + " " + frame["speciality"] + " " + frame["skills"] + " " + frame["journey"]
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(frame["text"])
    return frame, vec, matrix
profiles, vectorizer, matrix = matcher()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_publications(person_name):
    """Fetch recent public scholarly works using OpenAlex."""

    headers = {
        "User-Agent": "PathAI Student Prototype"
    }

    try:
        # Search for the author
        author_response = requests.get(
            "https://api.openalex.org/authors",
            params={
                "search": person_name,
                "per-page": 25
            },
            headers=headers,
            timeout=15
        )

        author_response.raise_for_status()

        authors = author_response.json().get("results", [])

        # Normalise names to handle hyphens and punctuation
        def normalise(name):
            return re.sub(r"[^a-z]", "", name.lower())

        target = normalise(person_name)

        # First try an exact normalised match
        exact = [
            author for author in authors
            if normalise(author.get("display_name", "")) == target
        ]

        if exact:
            author = exact[0]
        else:
            # Try a reliable partial match
            target_words = set(
                re.findall(r"[a-z]+", person_name.lower())
            )

            matches = []

            for candidate in authors:
                candidate_name = candidate.get("display_name", "")
                candidate_words = set(
                    re.findall(r"[a-z]+", candidate_name.lower())
                )

                if target_words and target_words.issubset(candidate_words):
                    matches.append(candidate)

            if not matches:
                return {
                    "status": "not_found",
                    "items": []
                }

            # Prefer the candidate with the most publications
            author = max(
                matches,
                key=lambda x: x.get("works_count", 0)
            )

        # Fetch the author's latest publications
        author_id = author.get("id", "").rstrip("/").split("/")[-1]

        works_response = requests.get(
            "https://api.openalex.org/works",
            params={
                "filter": f"authorships.author.id:{author_id}",
                "sort": "publication_date:desc",
                "per-page": 5
            },
            headers=headers,
            timeout=15
        )

        works_response.raise_for_status()

        works = works_response.json().get("results", [])

        items = []

        for work in works:
            primary = work.get("primary_location") or {}
            source = primary.get("source") or {}

            landing = (
                primary.get("landing_page_url")
                or work.get("doi")
                or work.get("id")
            )

            items.append({
                "title": work.get("title") or "Untitled publication",
                "date": work.get("publication_date") or "Date unavailable",
                "source": source.get("display_name") or "OpenAlex",
                "url": landing,
                "type": work.get("type") or "Research output"
            })

        return {
            "status": "ok",
            "author": author.get("display_name"),
            "items": items
        }

    except requests.RequestException:
        return {
            "status": "error",
            "items": []
        }

    except (ValueError, KeyError, TypeError):
        return {
            "status": "error",
            "items": []
        }
def search(goal, skills):
    q = vectorizer.transform([goal + " " + skills])
    scores = cosine_similarity(q, matrix).flatten()
    out = profiles.copy()
    out["similarity"] = (scores * 100).round(1)
    return out.sort_values("similarity", ascending=False).head(4)

def save(name):
    db.execute("INSERT OR IGNORE INTO saved(name, created) VALUES (?,?)", (name, datetime.now().strftime("%Y-%m-%d")))
    db.commit()

def saved():
    return pd.read_sql_query("SELECT name, created FROM saved ORDER BY id DESC", db)

st.markdown("""
<style>
:root {
  --ink: #382b32;
  --muted: #806d75;
  --rose: #d96f87;
  --rose-dark: #b84e6b;
  --peach: #f7c7b8;
  --cream: #fff9f7;
  --line: #f0d9d7;
}
.stApp { background: #fff9f7; color: var(--ink); }
.block-container { max-width: 1180px; padding: 2.2rem 2.5rem 3rem; }
[data-testid="stHeader"] { background: rgba(255,249,247,.92); }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#f7c7b8 0%,#f4b9c4 100%); border-right: 1px solid #edc2c1; }
[data-testid="stSidebar"] * { color: #49333b !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { font-weight: 600; }
[data-testid="stSidebar"] hr { border-color: rgba(73,51,59,.2); }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #765761 !important; }
h1,h2,h3,h4,p,li,label,.stMarkdown,.stText,.stCaption { color: var(--ink); }
.hero { padding: 2.7rem 3rem; border-radius: 28px; background: linear-gradient(115deg,#f5b9ae 0%,#ed9fb1 55%,#d9829c 100%); color: #432e38; margin: .3rem 0 2rem; box-shadow: 0 12px 30px rgba(191,111,129,.15); }
.hero h1 { font-size: 2.85rem; color: #432e38; line-height: 1.08; margin: .25rem 0 .8rem; letter-spacing: -.04em; }
.hero p { color: #5b3b47; font-size: 1.03rem; }
.hero p:first-child { font-size: .76rem; font-weight: 800; letter-spacing: .15em; }
[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); padding: 1rem 1.1rem; border-radius: 18px; box-shadow: 0 5px 18px rgba(120,72,84,.05); }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: #a94f69 !important; }
.stButton > button, .stFormSubmitButton > button, .stLinkButton > a { border-radius: 12px; border: 1px solid #d9869c; background: #d9869c; color: white !important; font-weight: 700; padding: .55rem 1rem; }
.stButton > button:hover, .stFormSubmitButton > button:hover { background: #bf6681; border-color: #bf6681; color: white !important; }
.stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] > div { background: #fff !important; color: #382b32 !important; border: 1px solid #e8c9ca !important; border-radius: 11px !important; }
.stTextInput label, .stSelectbox label { color: #59434c !important; font-weight: 650; }
[data-testid="stExpander"] {
  background: #fff !important;
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
  overflow: hidden;
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:focus {
  background: #fff !important;
  color: #382b32 !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary svg {
  color: #382b32 !important;
  fill: #382b32 !important;
  opacity: 1 !important;
}
[data-testid="stExpander"] summary {
  font-weight: 700 !important;
  min-height: 2.8rem;
}

.profile-card {
  background: #fff;
  border: 1px solid #f0d9d7;
  border-radius: 20px;
  padding: 1.15rem 1.25rem;
  margin: .35rem 0 1rem;
  box-shadow: 0 7px 22px rgba(120,72,84,.055);
}
.profile-kicker {
  color: #a94f69;
  font-size: .72rem;
  font-weight: 800;
  letter-spacing: .12em;
  text-transform: uppercase;
}
.profile-name {
  color: #382b32;
  font-size: 1.35rem;
  font-weight: 800;
  margin: .25rem 0 .15rem;
}
.profile-meta { color: #806d75; font-size: .92rem; }
.skill-pill {
  display: inline-block;
  background: #fff0f1;
  color: #8f425b;
  border: 1px solid #f0d2d6;
  padding: .25rem .6rem;
  border-radius: 999px;
  font-size: .78rem;
  margin: .15rem .2rem .1rem 0;
}


/* High-contrast peach callouts; avoid Streamlit's low-contrast default info banner. */
.pathai-note { display:block; width:100%; box-sizing:border-box;
  background:#fff0f1 !important; color:#382b32 !important;
  border:1px solid #e9c4cc; border-left:5px solid #c96f88;
  border-radius:14px; padding:.85rem 1rem; margin:.45rem 0 .9rem;
  font-size:.94rem; line-height:1.55; font-weight:550; }
.pathai-note * { color:#382b32 !important; opacity:1 !important; }
[data-testid="stAlert"], [data-testid="stAlert"] *, div[role="alert"], div[role="alert"] * {
  color:#382b32 !important; opacity:1 !important; }
[data-testid="stAlert"] { background:#fff0f1 !important;
  border:1px solid #e9c4cc !important; border-radius:14px !important; }

[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
hr { border-color: var(--line); }
.stCaption { color: var(--muted) !important; }
@media (max-width: 700px) { .block-container { padding: 1.2rem 1rem 2rem; } .hero { padding: 1.8rem; } .hero h1 { font-size: 2.15rem; } }
</style>
""", unsafe_allow_html=True)

def show_note(message):
    safe_message = html.escape(str(message))
    st.markdown(f"<div class='pathai-note'>{safe_message}</div>", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("# ♡ PathAI")
    st.caption("Your next chapter starts here.")
    page = st.radio("EXPLORE", ["Home", "Discover people", "Career roadmap", "Updates", "Saved & history"])
    st.divider()
    st.caption("STUDENT PROJECT · PROTOTYPE")

st.markdown("<div class='hero'><p>CAREER DISCOVERY, WITH DIRECTION</p><h1>Find your path.<br>Grow into it.</h1><p>Explore inspiring professional journeys, discover skills to build, and turn your ambition into practical next steps.</p></div>", unsafe_allow_html=True)

if page == "Home":
    left, right = st.columns([1.2,.8], gap="large")
    with left:
        st.subheader("Start with your ambition")
        with st.form("home"):
            goal = st.text_input("What career or field interests you?", placeholder="e.g. AI researcher, aerospace engineer")
            skills = st.text_input("What skills do you already have?", placeholder="e.g. Python, mathematics")
            go = st.form_submit_button("✨ Discover my path", type="primary", use_container_width=True)
        if go:
            if not goal.strip(): st.warning("Enter a career goal first.")
            else:
                st.session_state.goal, st.session_state.skills = goal.strip(), skills.strip()
                st.session_state.matches = search(goal.strip(), skills.strip())
                db.execute("INSERT INTO searches(goal,skills,created) VALUES(?,?,?)",(goal.strip(),skills.strip(),datetime.now().isoformat(timespec="minutes")))
                db.commit()
                st.success("Matches ready! Open Discover people or Career roadmap.")
    with right:
        st.subheader("How PathAI works")
        st.markdown("**01 · Discover** — find professionals connected to your interests.\\n\\n**02 · Understand** — explore education and career milestones.\\n\\n**03 · Plan** — identify skills to develop and practical next steps.")
        show_note("Prototype dataset only. No LinkedIn scraping or live notifications.")
    x,y,z=st.columns(3)
    x.metric("Example profiles",len(PEOPLE)); y.metric("Matching method","NLP / TF-IDF"); z.metric("Saved profiles",len(saved()))
    if "matches" in st.session_state:
        st.subheader("Your first matches")
        for _,p in st.session_state.matches.iterrows():
            with st.container(border=True):
                a,b=st.columns([4,1])
                a.markdown(f"**{p['name']}** · {p['field']}\\n\\n{p['journey']}")
                b.metric("Text similarity",f"{p['similarity']}%")
                if st.button(f"Save {p['name']}",key="home_"+p["name"]):
                    save(p["name"]); st.toast("Profile saved")

elif page == "Discover people":
    st.subheader("People whose journeys can inspire yours")
    st.caption("Explore curated example profiles matched to your interests. Similarity is based on text, not a measure of achievement.")
    with st.form("discover"):
        goal = st.text_input("Career / field", value=st.session_state.get("goal", "scientist"))
        skills = st.text_input("Your skills", value=st.session_state.get("skills", ""))
        go = st.form_submit_button("Find professionals", type="primary")
    if go and goal.strip():
        st.session_state.goal, st.session_state.skills = goal.strip(), skills.strip()
        st.session_state.matches = search(goal.strip(), skills.strip())
        db.execute(
            "INSERT INTO searches(goal,skills,created) VALUES(?,?,?)",
            (goal.strip(), skills.strip(), datetime.now().isoformat(timespec="minutes"))
        )
        db.commit()

    if "matches" in st.session_state:
        results = st.session_state.matches.copy()
        fields = ["All fields"] + sorted(results["field"].dropna().unique().tolist())
        chosen_field = st.selectbox("Filter by field", fields)
        if chosen_field != "All fields":
            results = results[results["field"] == chosen_field]

        st.markdown(f"**{len(results)} professionals shown**")
        for _, p in results.iterrows():
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(f"<div class='profile-kicker'>{p['field']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='profile-name'>{p['name']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='profile-meta'>{p['speciality']}</div>", unsafe_allow_html=True)
                    st.write(p["journey"])
                    pills = "".join(
                        f"<span class='skill-pill'>{re.sub(r'[^A-Za-z0-9+#.-]', '', skill).title()}</span>"
                        for skill in p["skills"].split()[:6]
                    )
                    st.markdown(pills, unsafe_allow_html=True)
                with right:
                    st.metric("Text match", f"{p['similarity']}%")
                    st.link_button("View source ↗", p["url"], use_container_width=True)
                    if st.button("♡ Save", key="discover_save_" + p["name"], use_container_width=True):
                        save(p["name"])
                        st.toast(f"{p['name']} saved")
                with st.expander("Explore career journey"):
                    st.markdown("**Education**")
                    st.write(p["education"])
                    st.markdown("**Key milestones**")
                    for item in p["milestones"]:
                        st.markdown(f"• {item}")
                    st.markdown("**Current / historical context**")
                    st.write(p["now"])
                    st.caption(f"Reference: {p['source']} · Verify details using the source.")
    else:
        st.info("Enter a career or field above to discover relevant professional journeys.")

elif page == "Career roadmap":
    st.subheader("Build your next steps")
    if "matches" not in st.session_state: st.info("Search for a career first on Home or Discover people.")
    else:
        results=st.session_state.matches
        name=st.selectbox("Choose a professional journey",results["name"].tolist())
        p=results[results.name==name].iloc[0]
        st.markdown(f"### A learning plan inspired by {name}")
        st.write("**Your goal:**",st.session_state.get("goal",""))
        current={x.strip().lower() for x in re.split(r"[,;]",st.session_state.get("skills","")) if x.strip()}
        gaps = [x.strip() for x in re.split(r"[;,]", p["skills"]) if x.strip() and x.strip().lower() not in current]
        st.markdown("#### Skills to explore")
        if gaps:
            cols=st.columns(3)
            for i,s in enumerate(gaps[:9]): cols[i%3].markdown("<div class='pathai-note'>" + html.escape(s.title()) + "</div>", unsafe_allow_html=True)
        else: st.success("Your entered skills overlap with this profile's listed skills.")
        st.markdown("#### Suggested action plan")
        for i,(title,desc) in enumerate([
            ("Build foundations","Strengthen maths and subject fundamentals relevant to your field."),
            ("Learn one new skill","Choose one suggested skill and complete a beginner course."),
            ("Create evidence","Build a small project, experiment, article or portfolio piece."),
            ("Connect thoughtfully","Read professional work, attend talks and ask informed questions."),
            ("Review your direction","Revisit your interests and update your learning plan periodically.")
        ],1):
            with st.container(border=True): st.markdown(f"**{i:02d} · {title}**"); st.write(desc)
        st.caption("General guidance inspired by a sample profile, not a guaranteed career route.")

elif page == "Updates":
    st.subheader("Your professional radar")
    st.markdown("Find recent scholarly publications indexed by **OpenAlex**, an open catalogue of research works and authors.")
    followed = saved()["name"].tolist()
    if not followed:
        show_note("Save a professional in Discover people first. Then you can look for their public research publications here.")
    else:
        st.markdown("**Your saved professionals**")
        st.write(" · ".join(followed))
        if st.button("↻ Check public research updates", type="primary"):
            st.session_state["run_public_updates"] = True

        if st.session_state.get("run_public_updates", False):
            st.caption("Checking OpenAlex. Results are retrieved live when you click; they are not automatic notifications.")
            for person in followed:
                with st.container(border=True):
                    st.markdown(f"**{person}**")
                    with st.spinner(f"Searching public research records for {person}..."):
                        result = fetch_publications(person)
                    if result["status"] == "not_found":
                        show_note("No exact author-name match was found in OpenAlex. This person may not have an indexed author record.")
                    elif result["status"] == "error":
                        st.warning("Could not reach OpenAlex right now. Please try again later.")
                    elif not result["items"]:
                        show_note("An author record was found, but no recent works were returned.")
                    else:
                        st.caption(f"Author record matched: {result['author']} · Source: OpenAlex")
                        for item in result["items"]:
                            st.markdown(f"**{item['title']}**")
                            st.caption(f"{item['date']} · {item['type']} · {item['source']}")
                            if item["url"]:
                                st.link_button("Open publication ↗", item["url"])
    st.divider()
    st.caption("Scope note: this feature searches public scholarly records only. It does not access private LinkedIn data, scrape LinkedIn, or guarantee coverage of every professional update. OpenAlex results can be incomplete; verify publication details at the linked source.")

else:
    st.subheader("Your collection")
    st.markdown("### Saved professionals")
    s=saved()
    if s.empty: st.info("No saved profiles yet. Visit Discover people to save one.")
    else: st.dataframe(s,use_container_width=True,hide_index=True)
    st.markdown("### Recent searches")
    h=pd.read_sql_query("SELECT goal,skills,created FROM searches ORDER BY id DESC LIMIT 15",db)
    if h.empty: st.info("Your searches will appear here.")
    else: st.dataframe(h,use_container_width=True,hide_index=True)

st.divider()
st.caption("PathAI student prototype · TF-IDF + cosine similarity ranks text similarity, not career success. Curated demo data; verify facts via sources. No private LinkedIn access, scraping or live alerts.")
