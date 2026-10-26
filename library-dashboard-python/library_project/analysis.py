"""Library data analysis engine: flexible loading, cleaning, Pandas analysis, Matplotlib/Seaborn plots."""
import io, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))
NAVY, ORANGE = "#1F3B73", "#F26A0F"
COLORS = [ORANGE, NAVY, "#4F8FC0", "#F2AE35", "#7A9E7E", "#A05A8C", "#C0504D", "#8E7CC3"]
REQUIRED = ["Student_ID", "Book_Title", "Issue_Date"]
FIELDS = ["Transaction_ID", "Student_ID", "Department", "Book_ID", "Book_Title", "Category", "Issue_Date", "Due_Date", "Return_Date"]
ALIASES = {
 "Transaction_ID": "transactionid txnid issueid id recordid tid srno slno",
 "Student_ID": "studentid studentno rollno rollnumber memberid userid borrowerid student regno admissionno",
 "Department": "department dept branch course stream",
 "Book_ID": "bookid bookcode isbn accessionno accno bookno",
 "Book_Title": "booktitle title bookname book nameofbook",
 "Category": "category genre subject bookcategory section",
 "Issue_Date": "issuedate issueddate borrowdate dateissued issuedon issued checkout checkoutdate borrowed",
 "Due_Date": "duedate returndue expectedreturn expectedreturndate dateofreturndue",
 "Return_Date": "returndate returneddate datereturned returnedon actualreturn actualreturndate checkin"}
sns.set_theme(style="whitegrid", font="DejaVu Sans")
_n = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())

def detect(cols):
    m, used = {}, set()
    for c in FIELDS:
        for col in cols:
            if col not in used and (_n(col) == _n(c) or _n(col) in ALIASES[c].split()):
                m[c] = col; used.add(col); break
    return m

def _dates(s):
    a = pd.to_datetime(s, errors="coerce", format="mixed"); b = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=True)
    return b if b.isna().sum() < a.isna().sum() else a

def normalize(raw, mapping=None):
    """Cleaning + preprocessing (methodology modules 3-4). Returns (clean_df, report)."""
    mapping = {k: v for k, v in (mapping or detect(list(raw.columns))).items() if v in raw.columns}
    miss = [c for c in REQUIRED if c not in mapping]
    if miss: raise ValueError("Missing required columns: " + ", ".join(miss))
    d = pd.DataFrame({c: raw[s] for c, s in mapping.items()})
    rep = {"raw_rows": len(raw), "notes": [], "mapping": mapping}
    d = d.drop_duplicates(); rep["duplicates"] = rep["raw_rows"] - len(d)
    for c in ["Issue_Date", "Due_Date", "Return_Date"]:
        if c in d: d[c] = _dates(d[c])
    n = len(d); d = d.dropna(subset=REQUIRED); rep["invalid"] = n - len(d)
    d["Student_ID"] = d.Student_ID.astype(str).str.strip(); d["Book_Title"] = d.Book_Title.astype(str).str.strip()
    if "Book_ID" not in d: d["Book_ID"] = d.Book_Title; rep["notes"].append("No Book_ID column: book titles used as IDs.")
    if "Category" not in d: d["Category"] = np.nan; rep["notes"].append("No Category column: shown as 'Uncategorized'.")
    nb = int(d.Category.isna().sum())
    cmap = d.dropna(subset=["Category"]).drop_duplicates("Book_ID").set_index("Book_ID")["Category"]
    d["Category"] = d.Category.fillna(d.Book_ID.map(cmap)); rep["filled_category"] = nb - int(d.Category.isna().sum())
    d["Category"] = d.Category.fillna("Uncategorized").astype(str)
    if "Department" not in d: d["Department"] = "Unknown"; rep["notes"].append("No Department column: shown as 'Unknown'.")
    d["Department"] = d.Department.fillna("Unknown").astype(str)
    if "Return_Date" not in d: d["Return_Date"] = pd.NaT; rep["notes"].append("No Return_Date column: all books treated as still issued.")
    if "Due_Date" not in d: d["Due_Date"] = pd.NaT; rep["notes"].append("No Due_Date column: due date assumed as issue date + 14 days.")
    d["Due_Date"] = d.Due_Date.fillna(d.Issue_Date + pd.Timedelta(days=14))
    if "Transaction_ID" not in d: d["Transaction_ID"] = [f"T{i+1:05d}" for i in range(len(d))]
    d["Status"] = np.where(d.Return_Date.isna(), "Issued", "Returned")
    d["Days_Held"] = (d.Return_Date - d.Issue_Date).dt.days
    d["Late_Days"] = (d.Return_Date - d.Due_Date).dt.days.clip(lower=0)
    d["Year"] = d.Issue_Date.dt.year
    rep["clean_rows"] = len(d); rep["missing_return"] = int(d.Return_Date.isna().sum())
    return d.reset_index(drop=True), rep

def apply_filters(df, dept="", cat="", year=""):
    if dept: df = df[df.Department == dept]
    if cat: df = df[df.Category == cat]
    if year: df = df[df.Year == int(year)]
    return df

def analyse(df, fine=2):
    """Groupby / aggregation (modules 5-6)."""
    if df.empty: return {"empty": True}
    ret = df[df.Status == "Returned"]; late = ret[ret.Late_Days > 0]; ref = df.Issue_Date.max()
    top = df.groupby("Book_Title").size().nlargest(10)
    cat = df.groupby("Category").size().sort_values(ascending=False)
    dept = df.groupby("Department").size().sort_values(ascending=False)
    per = df.Issue_Date.dt.to_period("M"); mon = per.value_counts().reindex(pd.period_range(per.min(), per.max()), fill_value=0)
    multi = per.min().year != per.max().year
    stu = df.groupby("Student_ID").agg(Books=("Transaction_ID", "count"), Dept=("Department", "first"),
        Avg_Days=("Days_Held", "mean"), Late=("Late_Days", lambda s: int((s > 0).sum())), Fine=("Late_Days", "sum")).sort_values("Books", ascending=False).head(15)
    pend = df[(df.Status == "Issued") & (df.Due_Date < ref)]
    return {"empty": False,
     "kpi": {"transactions": len(df), "students": df.Student_ID.nunique(), "books": df.Book_ID.nunique(), "categories": df.Category.nunique(),
             "returned": len(ret), "pending": int((df.Status == "Issued").sum()), "overdue": len(pend),
             "avg_days": round(float(ret.Days_Held.mean()), 1) if len(ret) else 0, "late": len(late),
             "late_pct": round(len(late) / len(ret) * 100, 1) if len(ret) else 0, "fine": int(late.Late_Days.sum() * fine),
             "top10_share": round(top.sum() / len(df) * 100, 1), "avg_per_student": round(len(df) / df.Student_ID.nunique(), 1),
             "from": str(df.Issue_Date.min().date()), "to": str(ref.date())},
     "top": [[k, int(v)] for k, v in top.items()], "cat": [[k, int(v)] for k, v in cat.items()], "dept": [[k, int(v)] for k, v in dept.items()],
     "mon": [[p.strftime("%b %y" if multi else "%b"), int(v)] for p, v in mon.items()],
     "status": [["Returned", len(ret)], ["Still issued", int((df.Status == "Issued").sum())]],
     "students": [[i, int(r.Books), r.Dept, None if pd.isna(r.Avg_Days) else round(float(r.Avg_Days), 1), r.Late, int(r.Fine * fine)] for i, r in stu.iterrows()],
     "titles": sorted(df.Book_Title.unique().tolist()),
     "records": [[r.Transaction_ID, r.Student_ID, r.Department, r.Book_Title, r.Category, r.Issue_Date.strftime("%Y-%m-%d"), r.Due_Date.strftime("%Y-%m-%d"),
                  "" if pd.isna(r.Return_Date) else r.Return_Date.strftime("%Y-%m-%d"), r.Status, int(r.Late_Days) if r.Status == "Returned" else 0]
                 for r in df.sort_values("Issue_Date", ascending=False).head(1500).itertuples()]}

def recommend(df, book, n=5):
    ids = df[df.Book_Title == book].Student_ID.unique()
    c = df[df.Student_ID.isin(ids) & (df.Book_Title != book)].groupby("Book_Title").Student_ID.nunique().nlargest(n)
    return {"readers": int(len(ids)), "items": [{"title": t, "students": int(v)} for t, v in c.items()]}

# ---------- Matplotlib / Seaborn report charts ----------
def _bar(ax, r):
    t = r["top"][::-1]; ax.barh([x[0][:30] for x in t], [x[1] for x in t], color=ORANGE)
    for y, x in enumerate(t): ax.text(x[1] * 1.01, y, x[1], va="center", fontsize=8)
    ax.set_title("Most Borrowed Books"); ax.set_xlabel("Times issued")
def _pie(ax, r):
    ax.pie([x[1] for x in r["cat"]], labels=[x[0] for x in r["cat"]], autopct="%1.0f%%", colors=COLORS, textprops={"fontsize": 8}); ax.set_title("Category Share")
def _line(ax, r):
    y = [x[1] for x in r["mon"]]; ax.plot([x[0] for x in r["mon"]], y, marker="o", color=NAVY, lw=2.2, ms=6)
    ax.set_title("Monthly Borrow Trend"); ax.set_ylabel("Books issued"); plt.setp(ax.get_xticklabels(), rotation=45 if len(y) > 12 else 0)

def plot(r, name):
    fig = None
    if name == "bar": fig, ax = plt.subplots(figsize=(7, 4.5)); _bar(ax, r)
    elif name == "pie": fig, ax = plt.subplots(figsize=(6, 4.5)); _pie(ax, r)
    elif name == "line": fig, ax = plt.subplots(figsize=(10, 4)); _line(ax, r)
    elif name == "dept":
        fig, ax = plt.subplots(figsize=(7, 4)); sns.barplot(x=[x[0] for x in r["dept"]], y=[x[1] for x in r["dept"]], hue=[x[0] for x in r["dept"]], palette="Blues_r", legend=False, ax=ax); ax.set_title("Books Issued by Department")
    elif name == "status":
        fig, ax = plt.subplots(figsize=(5, 4)); ax.pie([x[1] for x in r["status"]], labels=[x[0] for x in r["status"]], autopct="%1.1f%%", colors=[NAVY, ORANGE], startangle=90, wedgeprops=dict(width=.45)); ax.set_title("Returned vs Still Issued")
    elif name == "combined":
        fig, axs = plt.subplots(1, 3, figsize=(16.5, 5)); _bar(axs[0], r); _pie(axs[1], r); _line(axs[2], r); fig.tight_layout()
    if fig is None: return None
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=130, bbox_inches="tight"); plt.close(fig); b.seek(0); return b
