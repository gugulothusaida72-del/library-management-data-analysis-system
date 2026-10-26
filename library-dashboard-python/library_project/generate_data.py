"""Builds data/library_records.csv - a sample dataset sized to match the project review deck.
Raw: 2,430 rows (30 duplicates, 25 blank categories, 62 blank return dates). Clean: 2,400 rows."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)

BOOKS = {  # title: (category, issues)
 "Database System Concepts":("Computer Science",160),"Python Crash Course":("Computer Science",151),
 "Data Structures in C":("Computer Science",143),"Clean Code":("Computer Science",130),
 "Computer Networks":("Computer Science",116),"Deep Learning":("Computer Science",115),
 "Operating System Concepts":("Computer Science",114),"Hands-On Machine Learning":("Computer Science",112),
 "Artificial Intelligence: A Modern Approach":("Computer Science",108),"Introduction to Algorithms":("Computer Science",45),
 "Software Engineering":("Computer Science",37),
 "1984":("Fiction",106),"To Kill a Mockingbird":("Fiction",105),"Pride and Prejudice":("Fiction",104),
 "The Great Gatsby":("Fiction",101),"Harry Potter":("Fiction",100),
 "Thermodynamics":("Engineering",50),"Engineering Mechanics":("Engineering",48),"Fluid Mechanics":("Engineering",46),
 "Circuit Analysis":("Engineering",45),"Strength of Materials":("Engineering",44),"Control Systems":("Engineering",42),
 "A Brief History of Time":("Science",35),"Physics Fundamentals":("Science",33),"Organic Chemistry":("Science",32),
 "Biology Essentials":("Science",30),"Engineering Mathematics":("Science",30),
 "Principles of Management":("Management",40),"Marketing Management":("Management",37),
 "Financial Accounting":("Management",35),"Business Economics":("Management",33),
 "Indian History":("History",20),"World War II":("History",19),"Ancient Civilizations":("History",17),"Modern India":("History",17)}
MONTHS = [141,186,233,209,88,62,202,259,260,296,275,189]
DEPTS = {"AIML":(804,40,{"S009":67,"S069":54}),"CSE":(525,26,{"S077":59,"S101":51}),"ECE":(409,20,{"S008":48}),
         "MECH":(254,13,{"S004":46}),"EEE":(223,11,{}),"CIVIL":(185,10,{})}

def split(total, n, lo=4, hi=45):
    while True:
        w = rng.dirichlet(np.ones(n)*6); c = np.maximum(lo, np.round(w*total)).astype(int)
        c[np.argmax(c)] += total - c.sum()
        if c.sum()==total and c.min()>=lo and c.max()<=hi: return c

# students per department with exact totals
students, sdept, free_ids = [], {}, iter([f"S{i:03d}" for i in range(1,121)])
fixed = {s for d in DEPTS.values() for s in d[2]}
pool = [s for s in (f"S{i:03d}" for i in range(1,121)) if s not in fixed]; rng.shuffle(pool)
counts = {}
for dep,(tot,n,fx) in DEPTS.items():
    rest = split(tot-sum(fx.values()), n-len(fx))
    for s,c in fx.items(): counts[s]=c; sdept[s]=dep
    for c in rest: s=pool.pop(); counts[s]=int(c); sdept[s]=dep
student_col = np.array([s for s,c in counts.items() for _ in range(c)]); rng.shuffle(student_col)

titles = np.array([t for t,(c,n) in BOOKS.items() for _ in range(n)]); rng.shuffle(titles)
months = np.array([m+1 for m,n in enumerate(MONTHS) for _ in range(n)]); rng.shuffle(months)
N = 2400; ids = {t:f"B{i+1:03d}" for i,t in enumerate(BOOKS)}
days = [rng.integers(1, pd.Timestamp(2025,m,1).days_in_month+1) for m in months]
issue = pd.to_datetime([f"2025-{m:02d}-{d:02d}" for m,d in zip(months,days)])
due = issue + pd.Timedelta(days=14)

# returns: 60 pending, 840 late (sum of late days = 3421 -> Rs.6,842 at Rs.2/day), rest on time
kind = np.array(["pending"]*60+["late"]*840+["ok"]*1500); rng.shuffle(kind)
late = np.ones(840,int); 
while late.sum()<3421: late[rng.integers(0,840)]+=1 if late.max()<20 else 0
ok = rng.integers(3,15,1500)
while ok.sum()!=12899:
    i=rng.integers(0,1500)
    if ok.sum()<12899 and ok[i]<14: ok[i]+=1
    elif ok.sum()>12899 and ok[i]>1: ok[i]-=1

held = np.zeros(N); li=oi=0
for i,k in enumerate(kind):
    if k=="late": held[i]=14+late[li]; li+=1
    elif k=="ok": held[i]=ok[oi]; oi+=1
ret = np.where(kind=="pending", pd.NaT, (issue + pd.to_timedelta(held,unit="D")).values)

df = pd.DataFrame({"Transaction_ID":[f"T{i+1:04d}" for i in range(N)],"Student_ID":student_col,
  "Department":[sdept[s] for s in student_col],"Book_ID":[ids[t] for t in titles],"Book_Title":titles,
  "Category":[BOOKS[t][0] for t in titles],"Issue_Date":issue.strftime("%Y-%m-%d"),
  "Due_Date":due.strftime("%Y-%m-%d"),"Return_Date":pd.Series(ret).dt.strftime("%Y-%m-%d")})
# add dirt: 25 blank categories, 30 duplicate rows (incl. 2 pending -> 62 blank return dates)
pend = df.index[df.Return_Date.isna()]; other = df.index[df.Return_Date.notna()]
dup_idx = list(rng.choice(pend,2,replace=False))+list(rng.choice(other,28,replace=False))
df.loc[rng.choice(sorted(set(range(N))-set(dup_idx)),25,replace=False),"Category"] = np.nan
raw = pd.concat([df, df.loc[dup_idx]]).sample(frac=1, random_state=1).reset_index(drop=True)
raw.to_csv("data/library_records.csv", index=False)
print("raw", raw.shape, "blank cat", raw.Category.isna().sum(), "blank return", raw.Return_Date.isna().sum(), "dups", raw.duplicated().sum())
