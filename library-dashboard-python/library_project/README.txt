Library Analytics Dashboard (Python / Flask)
1. pip install -r requirements.txt
2. python app.py
3. Open http://127.0.0.1:5000

Pages: Dashboard, Students, Records, Recommend, Reports & Screenshots, Data Upload.
Upload your own .csv / .xlsx on the "Data Upload" page. Needed columns: Student_ID, Book_Title, Issue_Date
(optional: Transaction_ID, Department, Book_ID, Category, Due_Date, Return_Date). Column names are auto-detected
(e.g. "Roll No", "Title", "Borrow Date"); if not, you can map them by hand. Uploaded files are saved in data/.
Files: app.py (Flask API), analysis.py (Pandas/NumPy/Matplotlib/Seaborn), generate_data.py (sample data).
