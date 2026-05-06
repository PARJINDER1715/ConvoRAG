# ConvoRAG — Setup Guide for VS Code

## What you need
- Python 3.8+ installed
- VS Code installed
- Your `conversations.csv` file
- An groq API key 

---

## Step 1 — Open the project in VS Code

1. Create a folder on your computer, e.g. `ConvoRAG`
2. Copy these files into it:
   - `app.py`
   - `requirements.txt`
   - `conversations.csv`  ← your CSV file
   - `static/index.html`  ← put this inside a folder called `static`

Your folder should look like this:
```
ConvoRAG/
├── app.py
├── requirements.txt
├── conversations.csv
└── static/
    └── index.html
```

3. Open VS Code
4. Click **File → Open Folder** → select your `ConvoRAG` folder

---

## Step 2 — Open the Terminal in VS Code

Press **Ctrl + `** (backtick) to open the terminal
OR go to **Terminal → New Terminal** from the menu

---

## Step 3 — Create a virtual environment

In the terminal, type these commands one by one and press Enter:

**Windows:**
```
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of the terminal line. That means it worked.

---

## Step 4 — Install dependencies

```
pip install -r requirements.txt
```

Wait for everything to install (takes 1-2 minutes).

---

## Step 5 — Set your API key in .env file.

## Step 6 — Run the app

```
python app.py
```

You will see:
```
Loading CSV: conversations.csv
Parsed 191592 messages across 11001 days
Ready: 80 topic segments, 20 100-msg chunks
🚀 ConvoRAG running at http://localhost:5000
```

---

## Step 7 — Open in browser

Open your browser and go to:
```
http://localhost:5000
```

The app is now running! 🎉

---

## How to use the app

1. **RAG System tab** → Click "Process Conversations" first
   - Claude reads all segments, detects topics, creates summaries
   - Takes a few minutes (calls AI for each segment)
   - Then use the search box to ask questions

2. **Persona tab** → Click "Extract Persona"
   - Builds a profile of User 1 (habits, personality, interests, etc.)

3. **Chatbot tab** → Ask anything about the user
   - Works best AFTER running both RAG and Persona

---

## Stopping the app

Press **Ctrl + C** in the terminal to stop the server.

## Running again next time

Just do Steps 3 (activate venv) + 5 (set API key) + 6 (run) again.
You do NOT need to install requirements again.
