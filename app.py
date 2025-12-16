import os
import pandas as pd
import json
import textwrap
from dotenv import load_dotenv 
from google import genai
from google.genai import types
from google.genai.errors import APIError
from flask import Flask, render_template, request, flash, redirect, url_for, send_file
from werkzeug.utils import secure_filename

# --- Initialization and Configuration ---
load_dotenv() 

CSV_FILE = "research_gaps_summary.csv"
MODEL_NAME = "gemini-2.5-flash" 
UPLOAD_FOLDER = 'uploads' 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super_secret_key_for_flash_messages' 

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

try:
    GEMINI_CLIENT = genai.Client()
except Exception:
    GEMINI_CLIENT = None
    
# --- Helper Functions (Load/Save/AI Prompt) ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def _load_data(uploaded_file=None) -> pd.DataFrame:
    """Loads data, ensuring required structure for merging."""
    default_columns = ['Title_Source', 'Paper_Link', 'Date_Analyzed']
    df = pd.DataFrame()

    # Load logic (omitted for brevity, assume it's the same as previous step)
    # ... (loading from upload or CSV_FILE) ...
    
    # Check for existing local file
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
        except Exception:
            pass 
            
    # Load from uploaded file if provided and valid
    if uploaded_file and allowed_file(uploaded_file.filename):
        try:
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)
            df = pd.read_csv(file_path)
            flash(f"Successfully loaded data from uploaded file: {filename}", 'success')
        except Exception as e:
            flash(f"Error loading uploaded CSV: {e}. Falling back to existing/empty.", 'error')


    # Ensure default columns exist
    for col in default_columns:
        if col not in df.columns:
            df[col] = None

    return df

def _save_data(df: pd.DataFrame):
    """Saves the current DataFrame back to the permanent CSV file."""
    df.to_csv(CSV_FILE, index=False)
    
def _prompt_ai_for_analysis(paper_text: str, system_prompt: str) -> dict | None:
    """Sends text to Gemini for structured JSON extraction."""
    
    # Logic remains the same as the previous step
    full_system_prompt = textwrap.dedent(system_prompt).strip()

    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=MODEL_NAME,
            contents=paper_text,
            config=types.GenerateContentConfig(
                system_instruction=full_system_prompt,
                response_mime_type="application/json", 
                temperature=0.0
            )
        )
        json_string = response.text.strip()
        return json.loads(json_string)

    except APIError as e:
        flash(f"Gemini API Error for a paper. Check key/rate limits.", 'error')
        print(f"API Error: {e}")
        return None
    except json.JSONDecodeError:
        flash("AI returned invalid JSON for a paper. Check the prompt structure.", 'error')
        return None
    except Exception as e:
        flash(f"An unexpected error occurred during AI extraction: {e}", 'error')
        return None

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    csv_exists = os.path.exists(CSV_FILE)
    csv_size = round(os.path.getsize(CSV_FILE) / 1024, 2) if csv_exists else 0
    
    if request.method == 'POST':
        if GEMINI_CLIENT is None:
            flash("System Error: Gemini client failed to initialize. Check API key.", 'error')
            return redirect(url_for('index'))

        df = _load_data(request.files.get('csv_file'))
        system_prompt = request.form['system_prompt'].strip()
        
        # --- 1. Collect all dynamic form data ---
        
        # Determine the maximum number used in the form fields
        paper_ids = set()
        for key in request.form:
            if key.startswith('identifier_'):
                try:
                    paper_ids.add(int(key.split('_')[-1]))
                except ValueError:
                    pass
        
        if not paper_ids:
            flash("No paper data was submitted for analysis.", 'error')
            return redirect(url_for('index'))

        # --- 2. Iterate and Process ---
        
        papers_to_process = []
        for i in sorted(list(paper_ids)):
            identifier = request.form.get(f'identifier_{i}', '').strip()
            link = request.form.get(f'link_{i}', '').strip()
            text = request.form.get(f'text_{i}', '').strip()

            if not identifier and not text:
                # If only link is provided but no text/identifier, skip this empty block
                continue 
            
            papers_to_process.append({
                'identifier': identifier,
                'link': link,
                'text': text
            })
            
        if not papers_to_process:
            flash("All submitted forms were empty or incomplete.", 'error')
            return redirect(url_for('index'))

        # --- 3. Batch Analysis Loop ---
        
        new_papers_added = 0
        all_new_rows = []
        
        for i, paper in enumerate(papers_to_process):
            
            # Check for duplicates based on the link (if provided) or identifier
            is_duplicate = False
            if paper['link'] and 'Paper_Link' in df.columns and paper['link'] in df['Paper_Link'].astype(str).values:
                is_duplicate = True
            elif paper['identifier'] and 'Title_Source' in df.columns and paper['identifier'] in df['Title_Source'].astype(str).values:
                 is_duplicate = True

            if is_duplicate:
                flash(f"Paper '{paper['identifier']}' already exists. Skipping analysis.", 'error')
                continue

            flash(f"Analyzing Paper {i+1} of {len(papers_to_process)}: {paper['identifier'][:30]}...", 'info')
            
            extracted_data = _prompt_ai_for_analysis(paper['text'], system_prompt)
            
            if extracted_data is None:
                continue 

            # Prepare New Row
            new_row = {
                'Title_Source': paper['identifier'],
                'Paper_Link': paper['link'] if paper['link'] else "N/A",
                **extracted_data, 
                'Date_Analyzed': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
            }
            
            all_new_rows.append(new_row)
            new_papers_added += 1

        # 4. Final Append and Save
        if all_new_rows:
            new_df = pd.DataFrame(all_new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
            
            # Reorder columns to ensure standard output
            column_order = ['Title_Source', 'Paper_Link', 'Date_Analyzed'] + [c for c in df.columns if c not in ['Title_Source', 'Paper_Link', 'Date_Analyzed']]
            df = df.reindex(columns=column_order)
            
            _save_data(df)
            flash(f"Batch completed! Successfully analyzed and added {new_papers_added} new papers.", 'success')
        else:
            flash("No new papers were successfully analyzed or added to the dataset.", 'error')


        return redirect(url_for('index'))

    return render_template('index.html', csv_exists=csv_exists, csv_size=csv_size)

@app.route('/download')
def download():
    """Route to download the current CSV file."""
    if os.path.exists(CSV_FILE):
        return send_file(CSV_FILE, as_attachment=True, download_name=CSV_FILE, mimetype='text/csv')
    else:
        flash("No CSV file exists to download yet!", 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Use Waitress for production-ready local Windows testing
    try:
        from waitress import serve
        print("Starting Waitress server on http://127.0.0.1:5000/")
        serve(app, host='0.0.0.0', port=5000)
    except ImportError:
        print("Waitress not found. Running simple Flask server. Install 'waitress' for production use.")
        app.run(debug=True)