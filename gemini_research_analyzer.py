import os
import pandas as pd
import json
import textwrap
from dotenv import load_dotenv 
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- Load the .env file immediately at the start of the script ---
load_dotenv() 

# --- DIAGNOSTIC CHECK: ADD THESE 4 LINES ---
api_key_check = os.getenv("GEMINI_API_KEY")
if not api_key_check or api_key_check.startswith("YOUR_GEMINI_API_KEY_HERE"):
    print("\n[CRITICAL WARNING] API KEY IS MISSING OR GENERIC IN .env FILE.")
    print("Please check your .env file in this directory.")
# --- END DIAGNOSTIC CHECK ---

# --- Configuration ---
CSV_FILE = "research_gaps_summary.csv"
# Using the specified Gemini model for speed and efficiency
MODEL_NAME = "gemini-2.5-flash" 
# ---------------------

class ResearchAssistant:
    """
    A class to read research paper abstracts/texts, use the Gemini AI model 
    to extract structured data, and manage a self-updating CSV file.
    """
    
    def __init__(self):
        """Initializes the Gemini client and checks for API key."""
        try:
            # Client automatically reads GEMINI_API_KEY environment variable
            self.client = genai.Client()
        except Exception as e:
            print("Error initializing Gemini client.")
            print("Please ensure your GEMINI_API_KEY environment variable is set correctly.")
            print(f"Details: {e}")
            raise

        self.df = self._load_data()
        print(f"Gemini Assistant initialized. Currently {len(self.df)} papers analyzed.")

    def _load_data(self):
        """Loads the existing CSV file or creates an empty DataFrame."""
        if os.path.exists(CSV_FILE):
            print(f"Loading existing data from {CSV_FILE}...")
            return pd.read_csv(CSV_FILE)
        else:
            print(f"No existing CSV found. Starting new DataFrame.")
            # Define the columns the AI will populate
            return pd.DataFrame(columns=[
                "Paper_Title_Source", 
                "Key_Findings", 
                "Methodology", 
                "Research_Gap_Identified", 
                "Future_Work_Suggested",
                "Date_Analyzed"
            ])

    def _save_data(self):
        """Saves the current DataFrame back to the CSV file."""
        self.df.to_csv(CSV_FILE, index=False)
        print(f"\n--- Data successfully saved to {CSV_FILE}. ---")
        print(f"Total papers analyzed now: {len(self.df)}")
        
    def _prompt_ai_for_analysis(self, paper_text: str) -> dict | None:
        """
        Sends the text to the Gemini model for structured JSON extraction.
        
        Args:
            paper_text: The full text (or abstract/introduction) of the paper.
        
        Returns:
            A dictionary containing the extracted data fields, or None on error.
        """
        
        # We define the required output structure in the system prompt AND 
        # enforce JSON output using the API configuration for maximum reliability.
        system_prompt = textwrap.dedent(f"""
            You are a highly efficient academic research assistant. Your task is to
            analyze the provided research text and extract specific structured information 
            crucial for a literature review.

            You MUST respond only with a single JSON object. The JSON object must 
            strictly adhere to the following structure and keys. Do not add 
            any introductory or explanatory text outside the JSON:

            KEYS REQUIRED:
            - Paper_Title_Source: (The title or source ID of the paper)
            - Key_Findings: (Summarize the main results and contributions. Max 2 sentences.)
            - Methodology: (Briefly state the research approach, e.g., 'Qualitative Case Study', 'Deep Learning Model'.)
            - Research_Gap_Identified: (What specific gap did this paper fill, or what future gap did the authors explicitly mention? CRUCIAL for gap analysis.)
            - Future_Work_Suggested: (Specific next steps the authors suggest for follow-up research.)
        """).strip()

        try:
            print(f"-> Analyzing text of size: {len(paper_text)} using {MODEL_NAME}...")
            
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=paper_text,
                config=types.GenerateContentConfig(
                    # 1. Instruct the model on its role
                    system_instruction=system_prompt,
                    # 2. Force the model to output valid JSON
                    response_mime_type="application/json", 
                    temperature=0.0 # Force deterministic output
                )
            )

            # Gemini response text should be the raw JSON string
            json_string = response.text.strip()
            
            return json.loads(json_string)

        except APIError as e:
            print(f"A Gemini API error occurred: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"AI returned invalid JSON. Error: {e}")
            print(f"Raw AI Output: {response.text[:200]}...")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during AI extraction: {e}")
            return None

    def analyze_new_paper(self, title_or_source: str, paper_text: str):
        """
        Processes a new paper's text and updates the internal DataFrame and CSV.
        """
        
        # 1. Check if the paper has already been analyzed
        if title_or_source in self.df['Paper_Title_Source'].values:
            print(f"--- '{title_or_source}' already exists in the CSV. Skipping. ---")
            return

        print(f"\n--- Analyzing New Paper: {title_or_source} ---")
        
        # 2. Get structured data from AI
        extracted_data = self._prompt_ai_for_analysis(paper_text)
        
        if extracted_data is None:
            print(f"Skipping '{title_or_source}' due to extraction error.")
            return
            
        # 3. Add control data and format
        extracted_data['Paper_Title_Source'] = title_or_source # Use the reliable identifier
        extracted_data['Date_Analyzed'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
        
        # 4. Convert to a DataFrame row
        new_row_df = pd.DataFrame([extracted_data])
        
        # 5. Append (Self-Update Feature)
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)
        
        # 6. Save the updated CSV
        self._save_data()
        print(f"Successfully added: {title_or_source}")


# --- 3. Example Usage ---

def run_analysis_session():
    """Demonstrates how to use the ResearchAssistant."""
    
    # Initialize the assistant (loads or creates the CSV)
    assistant = ResearchAssistant()

    # --- Paper 1: Hypothetical Text (Copy-paste your content here) ---
    paper1_text = textwrap.dedent("""
        Title: The Impact of Climate Change on Coastal Fisheries in Southeast Asia
        Abstract: This study investigated the economic and ecological impact of rising 
        sea temperatures on three key coastal fish species across five Southeast Asian 
        nations. Data collected over ten years shows a clear correlation between 
        temperature anomalies and significant drops in catch volume (average 22%). 
        Current models often fail to integrate local socioeconomic factors (e.g., 
        small-scale fishing methods) with large-scale climate dynamics, representing 
        a major research gap. Our findings suggest that localized resource management 
        strategies are ineffective against global warming unless coupled with resilient 
        aquaculture adaptation programs. Future research must specifically design and 
        test region-specific financial aid models for fishermen transitioning to 
        sustainable practices.
    """).strip()
    
    assistant.analyze_new_paper("Pham et al. 2024 - Fisheries", paper1_text)


    # --- Paper 2: Another Hypothetical Text (Showing the 'Gap' more clearly) ---
    paper2_text = textwrap.dedent("""
        Title: Novel Explainable AI (XAI) for Fraud Detection in Digital Payments
        Introduction: While deep learning models achieve high accuracy in fraud detection, 
        their black-box nature makes them unsuitable for regulated financial environments 
        where explainability is legally mandated. The primary research gap addressed 
        is the lack of real-time, lightweight XAI methods applicable to high-volume 
        transaction streams. We propose a new Shapley value approximation technique 
        integrated directly into a recurrent neural network (RNN). Conclusion: Our 
        model, XAI-RNN, successfully reduced the false positive rate by 15% while 
        providing feature importance explanations in under 10 milliseconds. This 
        methodology overcomes the latency constraint prevalent in current XAI systems. 
        Further research should explore the scalability of this method to blockchain 
        transaction analysis.
    """).strip()

    assistant.analyze_new_paper("Murali & Singh 2023 - XAI", paper2_text)
    
    # --- Paper 3: Duplicate Test (Should be skipped automatically) ---
    assistant.analyze_new_paper("Pham et al. 2024 - Fisheries", "This is dummy text.") 
    
    
if __name__ == "__main__":
    print(textwrap.dedent("""
        ======================================================================
        STARTING GEMINI RESEARCH GAP ANALYSIS TOOL
        ----------------------------------------------------------------------
        Ensure your GEMINI_API_KEY environment variable is set.
        The output CSV will be named: research_gaps_summary.csv
        ======================================================================
    """))
    run_analysis_session()