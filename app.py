import re
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from thefuzz import fuzz
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
# --- 1. INITIALIZATION & DATA LOADING ---

print("Starting server... loading models and data.")

app = Flask(__name__)

#1. INITIALIZE VARIABLES 
df_users = None
df_transactions = None

 #2. Load data files inside the try block
try:
    df_users = pd.read_csv("users.csv")
    df_transactions = pd.read_csv("transactions.csv")
except FileNotFoundError:
    print("ERROR: Make sure 'users.csv' and 'transactions.csv' are in the same directory.")
    exit(1) 

df_users.dropna(subset=['name'], inplace=True)


# --- 2. TASK 2: EMBEDDING PRE-COMPUTATION ---
print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Get the tokenizer from the model to count tokens
tokenizer = model.tokenizer

# Pre-compute embeddings for all transaction descriptions
# Handle potential missing descriptions
df_transactions['description'] = df_transactions['description'].fillna('')

print("Pre-computing all transaction embeddings... (This may take a moment)")
transaction_embeddings = model.encode(df_transactions['description'].tolist())
transaction_ids = df_transactions['id'].tolist()
print("Embeddings are ready.")


# --- 3. TASK 1: HELPER FUNCTIONS ---

def extract_name_from_description(description):
    """
    Extracts a name from a transaction description using regex.
    This is a simple implementation and can be improved.
    """
    if not isinstance(description, str):
        return None

    # Regex to find names between "From" and "for Deel"
    # Handles "From" and "Transfer from"
    match = re.search(r'(?:From|Transfer from)\s+(.*?)\s+(?:for Deel|\\ for Deel)', description, re.IGNORECASE)
    
    if match:
        name = match.group(1)
        
        # Clean up the name: remove common artifacts
        name = re.sub(r'[\.,\\]', '', name) # Remove dots, commas, backslashes
        name = re.sub(r'\s+', ' ', name).strip() # Normalize whitespace
        
        # Handle cases like 'Matthew RogersWest' -> 'Matthew Rogers West'
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        
        return name
    
    return None

# --- 4. TASK 1: API ENDPOINT ---

@app.route('/match_transaction/<string:transaction_id>', methods=['GET'])
def match_transaction(transaction_id):
    
    # 1. Find the transaction
    transaction = df_transactions.loc[df_transactions['id'] == transaction_id]
    
    if transaction.empty:
        return jsonify({"error": "Transaction not found"}), 404
        
    description = transaction.iloc[0]['description']
    
    # 2. Extract name
    extracted_name = extract_name_from_description(description)
    
    if not extracted_name:
        return jsonify({
            "users": [],
            "total_number_of_matches": 0,
            "message": "Could not extract a name from the description."
        })

    # 3. Match against all users with fuzzy matching
    matches = []
    for index, user in df_users.iterrows():
        # Use token_set_ratio for better partial/jumbled name matching
        score = fuzz.token_set_ratio(extracted_name, user['name'])
        
        # Set a threshold for what's considered a "match"
        if score > 70:
            matches.append({
                "id": user['id'],
                "match_metric": score
            })
            
    # 4. Sort results
    sorted_matches = sorted(matches, key=lambda x: x['match_metric'], reverse=True)
    
    # 5. Format and return response
    return jsonify({
        "users": sorted_matches,
        "total_number_of_matches": len(sorted_matches)
    })

# --- 5. TASK 2: API ENDPOINT ---

@app.route('/similar_transactions', methods=['POST'])
def get_similar_transactions():
    
    data = request.get_json()
    
    if not data or 'description' not in data:
        return jsonify({"error": "Missing 'description' in JSON body"}), 400
        
    query_description = data['description']
    
    # 1. Get token count for the input
    # We use encode() as it's the direct way to get token IDs
    query_tokens = tokenizer.encode(query_description)
    token_count = len(query_tokens)
    
    # 2. Generate embedding for the query
    query_embedding = model.encode([query_description])
    
    # 3. Compute cosine similarity
    # Compare the query embedding against all pre-computed transaction embeddings
    cosine_scores = util.cos_sim(query_embedding, transaction_embeddings)[0]
    
    # 4. Combine IDs, scores, and embeddings
    results = []
    for i, score in enumerate(cosine_scores):
        results.append({
            "id": transaction_ids[i],
            "score": float(score),
            "embedding": transaction_embeddings[i].tolist() # Convert numpy array to list
        })
        
    # 5. Sort by score
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # 6. Format response as per PDF
    # Note: The PDF format 'transactions: [{id, embedding , {id, embedding), ...]'
    # appears to have a typo. I am interpreting it as:
    # 'transactions: [{id, embedding}, {id, embedding}, ...]'
    
    # Let's return the top 10 matches
    top_n = 10
    output_transactions = [
        {"id": r['id'], "embedding": r['embedding']} 
        for r in sorted_results[:top_n]
    ]

    # 7. Return final JSON
    return jsonify({
        "transactions": output_transactions,
        "total_number_of_tokens_used": token_count
    })

# --- 6. RUN THE APP ---

if __name__ == '__main__':
    # Setting host='0.0.0.0' makes it accessible on your network
    app.run(debug=True, host='0.0.0.0', port=5000)