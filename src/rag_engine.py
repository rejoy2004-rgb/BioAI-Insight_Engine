import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Read PDF
doc = fitz.open("papers/sample.pdf")
text = ""

for page in doc:
    text += page.get_text()

# Step 2: Split into chunks
chunks = [text[i:i+500] for i in range(0, len(text), 500)]

# Step 3: Convert text to embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# Step 4: Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Step 5: Ask a question
query = "What is the main finding of this research?"
query_embedding = model.encode([query])

# Step 6: Search similar chunks
D, I = index.search(np.array(query_embedding), 3)

print("Most Relevant Sections:\n")

for i in I[0]:
    print(chunks[i])
    print("-----------")