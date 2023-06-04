from sentence_transformers import SentenceTransformer

model = SentenceTransformer('saved_models/all-mpnet-base-v2')
print(model)