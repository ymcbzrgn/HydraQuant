import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), 'user_data', 'scripts'))
from colbert_reranker import ColBERTReranker

def main():
    model = ColBERTReranker()
    scores = model.rerank("Is bitcoin bullish?", [{"content": "Bitcoin breaks $100k, bullish market.", "id": "1"}, {"content": "Federal reserve increases interest rates.", "id": "2"}])
    for doc in scores:
        print(f"Doc: {doc['id']}, Score: {doc.get('colbert_score')}")

if __name__ == '__main__':
    main()
