DOMAIN_NAME=$1
INPUT_DIR=database/amazon_review/$DOMAIN_NAME/jsonl_docs
INDEX_DIR=database/amazon_review/$DOMAIN_NAME/pyserini_index

python -m pyserini.index.lucene -collection JsonCollection \
 -input $INPUT_DIR \
 -index $INDEX_DIR \
 -generator DefaultLuceneDocumentGenerator \
 -threads 4 \
 -storePositions -storeDocvectors -storeRaw
