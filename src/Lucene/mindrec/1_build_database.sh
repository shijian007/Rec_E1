INPUT_DIR=./data/mindsmall/mindsmall/jsonl_docs
INDEX_DIR=./data/mindsmall/mindsmall/pyserini_index

python -m pyserini.index.lucene -collection JsonCollection \
 -input $INPUT_DIR \
 -index $INDEX_DIR \
 -generator DefaultLuceneDocumentGenerator \
 -threads 4 \
 -storePositions -storeDocvectors -storeRaw
