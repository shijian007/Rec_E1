INPUT_DIR=/opt/data2/xiezhijun/code/Rec-R1/database/rl_rec/emrec_0401_2k/jsonl_docs
INDEX_DIR=/opt/data2/xiezhijun/code/Rec-R1/database/rl_rec/emrec_0401_2k/pyserini_index

python -m pyserini.index.lucene -collection JsonCollection \
 -input $INPUT_DIR \
 -index $INDEX_DIR \
 -generator DefaultLuceneDocumentGenerator \
 -threads 4 \
 -storePositions -storeDocvectors -storeRaw
