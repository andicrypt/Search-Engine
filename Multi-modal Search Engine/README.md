# Multi-modal Search Engine

Search image by a text query, also support adding more query gradually to enhance the output results

**Used libraries** FAISS, CLIP

**DEMO**: for demo, I sample 200 images from database only, these images will be encoded by CLIP and indexed by FAISS
Notice that I encode image and add it to FAISS indexer during running time; therefore, it is slow. Furthermore, I dont fine-tune the CLIP encoder so it is not good enough

**VIDEO**: Youtube https://www.youtube.com/watch?v=1puviIAW4Sg 