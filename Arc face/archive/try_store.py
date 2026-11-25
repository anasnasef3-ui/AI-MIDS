from face_engine import FaceEngine
from store import FaceStore

engine = FaceEngine(use_gpu=False)            # slow to load, so create it once
store = FaceStore("aimids.db")   # creates the file if it doesn't exist

# 1) Register a missing person with their photos
pid = store.add_person("Anas", notes="missing since 2015")
for path, age in [("my_photos/anas/childhood.jpeg", 8),
                  ("my_photos/anas/teen.jpeg", 17)]:
    r = engine.embed(path)
    if r is None:
        print("no face found in", path)
        continue
    store.add_photo(pid, r["embedding"], photo_path=path, age=age)

# 2) Search with a new photo
for n, face in enumerate(engine.embed_all("my_photos/anas/portsaid.png"), 1):
    hits = store.search(face["embedding"], k=10)
    print(f"face {n}  position {[int(v) for v in face['bbox']]}  ->", 
          [(h["name"], h["similarity"], h["tier"]) for h in hits] or "no match")
    
for face in engine.embed_all("my_photos/anas/now.png"):   # every face in the photo
    for h in store.search(face["embedding"], k=10, min_sim=0.0):
        print(h["name"], h["similarity"], h["tier"], h["photo_path"])