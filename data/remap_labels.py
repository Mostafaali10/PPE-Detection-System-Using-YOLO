from pathlib import Path

DATASET_PATH = Path(r"C:\Users\aa683\Desktop\ppe_detection\data\processed")

# Mapping: القديم class_id → new class_id
# فقط الكلاسات اللي عايزينها
CLASS_MAP = {
    0: 0,   # Hardhat → 0
    5: 1,   # Person → 1
    7: 2,   # Safety Vest → 2
}

for split in ["train", "valid", "test"]:
    labels_dir = DATASET_PATH / split / "labels"
    
    for label_file in labels_dir.glob("*.txt"):
        new_lines = []
        
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                old_id = int(parts[0])
                
                # لو الكلاس ده من اللي عايزينه
                if old_id in CLASS_MAP:
                    new_id = CLASS_MAP[old_id]
                    new_lines.append(f"{new_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
        
        # اكتب الملف من جديد (حتى لو فاضي)
        with open(label_file, "w") as f:
            f.write("\n".join(new_lines))
            if new_lines:
                f.write("\n")
    
    print(f"{split}: done")

print("Remapping finished successfully!")