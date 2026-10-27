
from pathlib import Path
import json

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "data_raw"

print("Script started")
print("Data directory:", DATA_DIR)

if not DATA_DIR.exists():
    print("ERROR: data/data_raw folder not found.")
    input("Press Enter to exit...")
    raise SystemExit(1)

files = sorted(DATA_DIR.glob("*.json"))
print("JSON files found:", len(files))

for path in files:
    print("\n" + "=" * 50)
    print("File:", path.name)

    try:
        text = path.read_text(encoding="utf-8-sig")

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]

        print("Top-level type:", type(data).__name__)

        if isinstance(data, list):
            print("Record count:", len(data))
            first_record = data[0] if data else None
        elif isinstance(data, dict):
            print("Top-level keys:", list(data.keys()))
            list_fields = [
                (key, value)
                for key, value in data.items()
                if isinstance(value, list)
            ]
            for key, value in list_fields:
                print("List field:", key, "count:", len(value))

            first_record = (
                list_fields[0][1][0]
                if list_fields and list_fields[0][1]
                else data
            )
        else:
            first_record = data

        print("First record:")
        print(json.dumps(first_record, ensure_ascii=False, indent=2)[:3000])

    except Exception as exc:
        print("ERROR:", repr(exc))

print("\nScript finished.")
input("Press Enter to exit...")
