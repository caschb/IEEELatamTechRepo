"""Print the text outputs of an executed notebook, for eyeballing validation runs."""
import json, sys
for path in sys.argv[1:]:
    print(f"\n########## {path}")
    for c in json.load(open(path))["cells"]:
        if c["cell_type"] != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                print("ERROR", o["ename"], o["evalue"])
            elif "text" in o:
                print("".join(o["text"]).rstrip())
            elif "data" in o and "text/plain" in o["data"]:
                print("".join(o["data"]["text/plain"]).rstrip()[:300])
