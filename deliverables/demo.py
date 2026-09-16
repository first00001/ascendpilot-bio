import argparse
import json
import os
from skill.plantcell_skill import run

parser = argparse.ArgumentParser()
parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
parser.add_argument("--api-token", default=os.getenv("PLANTCELL_API_TOKEN"))
parser.add_argument("--query", default="分析水稻干旱胁迫下的细胞类型和差异基因")
args = parser.parse_args()
print(
    json.dumps(
        run(args.query, args.endpoint, api_token=args.api_token),
        ensure_ascii=False,
        indent=2,
    )
)
