import json

print("=== train_qa.jsonl 前2条记录 ===")
with open("/home/cw/projects/M-Former/dataset/datasets/train_qa.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 2:
            break
        item = json.loads(line)
        print(json.dumps(item, ensure_ascii=False, indent=2))

print("\n=== 统计信息 ===")
stage_count = {}
total = 0
with open("/home/cw/projects/M-Former/dataset/datasets/train_qa.jsonl") as f:
    for line in f:
        item = json.loads(line)
        total += 1
        for conv in item["conversations"]:
            if conv["from"] == "human" and "stage" in conv:
                s = conv["stage"]
                stage_count[s] = stage_count.get(s, 0) + 1

print(f"train total QA: {total}")
print(f"stage 分布: {stage_count}")

total_test = 0
with open("/home/cw/projects/M-Former/dataset/datasets/test_qa.jsonl") as f:
    for line in f:
        total_test += 1
print(f"test total QA: {total_test}")
print(f"train+test = {total + total_test} (期望 81081)")
