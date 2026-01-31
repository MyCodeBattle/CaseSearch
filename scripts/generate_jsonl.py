import os
import sys
import json
import argparse
from pathlib import Path

SYSTEM_PROMPT = """# Role 
你是一个专业的证券合规数据分析师。你的任务是阅读《中国证监会行政处罚决定书》，并生成一份专注于**违规事实**的案件摘要。 
 
# Goal 
这份摘要将用于向量检索，目的是根据违法情节快速定位案件。**请完全忽略处罚结果（罚款、禁入等），只保留"谁、做了什么、怎么做的"。** 
 
# Instructions 
请提取以下关键要素，生成一段高密度的摘要： 
 
1. **文书标识**：文号。如找不到文号，填写"无"。 
2. **当事人**：核心违规主体（公司或个人）。 
3. **违规行为**：法律定性（如：内幕交易、财务造假、操纵市场等）。 
4. **核心违法事实（关键）**： 
    - **保留实体**：必须保留涉及的所有**股票名称/代码**、**关键时间点**（年份/月份）、**关联公司名称**。 
    - **描述手段**：简述违规的具体操作方式（例如：如何虚增营收、如何控制账户组、何时获取内幕信息等）。 
    - **【强制】去数值化**：**删除所有涉及的具体金额、百分比、股份数量、账户个数、虚增比例等数字信息。** 仅保留事实逻辑。 
    - **【强制】排版规则**： 
      - **多项违规**：若涉及多个独立事实，请使用序号（1. 2.）分点概括。 
      - **单项违规**：若仅有一个违规事实，**不要加序号**，直接撰写。 
    - **开头总结**：在每个事实段落的开头，用一句话精炼总结违法性质。 
 
# Constraints 
- **字数控制**：控制在 **100-300字** 之间。 
- **排除项**：**严禁包含"证监会认为"、"申辩意见"、"依据第X条"以及"罚没款金额/市场禁入"等内容。** 
- **去数值化约束**：禁止出现诸如"95623.04万元"、"20%"等具体数字，请用"巨额款项"、"重大比例"、"相应金额"代替或直接省略。 
- **风格**：陈述句，极简、精准。 
 
# Output Example (Single Fact Case) 
【文号】中国证券监督管理委员会广东监管局〔2024〕12号 
【当事人】爱康科技；邹某慧（实控人） 
【违规行为】未按规定披露关联担保（重大遗漏） 
【事实摘要】 
**未披露关联担保致年报重大遗漏：** 2021年至2023年，爱康科技为邹某慧控制的苏州慧昊、爱康实业等关联方提供担保，未按规定在相应定期报告中披露。 
 
# Output Example (Multi-Fact Case) 
【文号】中国证券监督管理委员会浙江监管局〔2025〕1号 
【当事人】诺泰生物；赵德中（实控人） 
【违规行为】年报虚假记载、欺诈发行 
【事实摘要】 
**1.通过资金闭环虚增业绩：** 2021年12月，诺泰生物向关联方"浙江华贝"转让药品技术并确认收入。浙江华贝既无支付能力也无生产能力，其支付款项实际来源于诺泰生物同期向其缴纳的增资款。该业务缺乏商业实质，导致2021年年报虚增营业收入及利润。 
**2.公开发行文件造假：** 2023年12月，诺泰生物发行可转换公司债券。其披露的《募集说明书》引用了上述包含虚假记载的财务数据，构成在公开发行文件中编造重大虚假内容。"""


def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
        except Exception as e:
            print(f"警告: 无法读取文件 {file_path}: {e}")
            return None
    except Exception as e:
        print(f"警告: 无法读取文件 {file_path}: {e}")
        return None


def generate_batch_requests(input_dir, output_file):
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"错误: 目录不存在: {input_dir}")
        sys.exit(1)
    
    if not input_path.is_dir():
        print(f"错误: 路径不是目录: {input_dir}")
        sys.exit(1)
    
    files = list(input_path.iterdir())
    files = [f for f in files if f.is_file()]
    
    if not files:
        print(f"警告: 目录中没有找到文件: {input_dir}")
        sys.exit(0)
    
    print(f"找到 {len(files)} 个文件待处理...")
    
    batch_requests = []
    skipped_count = 0
    
    for idx, file_path in enumerate(files, start=1):
        content = read_file_content(file_path)
        
        if content is None:
            skipped_count += 1
            continue
        
        if not content.strip():
            print(f"警告: 跳过空文件 {file_path.name}")
            skipped_count += 1
            continue
            
        # 尝试从文件名提取ID (假设格式为: 数字_...)
        file_id = None
        try:
            file_parts = file_path.name.split('_')
            if file_parts and file_parts[0].isdigit():
                file_id = file_parts[0]
        except Exception:
            pass
            
        # 如果无法提取，使用默认的序号
        custom_id = f"request-{file_id}" if file_id else f"request-idx-{idx}"
        
        request = {
            "custom_id": custom_id,
            "body": {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "temperature": 0.1,
                "thinking": {
                    "type": "enabled"
                }
            }
        }
        
        batch_requests.append(request)
        print(f"处理中 [{idx}/{len(files)}]: {file_path.name} -> {custom_id}")
    
    print(f"\n成功生成 {len(batch_requests)} 个请求")
    if skipped_count > 0:
        print(f"跳过 {skipped_count} 个文件")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=False) + '\n')
    
    print(f"\nJSONL文件已生成: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='生成火山方舟批量推理所需的JSONL文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python generate_jsonl.py --input-dir ./decisions
  python generate_jsonl.py --input-dir ./decisions --output batch_requests.jsonl
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='包含行政处罚决定书的目录路径'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='batch_requests.jsonl',
        help='输出的JSONL文件路径 (默认: batch_requests.jsonl)'
    )
    
    args = parser.parse_args()
    
    generate_batch_requests(args.input_dir, args.output)


if __name__ == '__main__':
    main()
