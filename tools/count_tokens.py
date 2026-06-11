import os
import tiktoken


def count_csv_tokens(directory_path):
    """统计目录中所有CSV文件的token数量，并额外加上150个汉字的token量。"""
    encoding = tiktoken.get_encoding("cl100k_base")
    file_token_counts = {}

    # 预先计算150个汉字对应的token数量（按汉字“一”占位）
    extra_token_count = len(encoding.encode("一" * 150))

    # 遍历目录及其子目录
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.lower().endswith(".csv"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        text_content = f.read()

                    tokens = encoding.encode(text_content)
                    token_count = len(tokens) + extra_token_count

                    file_token_counts[filepath] = token_count

                    print(f"已处理: {filepath} - {token_count} tokens")
                except Exception as e:
                    print(f"处理文件 {filepath} 时出错: {str(e)}")

    total_tokens = sum(file_token_counts.values())
    return total_tokens, file_token_counts


if __name__ == "__main__":
    directory_path = "../data/300class/origin"
    total, breakdown = count_csv_tokens(directory_path)

    print(f"\n总token数量: {total}")
    print(f"\n总token数量: {total / 1000000 * (1 + 6 + 21 + 10)}M")
    print(f"\n总价格: ￥{(total / 1000000) * (1 + 6 + 21 + 10) * 14.6}")
    # print("\n各文件详细统计:")
    # for filename, count in breakdown.items( ):
    #     print(f"{filename}: {count} tokens")
