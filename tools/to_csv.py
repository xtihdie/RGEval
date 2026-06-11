import pandas as pd
import os


def simple_convert_xlsx_to_csv(xlsx_file_path, csv_file_path=None):
    """
    简单的xlsx转csv函数

    参数:
        xlsx_file_path: xlsx文件路径
        csv_file_path: csv输出路径，如果为None则自动生成
    """
    if csv_file_path is None:
        csv_file_path = xlsx_file_path.replace('.xlsx', '.csv')

    # 读取xlsx文件
    df = pd.read_excel(xlsx_file_path)

    # 保存为csv文件
    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

    print(f"转换完成: {os.path.basename(xlsx_file_path)} → {os.path.basename(csv_file_path)}")


# 使用示例
if __name__ == "__main__":
    import glob

    for xlsx_file in glob.glob("../data/300class/results/*.xlsx"):
        simple_convert_xlsx_to_csv(xlsx_file)
