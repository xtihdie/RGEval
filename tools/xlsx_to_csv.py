import os
import pandas as pd


def convert_xlsx_to_csv(folder):
    """
    将指定文件夹内所有 .xlsx 文件转换为 .csv
    保留文件名：example.xlsx → example.csv
    """
    for filename in os.listdir(folder):
        if filename.endswith(".xlsx"):
            xlsx_path = os.path.join(folder, filename)
            csv_path = os.path.join(folder, filename.replace(".xlsx", ".csv"))

            try:
                df = pd.read_excel(xlsx_path)
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                print(f"转换成功：{filename} → {os.path.basename(csv_path)}")
            except Exception as e:
                print(f"转换失败：{filename}, 错误：{e}")


if __name__ == "__main__":
    origin_folder = "../data/300class/origin"  # 你的 origin 路径
    convert_xlsx_to_csv(origin_folder)
