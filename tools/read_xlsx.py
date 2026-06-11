import os
import pandas as pd

from rubric import RubricNode
from LLMs.agent import LLMAgent
from prompts import (
    score,
)

class_file_path = "../data/300class/origin"


def read_file_by_code(file_code, file_path_root="file_path"):
    """
    根据编号读取对应文件（支持.xlsx或.csv）
    :param file_path_root: 基础路径（如"./data/300class/origin"）
    :param file_code: 文件编号（如"2615"）
    :return: pandas.DataFrame 或 None（未找到时）
    """
    # 遍历目标路径下所有文件
    for filename in os.listdir(file_path_root):
        # 匹配文件名开头的编号（如"2615_xxx.xlsx"）
        if filename.startswith(f"{file_code}_"):
            file_path = os.path.join(file_path_root, filename)

            # 根据扩展名选择读取方式
            if filename.endswith(".xlsx"):
                return pd.read_excel(file_path)
            elif filename.endswith(".csv"):
                return pd.read_csv(file_path)

    print(f"未找到编号为 {file_code} 的文件")
    return None


def get_lesson_by_id(lesson_id, df):
    """根据课例id获取特定课例数据"""
    result = df[df['课例id'] == int(lesson_id)]
    if result.empty:
        print(f"未找到课例id为 {lesson_id} 的记录")
        return None

    # 转换为字典并处理NaN值
    lesson_data = result.iloc[0].to_dict( )
    return {k: (v if pd.notna(v) else "") for k, v in lesson_data.items( )}


if __name__ == "__main__":
    class_file_path = "../data/300class/origin"
    score_file_path = "../data/300class/score.xlsx"
    class_file_code = "12248"
    class_file_code_ = "12243"
    label = '观点表达类提问'

    data = read_file_by_code(class_file_code, class_file_path)
    data_ = read_file_by_code(class_file_code_, class_file_path)

    print(f"成功读取文件（编号: {class_file_code}）:")
    data_str = []
    data_str_ = []

    for index, row in data.iterrows( ):
        category = "" if pd.isna(row["类别"]) else row["类别"]
        data_str.append(f"{row['角色']}：{row['内容']}（{category}）")

    for index, row in data_.iterrows( ):
        category = "" if pd.isna(row["类别"]) else row["类别"]
        data_str_.append(f"{row['角色']}：{row['内容']}（{category}）")

    data_str = "\n\t".join(data_str)
    data_str = "\n\t" + data_str
    data_str_ = "\n\t".join(data_str_)
    data_str_ = "\n\t" + data_str_
    print(data_str)

    score_list = pd.read_excel(score_file_path)

    lesson = get_lesson_by_id(class_file_code, score_list)
    print(lesson)
    print(lesson[label])
    system_prompt = score['system'].format(tem_class_conv=data_str,
                                           tem_score_trait=label,
                                           tem_class_score=lesson[label])
    print(system_prompt)
    user_prompt = score['user'].format(class_conv=data_str_,
                                       score_trait=label)
    print(user_prompt)

    agent = LLMAgent(system_prompt=system_prompt, model_name="glm-z1-air")
    response = agent.query(prompt=user_prompt)
    print(response)
    # if lesson:
    #     print("\n查询结果：")
    #     for key, value in lesson.items():
    #         print(f"{key}: {value}")
